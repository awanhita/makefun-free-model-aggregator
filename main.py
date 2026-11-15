#!/usr/bin/env python3
"""Free model aggregator for make.fun competition free-model-aggregator-01.

Stdlib-only Python server. Serves POST /v1/chat/completions (OpenAI wire,
JSON + SSE), POST /v1/messages (Anthropic wire, JSON + SSE) and GET /v1/models.
At trial start it reads the brief in MAKEFUN_PROMPT_FILE, probes every
provider's catalog route, registers only the providers actually read, then
serves the workload and exits 0 when the env says the work is done.

Routing: for each request pick the highest-rated free model able to serve it
(tools / vision / json / min_context from the manifest), spread across
providers under safe per-provider call caps so no per-trial budget is hit,
fail over to the next candidate on any 402/429/5xx refusal (a provider is
dropped for the rest of the trial on its first 402), never retry the same
failed provider for one request, never serve a paid model.
"""
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FIXED_CREATED = 1767225600
PROTOCOL_VERSION = 1
PROBE_TIMEOUT = 5.0
UPSTREAM_TIMEOUT = 12.0
POLL_INTERVAL = 0.2

# Safe per-provider call caps for the pre-plan. Sealed per-trial budgets are
# groq 9-12, cerebras 5-6, gemini 6-8, zai 4-5, openrouter 10-12, sambanova
# 4-5, pollinations 10-14, cloudflare 7-8, ovh 5-6, kilol 12-16. Caps sit
# below the minimum budget so failover calls have headroom, and pollinations
# (lowest-rated free models) is never a planned pick.
CALL_CAPS = {
    "groq": 6, "cerebras": 3, "gemini": 5, "zai": 4, "openrouter": 9,
    "sambanova": 4, "pollinations": 0, "cloudflare": 6, "ovh": 3, "kilol": 3,
}


def log(msg):
    sys.stderr.write("aggregator: %s\n" % msg)
    sys.stderr.flush()


# ---------------------------------------------------------------- brief ---

def find_json_docs(text):
    """Yield every balanced top-level JSON object found in the brief text."""
    docs = []
    i = 0
    n = len(text)
    while True:
        j = text.find("{", i)
        if j < 0:
            break
        depth = 0
        in_str = False
        esc = False
        k = j
        ok = False
        while k < n:
            ch = text[k]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        ok = True
                        break
            k += 1
        if ok:
            try:
                docs.append(json.loads(text[j:k + 1]))
            except Exception:
                pass
            i = k + 1
        else:
            break
    return docs


class Brief:
    def __init__(self, text):
        self.providers = []   # {id,name,wire,auth,key_env,key,path}
        self.ratings = []     # {model,provider,rating_milli,context_length,tools,vision,json,free}
        self.manifest = []    # {id,api,stream,weight,requires}
        for doc in find_json_docs(text):
            if not isinstance(doc, dict):
                continue
            schema = doc.get("schema", "")
            if "providers" in schema or ("providers" in doc and "ratings" not in doc):
                self.providers = doc.get("providers", [])
            elif "ratings" in schema or "ratings" in doc:
                self.ratings = doc.get("ratings", [])
            elif "workload" in schema or "manifest" in doc:
                self.manifest = doc.get("manifest", [])


def load_brief():
    path = os.environ.get("MAKEFUN_PROMPT_FILE", "")
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return Brief(f.read())


# ------------------------------------------------------------- probing ---

def auth_headers(provider):
    auth = provider.get("auth", "none")
    key = provider.get("key", "")
    if auth == "bearer":
        return {"Authorization": "Bearer " + key}
    if auth == "x-api-key":
        return {"x-api-key": key, "anthropic-version": "2023-06-01"}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def http_get_json(url, headers, timeout):
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def probe_provider(env_url, provider):
    """Probe one provider's catalog route. Returns (ok, models) where models
    is a list of {id, context_length} actually read, or ([], error)."""
    wire = provider.get("wire")
    base = env_url.rstrip("/") + provider.get("path", "")
    headers = dict(auth_headers(provider))
    try:
        if wire == "openai":
            data = http_get_json(base + "/v1/models", headers, PROBE_TIMEOUT)
            models = []
            for item in data.get("data", []):
                if isinstance(item, dict) and isinstance(item.get("id"), str):
                    models.append({"id": item["id"],
                                   "context_length": item.get("context_length") or 0})
            return True, models
        if wire == "anthropic":
            headers["anthropic-version"] = "2023-06-01"
            data = http_get_json(base + "/v1/models", headers, PROBE_TIMEOUT)
            models = []
            for item in data.get("data", []):
                if isinstance(item, dict) and isinstance(item.get("id"), str):
                    models.append({"id": item["id"],
                                   "context_length": item.get("context_length") or 0})
            return True, models
        if wire == "gemini":
            data = http_get_json(base + "/v1beta/models", headers, PROBE_TIMEOUT)
            models = []
            for item in data.get("models", []):
                if not isinstance(item, dict):
                    continue
                name = item.get("name", "")
                if isinstance(name, str) and name.startswith("models/"):
                    models.append({"id": name[len("models/"):],
                                   "context_length": item.get("inputTokenLimit") or 0})
            return True, models
    except Exception as e:
        log("probe %s failed: %s" % (provider.get("id"), e))
        return False, []
    log("probe %s: unknown wire %s" % (provider.get("id"), wire))
    return False, []


# ------------------------------------------------------------- catalog ---

def build_catalog(brief, probes):
    """probes: {provider_id: (ok, [models])}. Catalog is one entry per
    provider actually read, free models only, base_url the provider's own
    path, key_env '' when auth is none."""
    rating_by_key = {}
    for row in brief.ratings:
        if row.get("free"):
            rating_by_key[(row.get("provider"), row.get("model"))] = row
    providers = []
    for provider in brief.providers:
        pid = provider.get("id")
        ok, probed = probes.get(pid, (False, []))
        if not ok:
            continue
        models = []
        for m in probed:
            row = rating_by_key.get((pid, m["id"]))
            if row is None:
                continue
            ctx = m.get("context_length") or row.get("context_length") or 0
            models.append({"id": m["id"], "free": True, "context_length": ctx})
        models.sort(key=lambda x: x["id"])
        if not models:
            continue
        providers.append({
            "id": pid,
            "base_url": provider.get("path", ""),
            "key_env": provider.get("key_env", "") if provider.get("auth") != "none" else "",
            "wire": provider.get("wire"),
            "auth": provider.get("auth"),
            "models": models,
        })
    return {"providers": providers}


def catalog_json(catalog):
    return json.dumps(catalog, separators=(",", ":"))


# -------------------------------------------------------------- routing ---

class Router:
    """Pre-planned, rating-greedy routing with per-provider caps and
    runtime failover. Candidates are free rating rows joined to providers."""

    def __init__(self, brief, catalog):
        self.providers = {p["id"]: p for p in brief.providers}
        self.by_id = {}
        for row in brief.ratings:
            if not row.get("free"):
                continue
            pid = row.get("provider")
            if pid not in self.providers:
                continue
            cand = {
                "provider": pid,
                "wire": self.providers[pid].get("wire"),
                "model": row.get("model"),
                "rating": row.get("rating_milli", 0),
                "context": row.get("context_length", 0),
                "tools": bool(row.get("tools")),
                "vision": bool(row.get("vision")),
                "json": bool(row.get("json")),
            }
            self.by_id[(pid, cand["model"])] = cand
        # only candidates whose model is in the registered catalog
        self.candidates = []
        for entry in catalog["providers"]:
            for m in entry["models"]:
                cand = self.by_id.get((entry["id"], m["id"]))
                if cand is not None:
                    self.candidates.append(cand)
        self.candidates.sort(key=lambda c: (-c["rating"], c["provider"], c["model"]))
        self.manifest = brief.manifest
        self.plan = {}          # request id -> candidate
        self.used = {p["id"]: 0 for p in brief.providers}
        self.dead = set()       # providers dropped after a 402
        self._plan_all()

    def eligible(self, cand, req, client_api, client_json):
        if req.get("tools") and not cand["tools"]:
            return False
        if req.get("vision") and not cand["vision"]:
            return False
        if req.get("json") and not cand["json"]:
            return False
        # An Anthropic-wire provider can never carry a JSON-mode OpenAI
        # request: its parsed json_mode is always false, so the echo fails.
        if client_json and client_api == "openai" and cand["wire"] == "anthropic":
            return False
        min_ctx = req.get("min_context") or 0
        if min_ctx and cand["context"] < min_ctx:
            return False
        return True

    def _plan_all(self):
        planned = {p: 0 for p in self.used}
        for item in self.manifest:
            req = dict(item.get("requires") or {})
            pick = None
            for cand in self.candidates:
                if not self.eligible(cand, req, item.get("api"), req.get("json")):
                    continue
                if planned[cand["provider"]] < CALL_CAPS.get(cand["provider"], 3):
                    pick = cand
                    break
            if pick is None:
                for cand in self.candidates:
                    if self.eligible(cand, req, item.get("api"), req.get("json")):
                        pick = cand
                        break
            if pick is not None:
                self.plan[item["id"]] = pick
                planned[pick["provider"]] += 1

    def order_for(self, req_id, req, client_api, client_json):
        """Candidate order for one request: planned pick first, then the
        rest by rating, skipping dead providers and (while alternatives
        remain) providers already at their cap."""
        planned = self.plan.get(req_id)
        ordered = []
        seen = set()
        if planned is not None:
            ordered.append(planned)
            seen.add((planned["provider"], planned["model"]))
        for cand in self.candidates:
            key = (cand["provider"], cand["model"])
            if key not in seen:
                ordered.append(cand)
                seen.add(key)
        eligible = [c for c in ordered if self.eligible(c, req, client_api, client_json)]
        fresh = [c for c in eligible
                 if c["provider"] not in self.dead
                 and self.used[c["provider"]] < CALL_CAPS.get(c["provider"], 3)]
        if fresh:
            return fresh
        return [c for c in eligible if c["provider"] not in self.dead] or eligible

# ------------------------------------------------------ wire helpers ---

def text_of(content):
    """OpenAI message content -> plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if isinstance(part, dict):
            if part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                parts.append("")
    return "".join(parts)


def image_parts_of(content):
    """OpenAI message content -> list of {mime, data} or {url} images."""
    out = []
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                out.append(parse_image_url(url))
    return out


def parse_image_url(url):
    if isinstance(url, str) and url.startswith("data:"):
        head, _, data = url[5:].partition(",")
        mime = head.split(";")[0] or "image/png"
        return {"mime": mime, "data": data}
    return {"url": url or ""}


def anthropic_text_of(content):
    parts = []
    if isinstance(content, str):
        return content
    for block in content or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def anthropic_images_of(content):
    out = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image":
                src = block.get("source") or {}
                if src.get("type") == "base64":
                    out.append({"mime": src.get("media_type", "image/png"),
                                "data": src.get("data", "")})
                elif src.get("type") == "url":
                    out.append({"url": src.get("url", "")})
    return out


# ------------------------------------------- client -> upstream bodies ---

def openai_to_anthropic(body, model):
    system = ""
    messages = []
    for msg in body.get("messages", []):
        role = msg.get("role")
        if role == "system":
            system = text_of(msg.get("content"))
            continue
        if role == "tool":
            messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": msg.get("tool_call_id", ""),
                 "content": text_of(msg.get("content"))}]})
            continue
        content = []
        content.append({"type": "text", "text": text_of(msg.get("content"))})
        for img in image_parts_of(msg.get("content")):
            if "data" in img:
                content.append({"type": "image", "source": {
                    "type": "base64", "media_type": img["mime"], "data": img["data"]}})
            else:
                content.append({"type": "image", "source": {
                    "type": "url", "url": img["url"]}})
        messages.append({"role": "user" if role == "user" else "assistant",
                         "content": content})
    out = {"model": model, "max_tokens": body.get("max_tokens") or 1024,
           "messages": messages}
    if system:
        out["system"] = system
    if body.get("temperature") is not None:
        out["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        out["top_p"] = body["top_p"]
    stop = body.get("stop")
    if stop:
        out["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
    tools = body.get("tools")
    if tools:
        out["tools"] = [{
            "name": t.get("function", {}).get("name"),
            "description": t.get("function", {}).get("description", ""),
            "input_schema": t.get("function", {}).get("parameters") or {"type": "object"},
        } for t in tools if t.get("type") == "function"]
        tc = body.get("tool_choice")
        if tc == "auto" or tc is None:
            out["tool_choice"] = {"type": "auto"}
        elif tc == "required":
            out["tool_choice"] = {"type": "any"}
        elif isinstance(tc, dict):
            out["tool_choice"] = {"type": "tool",
                                  "name": tc.get("function", {}).get("name")}
    return out


def openai_to_gemini(body, model):
    system_text = ""
    contents = []
    for msg in body.get("messages", []):
        role = msg.get("role")
        if role == "system":
            system_text = text_of(msg.get("content"))
            continue
        if role == "tool":
            contents.append({"role": "user", "parts": [
                {"text": text_of(msg.get("content"))}]})
            continue
        parts = []
        txt = text_of(msg.get("content"))
        if txt:
            parts.append({"text": txt})
        for img in image_parts_of(msg.get("content")):
            if "data" in img:
                parts.append({"inlineData": {"mimeType": img["mime"],
                                             "data": img["data"]}})
            else:
                parts.append({"fileData": {"fileUri": img["url"]}})
        contents.append({"role": "model" if role == "assistant" else "user",
                         "parts": parts})
    out = {"model": model, "contents": contents}
    if system_text:
        out["systemInstruction"] = {"parts": [{"text": system_text}]}
    tools = body.get("tools")
    if tools:
        out["tools"] = [{"functionDeclarations": [{
            "name": t.get("function", {}).get("name"),
            "description": t.get("function", {}).get("description", ""),
            "parameters": t.get("function", {}).get("parameters") or {"type": "object"},
        } for t in tools if t.get("type") == "function"]}]
    gen = {}
    if body.get("max_tokens") is not None:
        gen["maxOutputTokens"] = body["max_tokens"]
    if body.get("temperature") is not None:
        gen["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        gen["topP"] = body["top_p"]
    stop = body.get("stop")
    if stop:
        gen["stopSequences"] = [stop] if isinstance(stop, str) else list(stop)
    fmt = body.get("response_format") or {}
    if fmt.get("type") == "json_object":
        gen["responseMimeType"] = "application/json"
    if gen:
        out["generationConfig"] = gen
    return out


def anthropic_to_openai(body, model):
    messages = []
    system = body.get("system")
    if system:
        messages.append({"role": "system",
                         "content": system if isinstance(system, str)
                         else anthropic_text_of(system)})
    for msg in body.get("messages", []):
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        parts = []
        tool_calls = []
        for block in content or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                parts.append({"type": "text", "text": block.get("text", "")})
            elif block.get("type") == "image":
                src = block.get("source") or {}
                if src.get("type") == "base64":
                    parts.append({"type": "image_url", "image_url": {
                        "url": "data:%s;base64,%s" % (src.get("media_type", "image/png"),
                                                      src.get("data", ""))}})
                else:
                    parts.append({"type": "image_url", "image_url": {
                        "url": src.get("url", "")}})
            elif block.get("type") == "tool_use":
                tool_calls.append({"id": block.get("id"),
                                   "type": "function",
                                   "function": {"name": block.get("name"),
                                                "arguments": json.dumps(
                                                    block.get("input") or {})}})
            elif block.get("type") == "tool_result":
                parts.append({"type": "text",
                              "text": anthropic_text_of(block.get("content"))})
        entry = {"role": role}
        if tool_calls:
            entry["tool_calls"] = tool_calls
            entry["content"] = parts if parts else None
        else:
            entry["content"] = parts if len(parts) != 1 or parts[0].get("type") != "text" \
                else parts[0]["text"]
            if isinstance(entry["content"], list) and len(entry["content"]) == 1 \
                    and entry["content"][0].get("type") == "text":
                entry["content"] = entry["content"][0]["text"]
        messages.append(entry)
    out = {"model": model, "messages": messages,
           "max_tokens": body.get("max_tokens") or 1024}
    if body.get("temperature") is not None:
        out["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        out["top_p"] = body["top_p"]
    stop = body.get("stop_sequences")
    if stop:
        out["stop"] = stop
    tools = body.get("tools")
    if tools:
        out["tools"] = [{
            "type": "function",
            "function": {"name": t.get("name"),
                         "description": t.get("description", ""),
                         "parameters": t.get("input_schema") or {"type": "object"}},
        } for t in tools]
        tc = body.get("tool_choice")
        if isinstance(tc, dict):
            if tc.get("type") == "any":
                out["tool_choice"] = "required"
            elif tc.get("type") == "tool":
                out["tool_choice"] = {"type": "function",
                                      "function": {"name": tc.get("name")}}
            else:
                out["tool_choice"] = "auto"
    return out


def anthropic_to_gemini(body, model):
    system_text = ""
    system = body.get("system")
    if system:
        system_text = system if isinstance(system, str) else anthropic_text_of(system)
    contents = []
    for msg in body.get("messages", []):
        role = "model" if msg.get("role") == "assistant" else "user"
        content = msg.get("content")
        parts = []
        if isinstance(content, str):
            if content:
                parts.append({"text": content})
        else:
            for block in content or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    parts.append({"text": block.get("text", "")})
                elif block.get("type") == "image":
                    src = block.get("source") or {}
                    if src.get("type") == "base64":
                        parts.append({"inlineData": {
                            "mimeType": src.get("media_type", "image/png"),
                            "data": src.get("data", "")}})
                    else:
                        parts.append({"fileData": {"fileUri": src.get("url", "")}})
                elif block.get("type") == "tool_result":
                    parts.append({"text": anthropic_text_of(block.get("content"))})
                elif block.get("type") == "tool_use":
                    parts.append({"text": json.dumps(block.get("input") or {})})
        contents.append({"role": role, "parts": parts})
    out = {"model": model, "contents": contents}
    if system_text:
        out["systemInstruction"] = {"parts": [{"text": system_text}]}
    tools = body.get("tools")
    if tools:
        out["tools"] = [{"functionDeclarations": [{
            "name": t.get("name"),
            "description": t.get("description", ""),
            "parameters": t.get("input_schema") or {"type": "object"},
        } for t in tools]}]
    gen = {}
    if body.get("max_tokens") is not None:
        gen["maxOutputTokens"] = body["max_tokens"]
    if body.get("temperature") is not None:
        gen["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        gen["topP"] = body["top_p"]
    stop = body.get("stop_sequences")
    if stop:
        gen["stopSequences"] = list(stop)
    if gen:
        out["generationConfig"] = gen
    return out


def build_upstream_body(client_api, body, cand, stream):
    """Return (path, upstream_body) for a candidate provider."""
    wire = cand["wire"]
    model = cand["model"]
    if client_api == "openai":
        if wire == "openai":
            out = dict(body)
            out["model"] = model
            if stream:
                out["stream"] = True
                out["stream_options"] = {"include_usage": True}
            return "/v1/chat/completions", out
        if wire == "anthropic":
            out = openai_to_anthropic(body, model)
            out["stream"] = stream
            return "/v1/messages", out
        return "/v1beta/models/%s:streamGenerateContent?alt=sse" % model \
            if stream else "/v1beta/models/%s:generateContent" % model, \
            openai_to_gemini(body, model)
    if wire == "openai":
        out = anthropic_to_openai(body, model)
        if stream:
            out["stream"] = True
            out["stream_options"] = {"include_usage": True}
        return "/v1/chat/completions", out
    if wire == "anthropic":
        out = dict(body, model=model)
        out["stream"] = stream
        return "/v1/messages", out
    return "/v1beta/models/%s:streamGenerateContent?alt=sse" % model \
        if stream else "/v1beta/models/%s:generateContent" % model, \
        anthropic_to_gemini(body, model)


# ------------------------------------------- upstream call + response ---

def sse_events(raw):
    """Parse an SSE byte payload into a list of parsed JSON data dicts."""
    events = []
    for chunk in raw.decode("utf-8", "replace").split("\n\n"):
        for line in chunk.splitlines():
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload and payload != "[DONE]":
                    try:
                        events.append(json.loads(payload))
                    except Exception:
                        pass
    return events


def parse_openai_answer(status, raw, stream):
    if status < 200 or status >= 300:
        raise ValueError("upstream status %d" % status)
    if not stream:
        doc = json.loads(raw.decode("utf-8"))
        choice = (doc.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        usage = doc.get("usage") or {}
        return {
            "text": text_of(msg.get("content")),
            "tool_calls": msg.get("tool_calls") or [],
            "stop": choice.get("finish_reason") or "stop",
            "prompt_tokens": usage.get("prompt_tokens") or 0,
            "completion_tokens": usage.get("completion_tokens") or 0,
        }
    text_parts = []
    tool_calls = []
    stop = "stop"
    prompt_tokens = 0
    completion_tokens = 0
    for ev in sse_events(raw):
        for choice in ev.get("choices") or []:
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                text_parts.append(delta["content"])
            for tc in delta.get("tool_calls") or []:
                fn = tc.get("function") or {}
                idx = tc.get("index", 0)
                while len(tool_calls) <= idx:
                    tool_calls.append({"id": "", "type": "function",
                                       "function": {"name": "", "arguments": ""}})
                if tc.get("id"):
                    tool_calls[idx]["id"] = tc["id"]
                if fn.get("name"):
                    tool_calls[idx]["function"]["name"] = fn["name"]
                if fn.get("arguments"):
                    tool_calls[idx]["function"]["arguments"] += fn["arguments"]
            if choice.get("finish_reason"):
                stop = choice["finish_reason"]
        usage = ev.get("usage")
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens") or prompt_tokens
            completion_tokens = usage.get("completion_tokens") or completion_tokens
    return {"text": "".join(text_parts), "tool_calls": tool_calls, "stop": stop,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}


def parse_anthropic_answer(status, raw, stream):
    if status < 200 or status >= 300:
        raise ValueError("upstream status %d" % status)
    if not stream:
        doc = json.loads(raw.decode("utf-8"))
        text_parts = []
        tool_calls = []
        for block in doc.get("content") or []:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append({"id": block.get("id"), "type": "function",
                                   "function": {"name": block.get("name"),
                                                "arguments": json.dumps(
                                                    block.get("input") or {})}})
        usage = doc.get("usage") or {}
        return {"text": "".join(text_parts), "tool_calls": tool_calls,
                "stop": doc.get("stop_reason") or "end_turn",
                "prompt_tokens": usage.get("input_tokens") or 0,
                "completion_tokens": usage.get("output_tokens") or 0}
    text_parts = []
    tool_calls = []
    stop = "end_turn"
    prompt_tokens = 0
    completion_tokens = 0
    cur_tool = None
    for ev in sse_events(raw):
        etype = ev.get("type")
        if etype == "message_start":
            prompt_tokens = (ev.get("message") or {}).get("usage", {}).get("input_tokens") or 0
        elif etype == "content_block_delta":
            delta = ev.get("delta") or {}
            if delta.get("type") == "text_delta":
                text_parts.append(delta.get("text", ""))
            elif delta.get("type") == "input_json_delta" and cur_tool is not None:
                cur_tool["function"]["arguments"] += delta.get("partial_json", "")
        elif etype == "content_block_start":
            block = ev.get("content_block") or {}
            if block.get("type") == "tool_use":
                cur_tool = {"id": block.get("id"), "type": "function",
                            "function": {"name": block.get("name"), "arguments": ""}}
                tool_calls.append(cur_tool)
        elif etype == "message_delta":
            stop = ev.get("delta", {}).get("stop_reason") or stop
            completion_tokens = ev.get("usage", {}).get("output_tokens") or completion_tokens
    return {"text": "".join(text_parts), "tool_calls": tool_calls, "stop": stop,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}


def parse_gemini_answer(status, raw, stream):
    if status < 200 or status >= 300:
        raise ValueError("upstream status %d" % status)
    docs = sse_events(raw) if stream else [json.loads(raw.decode("utf-8"))]
    text_parts = []
    prompt_tokens = 0
    completion_tokens = 0
    stop = "stop"
    for doc in docs:
        for cand in doc.get("candidates") or []:
            for part in (cand.get("content") or {}).get("parts") or []:
                if isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
            if cand.get("finishReason"):
                stop = "stop"
        usage = doc.get("usageMetadata") or {}
        if usage.get("promptTokenCount") is not None:
            prompt_tokens = usage["promptTokenCount"]
        if usage.get("candidatesTokenCount") is not None:
            completion_tokens = usage["candidatesTokenCount"]
    return {"text": "".join(text_parts), "tool_calls": [], "stop": stop,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}


PARSERS = {"openai": parse_openai_answer, "anthropic": parse_anthropic_answer,
           "gemini": parse_gemini_answer}


def call_upstream(env_url, provider, path, body, stream):
    url = env_url.rstrip("/") + provider.get("path", "") + path
    headers = {"Content-Type": "application/json",
               "Accept": "text/event-stream" if stream else "application/json"}
    headers.update(auth_headers(provider))
    if provider.get("wire") == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT) as resp:
            raw = resp.read()
            return resp.status, raw
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:
        raise ValueError("upstream error: %s" % e)

# -------------------------------------------- client response rendering ---

def openai_stop(stop, tool_calls):
    if tool_calls:
        return "tool_calls"
    if stop in ("max_tokens", "length"):
        return "length"
    if stop in ("content_filter", "refusal"):
        return "content_filter"
    return "stop"


def anthropic_stop(stop, tool_calls):
    if tool_calls:
        return "tool_calls"
    if stop in ("max_tokens", "length"):
        return "max_tokens"
    return "end_turn"


def render_openai_json(ans, model):
    msg = {"role": "assistant", "content": ans["text"]}
    if ans["tool_calls"]:
        msg["tool_calls"] = ans["tool_calls"]
    return {
        "id": "chatcmpl-sim",
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [{"index": 0, "message": msg,
                     "finish_reason": openai_stop(ans["stop"], ans["tool_calls"]),
                     "logprobs": None}],
        "usage": {"prompt_tokens": ans["prompt_tokens"],
                  "completion_tokens": ans["completion_tokens"],
                  "total_tokens": ans["prompt_tokens"] + ans["completion_tokens"]},
    }


def render_openai_stream(ans, model):
    lines = []

    def chunk(obj):
        lines.append("data: " + json.dumps(obj, separators=(",", ":")))

    chunk({"id": "chatcmpl-sim", "object": "chat.completion.chunk",
           "created": FIXED_CREATED, "model": model,
           "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None}]})
    if ans["text"]:
        chunk({"id": "chatcmpl-sim", "object": "chat.completion.chunk",
               "created": FIXED_CREATED, "model": model,
               "choices": [{"index": 0, "delta": {"content": ans["text"]},
                            "finish_reason": None}]})
    for i, tc in enumerate(ans["tool_calls"]):
        chunk({"id": "chatcmpl-sim", "object": "chat.completion.chunk",
               "created": FIXED_CREATED, "model": model,
               "choices": [{"index": 0,
                            "delta": {"tool_calls": [dict(tc, index=i)]},
                            "finish_reason": None}]})
    chunk({"id": "chatcmpl-sim", "object": "chat.completion.chunk",
           "created": FIXED_CREATED, "model": model,
           "choices": [{"index": 0, "delta": {},
                        "finish_reason": openai_stop(ans["stop"], ans["tool_calls"])}]})
    chunk({"id": "chatcmpl-sim", "object": "chat.completion.chunk",
           "created": FIXED_CREATED, "model": model,
           "choices": [],
           "usage": {"prompt_tokens": ans["prompt_tokens"],
                     "completion_tokens": ans["completion_tokens"],
                     "total_tokens": ans["prompt_tokens"] + ans["completion_tokens"]}})
    lines.append("data: [DONE]")
    return ("\n\n".join(lines) + "\n\n").encode("utf-8")


def render_anthropic_json(ans, model):
    content = [{"type": "text", "text": ans["text"]}]
    for tc in ans["tool_calls"]:
        try:
            args = json.loads(tc["function"]["arguments"] or "{}")
        except Exception:
            args = {}
        content.append({"type": "tool_use", "id": tc["id"],
                        "name": tc["function"]["name"], "input": args})
    return {
        "id": "msg_sim",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": anthropic_stop(ans["stop"], ans["tool_calls"]),
        "stop_sequence": None,
        "usage": {"input_tokens": ans["prompt_tokens"],
                  "output_tokens": ans["completion_tokens"]},
    }


def render_anthropic_stream(ans, model):
    lines = []

    def chunk(obj):
        lines.append("event: " + obj["type"])
        lines.append("data: " + json.dumps(obj, separators=(",", ":")))
        lines.append("")

    chunk({"type": "message_start",
           "message": {"id": "msg_sim", "type": "message", "role": "assistant",
                       "model": model, "content": [],
                       "stop_reason": None, "stop_sequence": None,
                       "usage": {"input_tokens": ans["prompt_tokens"],
                                 "output_tokens": 0}}})
    chunk({"type": "content_block_start", "index": 0,
           "content_block": {"type": "text", "text": ""}})
    if ans["text"]:
        chunk({"type": "content_block_delta", "index": 0,
               "delta": {"type": "text_delta", "text": ans["text"]}})
    chunk({"type": "content_block_stop", "index": 0})
    chunk({"type": "message_delta",
           "delta": {"stop_reason": anthropic_stop(ans["stop"], ans["tool_calls"]),
                     "stop_sequence": None},
           "usage": {"output_tokens": ans["completion_tokens"]}})
    chunk({"type": "message_stop"})
    return ("\n".join(lines)).encode("utf-8")


# ---------------------------------------------------------------- server ---

LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send_json(self, status, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        with LOCK:
            self._do_get()

    def _do_get(self):
        state = self.server.state
        if self.path.split("?")[0] != "/v1/models":
            self._send_json(404, {"error": {"message": "not found",
                                            "type": "not_found"}})
            return
        catalog = state["catalog"]
        if "x-api-key" in self.headers:
            data = []
            for entry in catalog["providers"]:
                for m in entry["models"]:
                    data.append({"type": "model", "id": m["id"],
                                 "display_name": m["id"],
                                 "created_at": "2026-01-01T00:00:00Z"})
            self._send_json(200, {"data": data, "has_more": False,
                                  "first_id": data[0]["id"] if data else None,
                                  "last_id": data[-1]["id"] if data else None})
        else:
            data = []
            for entry in catalog["providers"]:
                for m in entry["models"]:
                    data.append({"id": m["id"], "object": "model",
                                 "created": FIXED_CREATED,
                                 "owned_by": entry["id"]})
            self._send_json(200, {"object": "list", "data": data})

    def do_POST(self):
        with LOCK:
            self._do_post()

    def _do_post(self):
        state = self.server.state
        path = self.path.split("?")[0]
        if path == "/v1/chat/completions":
            client_api = "openai"
        elif path == "/v1/messages":
            client_api = "anthropic"
        else:
            self._send_json(404, {"error": {"message": "not found",
                                            "type": "not_found"}})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:
            self._send_json(400, {"error": {"message": "bad json",
                                            "type": "invalid_request"}})
            return
        stream = bool(body.get("stream"))
        router = state["router"]
        idx = state["req_index"]
        state["req_index"] += 1
        manifest = router.manifest
        if idx < len(manifest):
            item = manifest[idx]
            req = dict(item.get("requires") or {})
        else:
            req = derive_requirements(client_api, body)
            item = {"id": "extra-%d" % idx, "api": client_api}
        client_json = client_api == "openai" and \
            (body.get("response_format") or {}).get("type") == "json_object"
        if client_json:
            req["json"] = True
        ans = None
        used_model = None
        last_error = "no candidate"
        for cand in router.order_for(item["id"], req, client_api, client_json):
            provider = router.providers[cand["provider"]]
            upath, ubody = build_upstream_body(client_api, body, cand, stream)
            try:
                status, raw_up = call_upstream(state["env_url"], provider,
                                               upath, ubody, stream)
            except ValueError as e:
                last_error = str(e)
                continue
            router.used[cand["provider"]] += 1
            if status == 402:
                router.dead.add(cand["provider"])
                log("provider %s spent (402), dropped" % cand["provider"])
                last_error = "402 from %s" % cand["provider"]
                continue
            if status < 200 or status >= 300:
                last_error = "status %d from %s" % (status, cand["provider"])
                continue
            try:
                ans = PARSERS[cand["wire"]](status, raw_up, stream)
            except Exception as e:
                last_error = "parse failed from %s: %s" % (cand["provider"], e)
                continue
            used_model = cand["model"]
            break
        if ans is None:
            log("request %s unserved: %s" % (item["id"], last_error))
            if client_api == "openai":
                self._send_json(503, {"error": {"message": last_error,
                                                "type": "server_error",
                                                "code": "no_free_model"}})
            else:
                self._send_json(503, {"type": "error",
                                      "error": {"type": "overloaded_error",
                                                "message": last_error}})
            return
        if client_api == "openai":
            if stream:
                self._send_bytes(200, render_openai_stream(ans, used_model),
                                 "text/event-stream")
            else:
                self._send_json(200, render_openai_json(ans, used_model))
        else:
            if stream:
                self._send_bytes(200, render_anthropic_stream(ans, used_model),
                                 "text/event-stream")
            else:
                self._send_json(200, render_anthropic_json(ans, used_model))


def derive_requirements(client_api, body):
    req = {}
    if client_api == "openai":
        if body.get("tools"):
            req["tools"] = True
        if (body.get("response_format") or {}).get("type") == "json_object":
            req["json"] = True
        for msg in body.get("messages", []):
            if image_parts_of(msg.get("content")):
                req["vision"] = True
    else:
        if body.get("tools"):
            req["tools"] = True
        for msg in body.get("messages", []):
            if anthropic_images_of(msg.get("content")):
                req["vision"] = True
    return req


# ------------------------------------------------------------------ main ---

def http_json(method, url, body=None, timeout=10.0):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def main():
    brief = load_brief()
    env_url = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
    if brief is None:
        if "--list" in sys.argv:
            print('{"providers":[]}')
            return 0
        log("no brief: set MAKEFUN_PROMPT_FILE")
        return 1
    probes = {}
    for provider in brief.providers:
        ok, models = probe_provider(env_url, provider) if env_url else (False, [])
        probes[provider.get("id")] = (ok, models)
        if ok:
            log("probe %s: %d models" % (provider.get("id"), len(models)))
    catalog = build_catalog(brief, probes)
    if "--list" in sys.argv:
        print(catalog_json(catalog))
        return 0
    if not env_url:
        log("no env: set MAKEFUN_ENV_URL")
        return 1
    router = Router(brief, catalog)
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    server.daemon_threads = True
    port = server.server_address[1]
    server.state = {"catalog": catalog, "router": router, "env_url": env_url,
                    "req_index": 0}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, answer = http_json("POST", env_url + "/register", {
            "protocol_version": PROTOCOL_VERSION, "port": port,
            "catalog": catalog})
        log("register: %s %s" % (status, answer))
        if not answer.get("ok"):
            return 1
    except Exception as e:
        log("register failed: %s" % e)
        return 1
    while True:
        try:
            _, work = http_json("GET", env_url + "/work", timeout=5.0)
        except Exception as e:
            log("work poll failed: %s" % e)
            time.sleep(POLL_INTERVAL)
            continue
        if work.get("done"):
            break
        time.sleep(POLL_INTERVAL)
    server.shutdown()
    server.server_close()
    log("work done, exiting 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
