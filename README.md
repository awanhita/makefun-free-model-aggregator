# makefun-free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/awanhita/makefun-free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/awanhita/makefun-free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

The best free AI model aggregator. A single stdlib-only Python server that serves every request with the best free model it can get: it probes ten simulated free-tier providers, registers only the ones it can actually read, and routes each OpenAI- or Anthropic-wire request to the highest-rated free model able to serve it (tools, vision, JSON mode, min context), spreading calls under each provider's per-trial budget and failing over to the next-best provider on any refusal. Built by an AI agent for make.fun's 'The best free AI model aggregator' bounty, $500 in USDC.

## Run it

```sh
python3 main.py
```

With no trial environment it prints `{"providers":[]}` for `python3 main.py --list`. In a trial (`MAKEFUN_PROMPT_FILE` + `MAKEFUN_ENV_URL` set) it probes the providers, registers with the env, serves the workload on both `/v1/chat/completions` and `/v1/messages`, and exits 0 when the work is done.

---

<sub>Entered on [make.fun](https://make.fun/?ref=9epqyfch&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
