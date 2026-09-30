# Kimi (Moonshot) integration

Status: **integration-ready, `KIMI_CREDENTIAL_REQUIRED`** — no Kimi/Moonshot key is set in
this environment, so nothing has been called. All behaviour is covered by mocked tests.

## What exists

| Piece | Where |
|---|---|
| Model id `kimi-k3` (provider `Kimi`) | `hedge_fund/llm/api_models.json` |
| Key variable `KIMI_API_KEY`, alias `MOONSHOT_API_KEY` (read first) | `hedge_fund/llm/registry.py` |
| OpenAI-compatible client, base URL `MOONSHOT_BASE_URL` or `https://api.moonshot.ai/v1` | `hedge_fund/llm/client.py` |
| Cross-model review (proposer/critic) with prompt cache | `hedge_fund/research/crossreview.py` |

## To enable (human action)

Add one secret to the cloud environment settings (never paste it in chat):

- `KIMI_API_KEY` — or `MOONSHOT_API_KEY`
- optional `MOONSHOT_BASE_URL=https://api.moonshot.cn/v1` for mainland-China accounts

## Workflow

```
Claude proposes  -> Kimi critiques        CrossReviewer.from_models("claude-opus-5-5", "kimi-k3")
Kimi proposes    -> Claude critiques      CrossReviewer.from_models("kimi-k3", "claude-opus-5-5")
Python validation gate -> PASS / FAIL     review(brief, gate=<validation result>)
```

- The verdict is the gate's result. A critique can only raise concerns
  (`needs_human_attention`); it can never turn FAIL into PASS. `ai_can_promote` is False.
- Models must be explicit registry ids. Router aliases (`auto`, `default`, `router`, ...)
  are rejected, including when calling through OmniRoute later.
- Every response is cached (`~/.hedge-fund/cache/llm`, key = role + model + prompt);
  a repeated review costs nothing.
