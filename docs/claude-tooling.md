# Claude Code tooling for this repository

Status as of 2026-09-30, cloud environment (Claude Code 2.1.285, managed install).

## Kept as is

| Tool | Version | Role |
|---|---|---|
| Claude Code (managed, `/opt/claude-code`) | 2.1.285 | Main agent. The stale npm global `@anthropic-ai/claude-code@2.1.42` is unused (the managed symlink wins). |
| Playwright + `@playwright/cli` + `playwright-cli` skill | 1.56.1 / 0.1.22 | Browser automation; only needed later for a dashboard UI. |
| claude-mem (plugin, thedotmack) | 13.28.0 | Cross-session memory. Note: its worker runs its own Claude calls. |
| Task Observer (synced skill, rebelytics) | unversioned | Skill-improvement observations. |

## Claude Code Setup (installed)

- Source: official `anthropics/claude-plugins-official` marketplace (commit `aa5654b`),
  plugin `claude-code-setup@claude-plugins-official` v1.0.0, publisher tier "anthropic".
- Contents: one read-only skill, `claude-automation-recommender`. No hooks, agents or MCP
  servers. Projected cost: ~141 tokens always-on, ~4.1k per invocation.
- Installed at user scope in this container. The container is ephemeral; reinstall with
  `claude plugin marketplace add anthropics/claude-plugins-official` and
  `claude plugin install claude-code-setup@claude-plugins-official`.

### Its recommendations for this repo (read-only run) and our evaluation

Codebase profile: Python 3.11 + Poetry, pytest (~940 tests), black configured, no ruff/mypy
config, GitHub repo without CI workflows, pandas/numpy/scipy, Anthropic/OpenAI SDKs via
langchain, no web frontend, no database.

| Category | Recommendation | Useful | Overlap | Security | Token cost | Deterministic | Decision |
|---|---|---|---|---|---|---|---|
| Hook | PreToolUse: block edits to `poetry.lock` and `.env*` | High | None | Improves | ~0 | Yes | Adopt later (needs a small script) |
| Hook | PostToolUse: run related pytest file after edits | Medium | Our explicit test runs | Neutral | Test time per edit | Yes | Defer: suite is fast enough to run per phase |
| Subagent | `test-writer` / look-ahead reviewer for `hedge_fund/systematic` | Medium | Built-in general-purpose agent | Neutral | High per use | No | Defer |
| Skill | `project-conventions` (Claude-only) | Low | Duplicates `CLAUDE.md` | Neutral | Always-on | n/a | Reject: `CLAUDE.md` already does this |
| MCP | context7 (library docs) | Medium | WebFetch | External calls | Per call | No | Reject for now: adds a network dependency; pandas/numpy are stable |
| MCP | GitHub MCP | — | Already provided by the session | — | — | — | Already present |
| Plugin | `pyright-lsp` | Medium | None | Neutral | Low | Yes | Candidate once type hints are enforced |
| Plugin | `frontend-design` | Low now | — | — | — | — | Not enabled (no UI yet) |

Nothing from the list was installed automatically.

## Frontend Design

Present on disk at `/mnt/skills/public/frontend-design` and in the official marketplace.
Not enabled: there is no UI in scope. Revisit when a monitoring dashboard is built.

## Headroom (installed, not in the loop)

`headroom-ai` 0.39.1 is installed as a uv tool; `ANTHROPIC_BASE_URL` points straight at
`api.anthropic.com`, so nothing is routed through it. Keep it out of the main workflow.

Controlled A/B evaluation, if wanted later:
1. Fix a frozen task set: e.g. 20 recorded prompts from `runs/*/llm_cache` (research
   critiques, not trading decisions) plus 5 coding tasks with known test outcomes.
2. Arm A: direct API. Arm B: same model and parameters through Headroom. Same seeds/order.
3. Measure input/output tokens, cost, latency, and output quality: exact-match of parsed
   JSON fields for the cached prompts, test pass rate for coding tasks.
4. Accept only if tokens drop materially with no quality regression; never for the
   production research pipeline without a second confirmation run.

## OmniRoute (installed, unconfigured)

`omniroute` 3.8.51 (npm global) has no config or data directory and is not running.

Rules if it is used later:
- Never `auto` routing for research or backtests: a model swap mid-experiment breaks
  reproducibility and cache keys.
- Define explicit named profiles, one model each, for example
  `research-claude -> claude-opus-5-5` and `research-kimi -> kimi-k3`, and call the
  profile name.
- The model id in every LLM cache record must be the upstream model actually served;
  a profile that silently falls back must fail instead.
- Keys stay in the OmniRoute store or env vars; nothing is committed.
