from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parents[2] / "deploy" / "github-actions" / "daily.yml"


def test_workflow_shape():
    wf = yaml.safe_load(WORKFLOW.read_text())
    triggers = wf.get("on") or wf.get(True)                 # YAML 1.1 reads a bare `on` as True
    assert triggers["schedule"] == [{"cron": "0 8 * * 1-5"}]
    assert "workflow_dispatch" in triggers
    assert wf["concurrency"]["group"] == "hedge-fund-daily" and wf["concurrency"]["cancel-in-progress"] is False
    steps = wf["jobs"]["daily"]["steps"]
    run = next(s for s in steps if s.get("id") == "run")
    assert run["run"].strip() == "aihf-daily --notify"
    assert "ANTHROPIC_API_KEY" in run["env"] and "TELEGRAM_BOT_TOKEN" in run["env"]
    save = next(s for s in steps if s.get("name") == "Save state")
    assert save["if"] == "always()" and "git push" in save["run"]
