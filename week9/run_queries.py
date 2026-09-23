"""One allergen query on both servers, then the scoped gateway denial."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mcp_agent

OUT = Path(__file__).resolve().parent
AUDIT = OUT / "gateway_audit.log"
if AUDIT.exists():
    AUDIT.unlink()


def show(label: str, result: dict) -> None:
    print("==", label)
    print("tools", result["tools"])
    for step in result["steps"]:
        print(step)
    print("ANSWER", result["answer"])


def main() -> None:
    butter = mcp_agent.run("What allergens does butter have?")
    show("both-servers", butter)
    gateway = mcp_agent.run(
        "What allergens does butter have, and what is its nutrition per 100g?",
        config_path=OUT / "mcp_config.gateway.json",
    )
    show("gateway", gateway)
    (OUT / "query_runs.json").write_text(
        json.dumps({"butter": butter, "gateway": gateway}, indent=2),
        encoding="utf-8",
    )
    print("AUDIT")
    print(AUDIT.read_text(encoding="utf-8") if AUDIT.exists() else "(missing)")


if __name__ == "__main__":
    main()
