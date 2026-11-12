import argparse
import json
import re
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.answer_judge import Judge

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--out-dir", type=Path, required=True)
args, _ = parser.parse_known_args()

original_parse = Judge.parse
audit_path = args.out_dir / "brace_repair_audit.jsonl"


def parse_with_repair(raw):
    try:
        return original_parse(raw)
    except json.JSONDecodeError as original_error:
        text = re.sub(
            r"^```(?:json)?\s*", "", raw.strip(), flags=re.I
        )
        text = re.sub(r"\s*```$", "", text).strip()

        try:
            json.loads(text)
        except json.JSONDecodeError as error:
            can_repair = (
                error.msg == "Expecting ',' delimiter"
                and error.pos == len(text)
                and text.startswith("{")
                and not text.endswith("}")
            )
            if not can_repair:
                raise original_error
        else:
            raise original_error

        fixed = text + "\n}"
        result = original_parse(fixed)

        record = {
            "policy": "append-one-final-brace",
            "original_response": raw,
            "repaired_response": fixed,
            "validated_result": dict(result),
        }

        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

        result["_format_repair"] = {
            "policy": "append-one-final-brace",
            "audit_file": str(audit_path),
        }
        print("FORMAT REPAIR: added one closing brace", flush=True)
        return result


Judge.parse = staticmethod(parse_with_repair)

script = ROOT / "scripts/evaluate_answers.py"
sys.argv = [str(script), *sys.argv[1:]]
runpy.run_path(str(script), run_name="__main__")
