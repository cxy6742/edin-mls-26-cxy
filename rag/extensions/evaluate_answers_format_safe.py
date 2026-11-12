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
audit_path = args.out_dir / "control_character_repair_audit.jsonl"



def repair_identical_explanation(raw):
    text = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.I)
    text = re.sub(r"\s*```$", "", text).strip()
    string = r'"(?:[^"\\\x00-\x1f]|\\.)*"'
    pattern = (
        r'\{\s*"correctness"\s*:\s*[0-4]\s*,'
        r'\s*"grounding"\s*:\s*[0-4]\s*,'
        r'\s*"completeness"\s*:\s*[0-4]\s*,'
        r'\s*"explanation"\s*:\s*(?P<first>' + string + r')\s*'
        r'(?P<duplicate>"explanation"\s*:\s*(?P<second>' + string + r'))'
        r'\s*\}?\s*'
    )
    match = re.fullmatch(pattern, text)
    if match is None:
        return None
    if json.loads(match["first"]) != json.loads(match["second"]):
        return None

    fixed = text[:match.start("duplicate")].rstrip() + "\n}"
    result = original_parse(fixed)
    policy = "remove-identical-duplicate-explanation-and-close-object"
    record = {
        "policy": policy,
        "original_response": raw,
        "repaired_response": fixed,
        "validated_result": dict(result),
    }
    audit = args.out_dir / "duplicate_explanation_repair_audit.jsonl"
    audit.parent.mkdir(parents=True, exist_ok=True)
    with audit.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=True) + "\n")
    result["_format_repair"] = {
        "policy": policy,
        "audit_file": str(audit),
    }
    print("FORMAT REPAIR: removed identical duplicate explanation", flush=True)
    return result

def parse_control_safe(raw):
    try:
        return original_parse(raw)
    except json.JSONDecodeError as error:
        repaired = repair_identical_explanation(raw)
        if repaired is not None:
            return repaired
        failure_path = args.out_dir / "parse_failure_audit.jsonl"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "error": str(error),
            "position": error.pos,
            "response_length": len(raw),
            "raw_response": raw,
        }
        with failure_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=True) + "\n")
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        stray_quote = re.search(r'\n[ \t]*(")[ \t]*\}\s*$', cleaned)
        if (
            error.msg == "Expecting ',' delimiter"
            and stray_quote is not None
            and error.pos == stray_quote.start(1)
        ):
            position = stray_quote.start(1)
            fixed = cleaned[:position] + cleaned[position + 1:]
            result = original_parse(fixed)
            result["_format_repair"] = {
                "policy": "remove-stray-quote-before-final-brace",
                "audit_file": str(failure_path),
            }
            print("FORMAT REPAIR: removed stray quote before final brace", flush=True)
            return result

        if not error.msg.startswith("Invalid control character"):
            raise

        text = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.I)
        text = re.sub(r"\s*```$", "", text).strip()
        try:
            decoded = json.loads(text, strict=False)
        except json.JSONDecodeError:
            raise error from None

        fixed = json.dumps(decoded, ensure_ascii=True)
        result = original_parse(fixed)

        record = {
            "policy": "escape-json-string-control-characters",
            "original_response": raw,
            "repaired_response": fixed,
            "validated_result": dict(result),
        }
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=True) + "\n")

        print("FORMAT REPAIR: escaped JSON string control characters", flush=True)
        return result


Judge.parse = staticmethod(parse_control_safe)
runpy.run_path(
    str(ROOT / "extensions" / "evaluate_answers_brace_safe.py"),
    run_name="__main__",
)
