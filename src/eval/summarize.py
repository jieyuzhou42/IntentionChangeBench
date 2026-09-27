"""Shared metrics for independent Action and Intention results."""
from eval import travelplanner_eval_v2 as intention


def metric(values, total):
    return {"value": sum(values) / len(values) if values else None,
            "numerator": sum(values), "denominator": len(values), "excluded": total - len(values)}


def summarize(rows):
    output = {}
    for model in sorted({r["model"] for r in rows}):
        group = [r for r in rows if r["model"] == model]
        actions = [r["action"] for r in group if r.get("action") is not None]
        intents = [r for r in group if r.get("intention") is not None]
        # Reuse the existing intention aggregation without supplying fake Actions.
        im = intention.summarize([{**r, "action": None} for r in intents], ["all"])["intention"]
        output[model] = {"turns": len(group), "action_scored_turns": len(actions),
            "intention_scored_turns": len(intents),
            "action": {key: metric([float(a[key]) for a in actions], len(group))
                       for key in ("action_score", "action_success", "must_gate", "gate_pass")},
            "intention": im,
            "evaluation_errors": [{"instance_id": r["instance_id"], "turn_id": r["turn_id"], "errors": r["errors"]}
                                  for r in group if r.get("errors")]}
    return output


def tables(metrics):
    lines = ["| Model | Action score | Binary success | Must gate | Action coverage | Intention coverage |",
             "|---|---:|---:|---:|---:|---:|"]
    for model, m in metrics.items():
        def pct(key):
            v = m["action"][key]["value"]
            return "N/A" if v is None else "{:.2%}".format(v)
        lines.append("| {} | {} | {} | {} | {}/{} | {}/{} |".format(model, pct("action_score"),
            pct("action_success"), pct("must_gate"), m["action_scored_turns"], m["turns"], m["intention_scored_turns"], m["turns"]))
    return "\n".join(lines) + "\n"
