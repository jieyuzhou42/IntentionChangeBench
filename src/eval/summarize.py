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
        im = intention.summarize([{**r, "action": None} for r in group], ["all"])["intention"]
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
    return "\n".join(lines) + "\n\n" + intention_table(metrics)


def intention_table(metrics):
    """Explicit names, equal turn weights, and conditional/change coverage."""
    fields = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"),
              ("conditional_priority_accuracy", "Conditional Priority Accuracy"),
              ("priority_precision", "Priority-aware Precision"),
              ("priority_recall", "Priority-aware Recall"), ("priority_f1", "Priority-aware F1 (primary)"),
              ("change_precision", "Change Precision"), ("change_recall", "Change Recall"), ("change_f1", "Change F1"),
              ("priority_change_precision", "Delta Priority-aware Precision"),
              ("priority_change_recall", "Delta Priority-aware Recall"),
              ("priority_change_f1", "Delta Priority-aware F1 (primary change metric)")]
    lines = ["Intention level (turn macro; each cell is percent [scored turns])", "",
             "| Model | " + " | ".join(label for _, label in fields) + " |",
             "|---|" + "---:|" * len(fields)]
    for model, m in metrics.items():
        cells = []
        for key, _ in fields:
            metric = m["intention"][key]
            value = metric["value"]
            cells.append("N/A [0]" if value is None else "{:.2%} [{}]".format(value, metric["denominator"]))
        lines.append("| " + model + " | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
