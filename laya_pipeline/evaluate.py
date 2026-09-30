"""Avalia um checkpoint (o base, para baseline, ou o treinado) no conjunto de
avaliação e grava runs/<versão>/eval_<rótulo>.json + report.md."""

import json
import os
import time
from typing import Any, Dict, List, Optional

from .metrics import QuestionMetrics, overall, percentile
from .task import Task, option_keys
from .teacher import read_jsonl


def load_agent(task: Task, checkpoint: Optional[str], device: Optional[str] = None):
    """checkpoint=None → modelo base do task.yaml; senão, diretório local."""
    import laya

    try:
        from transformers.initialization import no_init_weights
    except ImportError:
        from transformers.modeling_utils import no_init_weights

    with no_init_weights():
        if checkpoint:
            agent = laya.load(os.path.abspath(checkpoint), device=device)
        else:
            agent = laya.load(task.base_model["id"], device=device,
                              subfolder=task.base_model.get("subfolder"))
    # O base foi treinado com 1024 tokens; avaliar já com o contexto da tarefa
    # mostra o quanto ele perde (ou não) sem treino nesse tamanho.
    agent.cfg["max_len"] = task.context["max_len"]
    agent.cfg["head_max_len"] = task.context["head_max_len"]
    return agent


def to_probabilities(q: Dict[str, Any], answer: Dict[str, Any]) -> Dict[str, float]:
    if q["type"] == "noul":
        return {"true": answer["noul"], "false": round(1.0 - answer["noul"], 4)}
    return answer["probabilities"]


def evaluate(task: Task, label: str, checkpoint: Optional[str] = None,
             device: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
    rows = read_jsonl(os.path.join(task.run_dir, "eval.jsonl"))
    if not rows:
        raise SystemExit("eval.jsonl vazio: rode `split` primeiro")
    if limit:
        rows = rows[:limit]

    agent = load_agent(task, checkpoint, device)
    metrics = {qid: QuestionMetrics(q["type"], option_keys(q)) for qid, q in task.questions.items()}
    latencies: List[float] = []
    worst: List[Dict[str, Any]] = []
    vs_human = {"cases": 0, "agree": 0}

    agent.system_one(rows[0]["state"], task.questions)  # aquecimento, fora da medição
    for i, row in enumerate(rows, 1):
        start = time.perf_counter()
        out = agent.system_one(row["state"], task.questions)
        latencies.append(time.perf_counter() - start)
        for qid, q in task.questions.items():
            pred = to_probabilities(q, out["answers"][qid])
            gold = row["gold"][qid]["probabilities"]
            metrics[qid].add(pred, gold)
            expected = row.get("human", {}).get(qid)
            if expected is not None:
                vs_human["cases"] += 1
                vs_human["agree"] += int(max(pred, key=pred.get) == expected)
            if max(pred, key=pred.get) != max(gold, key=gold.get):
                worst.append({"id": row["id"], "question": qid, "gold": gold, "pred": pred,
                              "gap": round(max(gold.values()) - gold.get(max(pred, key=pred.get), 0), 4),
                              "excerpt": row["state"][:300]})
        if i % 50 == 0:
            print("avaliados %d/%d" % (i, len(rows)), flush=True)

    per_question = {qid: m.summary() for qid, m in metrics.items()}
    worst.sort(key=lambda w: -w["gap"])
    result = {
        "task": task.name, "version": task.version, "label": label,
        "checkpoint": checkpoint or "%s/%s" % (task.base_model["id"], task.base_model.get("subfolder") or ""),
        "device": str(agent.device), "max_len": task.context["max_len"], "cases": len(rows),
        "overall": overall(per_question), "questions": per_question,
        "confusion": {qid: m.confusion() for qid, m in metrics.items()},
        "latency_s": {"p50": round(percentile(latencies, 50), 3),
                      "p95": round(percentile(latencies, 95), 3)},
        "worst_errors": worst[:15],
        "vs_human": dict(vs_human, accuracy=round(vs_human["agree"] / vs_human["cases"], 4)
                         if vs_human["cases"] else None),
    }
    path = os.path.join(task.run_dir, "eval_%s.json" % label)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    write_report(task)
    return result


def check_goals(task: Task, result: Dict[str, Any]) -> Dict[str, Optional[bool]]:
    goals, ov = task.goals, result.get("overall", {})
    return {
        "min_accuracy": None if goals.get("min_accuracy") is None
        else ov.get("accuracy", 0) >= goals["min_accuracy"],
        "max_ece": None if goals.get("max_ece") is None else ov.get("ece", 1) <= goals["max_ece"],
    }


def write_report(task: Task) -> str:
    """report.md comparando baseline e treinado (o que existir)."""
    runs = {}
    for label in ("baseline", "trained"):
        path = os.path.join(task.run_dir, "eval_%s.json" % label)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                runs[label] = json.load(f)
    lines = ["# Relatório — %s %s" % (task.name, task.version), "",
             "Contexto: max_len=%d, head_max_len=%d." % (task.context["max_len"], task.context["head_max_len"]),
             "", "| | casos | acurácia | Brier | ECE | latência p50 | p95 | dispositivo |",
             "|---|---|---|---|---|---|---|---|"]
    for label, r in runs.items():
        ov = r["overall"]
        lines.append("| %s | %d | %s | %s | %s | %.2fs | %.2fs | %s |" % (
            label, r["cases"], ov.get("accuracy"), ov.get("brier"), ov.get("ece"),
            r["latency_s"]["p50"], r["latency_s"]["p95"], r["device"]))
    if any(r.get("vs_human", {}).get("cases") for r in runs.values()):
        lines += ["", "Contra respostas humanas (data/exemplos/): " + "; ".join(
            "%s %s/%s" % (label, r["vs_human"]["agree"], r["vs_human"]["cases"])
            for label, r in runs.items() if r.get("vs_human", {}).get("cases"))]
    lines += ["", "Latência medida no dispositivo da avaliação; a de produção (CPU serverless) "
              "vem do bench/latency.py.", "", "## Por pergunta", ""]
    for label, r in runs.items():
        lines.append("**%s**" % label)
        for qid, m in r["questions"].items():
            lines.append("- %s: %s" % (qid, ", ".join("%s=%s" % kv for kv in m.items())))
        lines.append("")
    if "trained" in runs:
        goals = check_goals(task, runs["trained"])
        lines += ["## Metas", ""] + ["- %s: %s" % (k, {None: "não definida", True: "atingida",
                                                         False: "NÃO atingida"}[v])
                                      for k, v in goals.items()] + [""]
        lines += ["## Piores erros (treinado)", ""]
        for w in runs["trained"]["worst_errors"][:10]:
            lines.append("- `%s` %s: professor=%s modelo=%s — %s…" % (
                w["id"], w["question"], max(w["gold"], key=w["gold"].get),
                max(w["pred"], key=w["pred"].get), w["excerpt"][:160].replace("\n", " ")))
    path = os.path.join(task.run_dir, "report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path
