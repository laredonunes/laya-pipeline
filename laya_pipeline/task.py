"""Carrega e valida o `task.yaml` de um repositório de tarefa.

O `task.yaml` é o contrato entre o repositório da tarefa e o motor: tudo o
que o pipeline faz (rotular, treinar, avaliar, publicar) é derivado dele.
A validação é deliberadamente estrita — um erro aqui custa segundos; o mesmo
erro descoberto depois de horas de GPU custa a rodada inteira."""

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

import yaml

QUESTION_TYPES = ("choice", "score", "noul")
TEACHER_PROVIDERS = ("deepseek", "qwen", "openrouter")

DEFAULTS: Dict[str, Any] = {
    "language": "pt",
    "base_model": {"id": "convaiinnovations/laya", "subfolder": "multilingual"},
    "context": {"max_len": 2560, "head_max_len": 256},
    "teacher": {"provider": "deepseek", "model": None, "prompt": "prompts/teacher.md",
                "generator_prompt": "prompts/generator.md", "concurrency": 4,
                "temperature": 0.2},
    "data": {"texts": "data/texts.jsonl", "eval_fraction": 0.1, "min_texts": 200},
    "train": {"epochs": 4, "micro_batch": 2, "grad_accum": 8, "lr_encoder": 2.5e-5,
              "lr_head": 1.0e-4, "seed": 0},
    "goals": {"min_accuracy": None, "max_ece": None, "max_latency_p95_s": None},
    "publish": {"s3_prefix": "models/laya/"},
}


class TaskError(ValueError):
    """task.yaml inválido. A mensagem lista todos os problemas de uma vez."""


@dataclass
class Task:
    root: str
    name: str
    version: str
    description: str
    language: str
    base_model: Dict[str, Any]
    context: Dict[str, int]
    questions: Dict[str, Dict[str, Any]]
    teacher: Dict[str, Any]
    data: Dict[str, Any]
    train: Dict[str, Any]
    goals: Dict[str, Any]
    publish: Dict[str, Any]
    raw: Dict[str, Any] = field(repr=False, default_factory=dict)

    def path(self, *parts: str) -> str:
        """Caminho relativo à raiz do repositório da tarefa."""
        return os.path.join(self.root, *parts)

    @property
    def run_dir(self) -> str:
        """Artefatos da versão atual (dados rotulados, checkpoint, relatórios).
        Fica fora do git (ver .gitignore do template)."""
        return self.path("runs", self.version)

    @property
    def s3_key_prefix(self) -> str:
        return "%s%s/%s/" % (self.publish["s3_prefix"].rstrip("/") + "/", self.name, self.version)


def _merge(defaults: Dict[str, Any], given: Any) -> Dict[str, Any]:
    out = dict(defaults)
    if isinstance(given, dict):
        out.update(given)
    return out


def _check_question(qid: str, q: Any, errors: List[str]) -> None:
    where = "questions.%s" % qid
    if not re.fullmatch(r"[a-z][a-z0-9_]*", qid):
        errors.append("%s: id deve ser snake_case (a-z, 0-9, _)" % where)
    if not isinstance(q, dict):
        errors.append("%s: deve ser um mapeamento" % where)
        return
    t = q.get("type")
    if t not in QUESTION_TYPES:
        errors.append("%s.type: deve ser um de %s" % (where, ", ".join(QUESTION_TYPES)))
        return
    if not isinstance(q.get("instructions"), str) or not q["instructions"].strip():
        errors.append("%s.instructions: texto obrigatório" % where)
    crit = q.get("criteria")
    if t == "choice":
        if isinstance(crit, list):
            crit = {c: None for c in crit}
        if not isinstance(crit, dict) or len(crit) < 2:
            errors.append("%s.criteria: choice precisa de pelo menos 2 opções" % where)
        elif len(crit) > 20:
            errors.append("%s.criteria: mais de 20 opções degrada o Laya (orçamento de "
                          "head_max_len); divida em perguntas menores" % where)
        elif not all(isinstance(k, str) and re.fullmatch(r"[a-z0-9_]+", k) for k in crit):
            errors.append("%s.criteria: chaves das opções devem ser snake_case" % where)
    elif t == "score":
        if not isinstance(crit, list) or len(crit) < 2:
            errors.append("%s.criteria: score precisa de uma lista com pelo menos 2 níveis" % where)
    elif crit is not None and (not isinstance(crit, dict) or set(crit) - {"true", "false"}):
        errors.append("%s.criteria: noul aceita só {true: ..., false: ...}" % where)


def normalize_question(q: Dict[str, Any]) -> Dict[str, Any]:
    """Forma canônica usada no dataset e no treino (choice sempre como dict)."""
    out = {"type": q["type"], "instructions": q["instructions"]}
    crit = q.get("criteria")
    if q["type"] == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    if crit is not None:
        out["criteria"] = crit
    return out


def option_keys(q: Dict[str, Any]) -> List[str]:
    """Chaves de probabilidade do gold, na ordem das opções do Laya."""
    if q["type"] == "choice":
        crit = q["criteria"]
        return list(crit.keys()) if isinstance(crit, dict) else list(crit)
    if q["type"] == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    return ["false", "true"]


def load_task(root: str = ".") -> Task:
    root = os.path.abspath(root)
    path = os.path.join(root, "task.yaml")
    if not os.path.exists(path):
        raise TaskError("task.yaml não encontrado em %s" % root)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    errors: List[str] = []
    name = raw.get("name")
    if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,48}", name or ""):
        errors.append("name: slug obrigatório (a-z, 0-9, -), ex.: severidade-incidentes")
    version = str(raw.get("version", ""))
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        errors.append("version: use semver, ex.: 0.1.0")

    language = raw.get("language", DEFAULTS["language"])
    if language != "pt":
        errors.append("language: só 'pt' é suportado por enquanto")

    base_model = _merge(DEFAULTS["base_model"], raw.get("base_model"))
    context = _merge(DEFAULTS["context"], raw.get("context"))
    max_len, head_max_len = context.get("max_len"), context.get("head_max_len")
    if not isinstance(max_len, int) or not 128 <= max_len <= 8192:
        errors.append("context.max_len: inteiro entre 128 e 8192")
    if not isinstance(head_max_len, int) or not 32 <= head_max_len <= 1024:
        errors.append("context.head_max_len: inteiro entre 32 e 1024")
    elif isinstance(max_len, int) and head_max_len >= max_len:
        errors.append("context.head_max_len deve ser menor que context.max_len")

    questions = raw.get("questions")
    if not isinstance(questions, dict) or not questions:
        errors.append("questions: defina ao menos uma pergunta")
        questions = {}
    for qid, q in questions.items():
        _check_question(str(qid), q, errors)

    teacher = _merge(DEFAULTS["teacher"], raw.get("teacher"))
    if teacher["provider"] not in TEACHER_PROVIDERS:
        errors.append("teacher.provider: um de %s" % ", ".join(TEACHER_PROVIDERS))
    if teacher["provider"] == "openrouter" and not teacher.get("model"):
        errors.append("teacher.model: obrigatório com openrouter")
    if not os.path.exists(os.path.join(root, teacher["prompt"])):
        errors.append("teacher.prompt: arquivo %s não existe" % teacher["prompt"])
    if not isinstance(teacher["concurrency"], int) or not 1 <= teacher["concurrency"] <= 16:
        errors.append("teacher.concurrency: inteiro entre 1 e 16")

    data = _merge(DEFAULTS["data"], raw.get("data"))
    if not 0.05 <= float(data["eval_fraction"]) <= 0.5:
        errors.append("data.eval_fraction: entre 0.05 e 0.5")

    train = _merge(DEFAULTS["train"], raw.get("train"))
    for key in ("epochs", "micro_batch", "grad_accum"):
        if not isinstance(train[key], int) or train[key] < 1:
            errors.append("train.%s: inteiro >= 1" % key)

    goals = _merge(DEFAULTS["goals"], raw.get("goals"))
    publish = _merge(DEFAULTS["publish"], raw.get("publish"))

    if errors:
        raise TaskError("task.yaml inválido:\n  - " + "\n  - ".join(errors))

    return Task(
        root=root, name=name, version=version, description=str(raw.get("description", "")),
        language=language, base_model=base_model, context=context,
        questions={qid: normalize_question(q) for qid, q in questions.items()},
        teacher=teacher, data=data, train=train, goals=goals, publish=publish, raw=raw,
    )
