import json
import os

import numpy as np
import pytest
import yaml

from laya_pipeline import dataset, state
from laya_pipeline.metrics import QuestionMetrics, ece_score, overall
from laya_pipeline.task import TaskError, load_task, option_keys
from laya_pipeline.teacher import output_schema, parse_gold, render_questions

TASK = {
    "name": "severidade-incidentes",
    "version": "0.1.0",
    "description": "teste",
    "questions": {
        "severidade": {"type": "score", "instructions": "Qual a severidade?",
                       "criteria": ["baixa", "média", "alta"]},
        "tipo": {"type": "choice", "instructions": "Qual o tipo?",
                 "criteria": {"vazamento": "exfiltração", "malware": None}},
        "escalar": {"type": "noul", "instructions": "Escalar?"},
    },
}


def make_task(tmp_path, **overrides):
    raw = json.loads(json.dumps(TASK))
    raw.update(overrides)
    (tmp_path / "prompts").mkdir(exist_ok=True)
    (tmp_path / "prompts" / "teacher.md").write_text("Você é um analista.", encoding="utf-8")
    (tmp_path / "task.yaml").write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return load_task(str(tmp_path))


def test_defaults_and_paths(tmp_path):
    task = make_task(tmp_path)
    assert task.context == {"max_len": 2560, "head_max_len": 256}
    assert task.base_model["subfolder"] == "multilingual"
    assert task.run_dir.endswith(os.path.join("runs", "0.1.0"))
    assert task.s3_key_prefix == "models/laya/severidade-incidentes/0.1.0/"


def test_choice_list_is_normalized(tmp_path):
    q = dict(TASK["questions"])
    q["tipo"] = {"type": "choice", "instructions": "Tipo?", "criteria": ["a", "b"]}
    task = make_task(tmp_path, questions=q)
    assert task.questions["tipo"]["criteria"] == {"a": None, "b": None}


def test_validation_lists_every_error(tmp_path):
    with pytest.raises(TaskError) as err:
        make_task(tmp_path, name="Nome Ruim", version="1", context={"max_len": 200, "head_max_len": 256},
                  questions={"x": {"type": "choice", "instructions": "?", "criteria": ["só uma"]}})
    msg = str(err.value)
    for fragment in ("name:", "version:", "head_max_len deve ser menor", "pelo menos 2 opções"):
        assert fragment in msg


def test_option_keys():
    assert option_keys({"type": "score", "criteria": ["a", "b", "c"]}) == ["0", "1", "2"]
    assert option_keys({"type": "noul"}) == ["false", "true"]
    assert option_keys({"type": "choice", "criteria": {"x": 1, "y": 2}}) == ["x", "y"]


def test_parse_gold_normalizes_and_rejects(tmp_path):
    task = make_task(tmp_path)
    content = ('Claro! {"severidade": {"0": 0.1, "1": 0.3, "2": 0.6}, '
               '"tipo": {"vazamento": 2, "malware": 2}, "escalar": {"true": 0.9, "false": 0.1}}')
    gold = parse_gold(task, content)
    assert gold["tipo"]["probabilities"] == {"vazamento": 0.5, "malware": 0.5}
    assert gold["severidade"]["probabilities"]["2"] == 0.6
    with pytest.raises(ValueError):
        parse_gold(task, '{"severidade": {"0": 1}}')
    with pytest.raises(ValueError):
        parse_gold(task, content.replace("0.9", "-1"))


def test_prompt_rendering_mentions_every_key(tmp_path):
    task = make_task(tmp_path)
    block, schema = render_questions(task), output_schema(task)
    for key in ('"vazamento"', '"malware"', '"0"', '"2"', '"true"', '"false"'):
        assert key in block and key in schema


def _write_labeled(task, n):
    os.makedirs(task.run_dir, exist_ok=True)
    with open(os.path.join(task.run_dir, "labeled.jsonl"), "w", encoding="utf-8") as f:
        for i in range(n):
            gold = {"severidade": {"probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}},
                    "tipo": {"probabilities": {"vazamento": 0.8, "malware": 0.2}},
                    "escalar": {"probabilities": {"false": 0.3, "true": 0.7}}}
            f.write(json.dumps({"id": "caso%d" % i, "state": "texto %d" % i, "state_tokens": 10,
                                "questions": task.questions, "gold": gold}) + "\n")


def test_split_is_stable_when_dataset_grows(tmp_path):
    task = make_task(tmp_path)
    _write_labeled(task, 200)
    dataset.split(task)
    first_eval = {json.loads(l)["id"] for l in open(os.path.join(task.run_dir, "eval.jsonl"))}
    _write_labeled(task, 400)
    n_train, n_eval = dataset.split(task)
    second_eval = {json.loads(l)["id"] for l in open(os.path.join(task.run_dir, "eval.jsonl"))}
    assert first_eval <= second_eval
    assert n_train + n_eval == 400 and 10 < n_eval < 80


def test_sample_and_stats(tmp_path):
    task = make_task(tmp_path)
    _write_labeled(task, 30)
    path = dataset.sample(task, n=5)
    text = open(path, encoding="utf-8").read()
    assert text.count("### ") == 5 and "vazamento=30" in text


def test_metrics():
    m = QuestionMetrics("score", ["0", "1", "2"])
    m.add({"0": 0.0, "1": 0.0, "2": 1.0}, {"0": 0.0, "1": 0.0, "2": 1.0})
    m.add({"0": 1.0, "1": 0.0, "2": 0.0}, {"0": 0.0, "1": 0.0, "2": 1.0})
    s = m.summary()
    assert s["accuracy"] == 0.5 and s["score_mae"] == 1.0 and s["brier"] == 1.0
    assert m.confusion()["2"] == {"0": 1, "1": 0, "2": 1}
    assert ece_score(np.array([1.0, 1.0]), np.array([1.0, 0.0])) == pytest.approx(0.5)
    assert overall({"a": {"n": 1, "accuracy": 1.0}, "b": {"n": 3, "accuracy": 0.0}})["accuracy"] == 0.25


def test_gates(tmp_path):
    task = make_task(tmp_path)
    with pytest.raises(SystemExit):
        state.require(task, "data")
    state.approve(task, "data", by="revisor")
    state.require(task, "data")
    steps = {s["step"]: s["done"] for s in state.status(task)}
    assert steps["approve data"] and not steps["approve report"]
