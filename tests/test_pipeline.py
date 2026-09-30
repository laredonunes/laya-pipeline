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


# --- ingest: documentos e exemplos respondidos por uma pessoa -------------

ONE_Q = {"categoria": {"type": "choice", "instructions": "Qual a categoria?",
                       "criteria": {"acesso": "login", "rede": "internet"}}}


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_ingest_documents_and_examples(tmp_path):
    from laya_pipeline.ingest import ingest, load_human

    task = make_task(tmp_path, questions=ONE_Q)
    docs, ex = tmp_path / "data" / "documentos", tmp_path / "data" / "exemplos"
    _write(docs / "a.txt", "Minha senha expirou e não consigo entrar no sistema.")
    _write(docs / "sub" / "b.md", "A VPN cai toda hora desde a manhã de hoje.")
    _write(docs / "lote.csv", "id;texto\nc1;Impressora do terceiro andar parou de funcionar.\nc2;curto\n")
    _write(docs / "EXEMPLO-ignorar.txt", "Este arquivo é só um exemplo do template.")
    _write(docs / "README.md", "explicação da pasta")
    _write(ex / "acesso" / "1.txt", "Conta bloqueada após três tentativas de login.")
    _write(ex / "rede" / "EXEMPLO-1.txt", "Arquivo de exemplo do template, não conta.")
    (tmp_path / "data" / "texts.jsonl").write_text(
        json.dumps({"id": "s1", "state": "texto sintético", "source": "synthetic"}) + "\n", encoding="utf-8")

    out = ingest(task)
    assert out["documents"] == 3 and out["human_examples"] == 1 and out["synthetic"] == 1
    assert out["texts"] == 5
    assert out["ignored_template_samples"] == 2
    assert any("curto" in w for w in out["warnings"])
    human = load_human(task)
    assert list(human.values()) == [{"categoria": "acesso"}]
    ids = {json.loads(l)["id"] for l in (tmp_path / "data" / "texts.jsonl").read_text().splitlines()}
    assert "c1" in ids and "s1" in ids


def test_ingest_rejects_unknown_answer_folder(tmp_path):
    from laya_pipeline.ingest import ingest

    task = make_task(tmp_path, questions=ONE_Q)
    _write(tmp_path / "data" / "exemplos" / "hardware" / "1.txt", "Monitor não liga de jeito nenhum.")
    with pytest.raises(SystemExit) as err:
        ingest(task)
    assert "hardware" in str(err.value) and "acesso, rede" in str(err.value)


def test_human_examples_always_go_to_eval_and_agreement(tmp_path):
    from laya_pipeline.ingest import ingest

    task = make_task(tmp_path, questions=ONE_Q, data={"eval_fraction": 0.05})
    for i in range(6):
        _write(tmp_path / "data" / "exemplos" / "acesso" / ("%d.txt" % i), "Senha bloqueada, caso número %d." % i)
    ingest(task)
    texts = [json.loads(l) for l in (tmp_path / "data" / "texts.jsonl").read_text().splitlines()]
    os.makedirs(task.run_dir)
    with open(os.path.join(task.run_dir, "labeled.jsonl"), "w", encoding="utf-8") as f:
        for i, t in enumerate(texts):  # professor discorda em 2 dos 6
            p = {"acesso": 0.2, "rede": 0.8} if i < 2 else {"acesso": 0.9, "rede": 0.1}
            f.write(json.dumps({"id": t["id"], "state": t["state"], "gold": {"categoria": {"probabilities": p}}}) + "\n")
    n_train, n_eval = dataset.split(task)
    assert (n_train, n_eval) == (0, 6)
    rows = [json.loads(l) for l in open(os.path.join(task.run_dir, "eval.jsonl"), encoding="utf-8")]
    assert all(r["human"] == {"categoria": "acesso"} for r in rows)
    agreement = dataset.teacher_vs_human(task, rows)
    assert agreement["cases"] == 6 and agreement["agree"] == 4
    assert "Concordância: **4 de 6" in open(dataset.sample(task, n=2), encoding="utf-8").read()
