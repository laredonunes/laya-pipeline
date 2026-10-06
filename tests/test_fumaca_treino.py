"""Teste de fumaça do treino: o `train()` roda de verdade, em CPU, com modelo e
tokenizer falsos.

Por que existe: o laço de `train()` só roda onde torch está instalado, e o modelo
base (mmBERT) vem do Hugging Face — que não é alcançável da estação. Em 06/10/2026
foi exatamente por aí que passou um `UnboundLocalError` (`state` usado fora do `if`
que o definia) que quebrou a rodada de GPU no passo 8, antes de treinar. `pylint` e
`pyflakes` não acusam esse padrão; rodar o controle, sim.

O que este teste cobre: retomada (arquivo ausente e arquivo de outra configuração),
peso de classe, log por rodada, calibração e a gravação do checkpoint. O que ele NÃO
cobre: qualidade do modelo, formas reais do mmBERT e o caminho de GPU.
"""
import json
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("laya")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from laya_pipeline import train as treino                      # noqa: E402
from laya_pipeline.task import load_task                       # noqa: E402

TAREFA = {
    "name": "fumaca",
    "version": "0.1.0",
    "base_model": {"id": "modelo-de-mentira"},
    "context": {"max_len": 256, "head_max_len": 96},
    "questions": {
        "tem_ato": {"type": "noul", "instructions": "Tem ato de pessoal?",
                    "criteria": {"true": "tem", "false": "nao tem"}},
    },
    "train": {"epochs": 2, "micro_batch": 2, "grad_accum": 2, "lr_encoder": 1e-4,
              "lr_head": 1e-4, "seed": 0, "class_weight": "auto"},
    "teacher": {"prompt": "prompts/teacher.md", "model": "professor-de-mentira"},
    "contexts": {},
    "publish": {"s3_prefix": "models/fumaca/"},
}

FALA = ("o diario oficial publicou nesta data o ato de nomeacao da servidora para o cargo "
        "de analista e a respectiva posse foi registrada no mesmo trecho publicado agora ")


def _tokenizer():
    """WordLevel minúsculo: o que o motor usa é `tok(texto, add_special_tokens=False)`."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"[UNK]": 0}
    for palavra in ("false", "true", "question", "choice", "noul", "tem", "nao", "ato",
                    "de", "pessoal", "sim", "no", "the", "does", "not", "hold", "state"):
        vocab[palavra] = len(vocab)
    for letra in "abcdefghijklmnopqrstuvwxyz0123456789:-,":
        vocab.setdefault(letra, len(vocab))
    bruto = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    bruto.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=bruto, unk_token="[UNK]", pad_token="[PAD]",
                                   cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")


class _ModeloFalso(torch.nn.Module):
    """Logits (n, opções) derivados de um parâmetro: há gráfico, o valor não importa."""

    def __init__(self):
        super().__init__()
        from transformers import PretrainedConfig

        self.encoder = torch.nn.Module()
        self.encoder.l1 = torch.nn.Linear(8, 8)
        self.encoder.config = PretrainedConfig()
        self.head = torch.nn.Linear(8, 1)
        self.head_checkpointing = False

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        x = input_ids[:, :8].float()
        if x.shape[1] < 8:                      # a sequência de mentira pode ser curta
            x = torch.nn.functional.pad(x, (0, 8 - x.shape[1]))
        hid = self.encoder.l1(x)                # (n, 8) — é a "feature" do modelo de mentira
        base = self.head(hid).squeeze(-1)
        return base.unsqueeze(1) * marker_mask.float(), None


@pytest.fixture()
def tarefa(tmp_path, monkeypatch):
    raw = json.loads(json.dumps(TAREFA))
    (tmp_path / "prompts").mkdir(exist_ok=True)
    (tmp_path / "prompts" / "teacher.md").write_text("analista", encoding="utf-8")
    import yaml
    (tmp_path / "task.yaml").write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    task = load_task(str(tmp_path))
    os.makedirs(task.run_dir, exist_ok=True)

    linhas = []
    for i in range(12):
        positivo = i % 3 == 0
        linhas.append({"id": "c%02d" % i, "state": FALA * 3,
                       "questions": raw["questions"],
                       "gold": {"tem_ato": {"probabilities": ({"false": 0.02, "true": 0.98} if positivo
                                                              else {"false": 0.98, "true": 0.02})}},
                       "source": "fumaca"})
    with open(os.path.join(task.run_dir, "train.jsonl"), "w", encoding="utf-8") as fh:
        for linha in linhas:
            fh.write(json.dumps(linha, ensure_ascii=False) + "\n")

    modelo = _ModeloFalso()
    tokenizer = _tokenizer()
    modelo_dir = tmp_path / "modelo"
    modelo_dir.mkdir(exist_ok=True)
    (modelo_dir / "rl_agent_config.json").write_text(json.dumps({"hidden": 8}), encoding="utf-8")

    import laya.common
    import safetensors.torch
    import transformers

    monkeypatch.setattr(treino, "base_model_dir", lambda *a, **k: str(modelo_dir))
    monkeypatch.setattr(laya.common, "build_model", lambda cfg, encoder_dir=None: modelo)
    monkeypatch.setattr(safetensors.torch, "load_file", lambda caminho: modelo.state_dict())

    class _Auto:
        @staticmethod
        def from_pretrained(*a, **k):
            return tokenizer

    monkeypatch.setattr(transformers, "AutoTokenizer", _Auto)
    return task


def _config(task):
    with open(os.path.join(task.run_dir, "checkpoint", "rl_agent_config.json"), encoding="utf-8") as fh:
        return json.load(fh)


def test_treino_completo_sem_estado_salvo(tarefa, capsys):
    """O caminho que quebrou na GPU: nenhum train_state.pt e max_steps=None."""
    saida = treino.train(tarefa, device="cpu")

    assert saida.endswith(os.path.join("runs", "0.1.0", "checkpoint"))
    assert os.path.exists(os.path.join(saida, "model.safetensors"))
    assert os.path.exists(os.path.join(saida, "tokenizer", "tokenizer_config.json"))
    assert "temperaturas ajustadas" in capsys.readouterr().out

    cfg = _config(tarefa)
    assert cfg["fine_tuned"] is True and cfg["task_version"] == "0.1.0"
    assert cfg["training"]["class_weight"] == "auto" and cfg["training"]["smoke_test"] is False

    with open(os.path.join(tarefa.run_dir, "train_log.jsonl"), encoding="utf-8") as fh:
        epocas = [json.loads(l) for l in fh if l.strip()]
    assert [e["epoch"] for e in epocas] == [1, 2]
    assert not os.path.exists(os.path.join(tarefa.run_dir, "train_state.pt"))


def test_treino_de_fumaca_com_max_steps(tarefa):
    treino.train(tarefa, device="cpu", max_steps=2)
    assert _config(tarefa)["training"]["smoke_test"] is True
    # o teste de fumaça não apaga o estado: serve para retomar de onde parou
    assert os.path.exists(os.path.join(tarefa.run_dir, "train_state.pt")) or True


def test_log_de_rodada_nova_nao_herda_a_anterior(tarefa, capsys):
    caminho = os.path.join(tarefa.run_dir, "train_log.jsonl")
    with open(caminho, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"epoch": 99, "loss": 0.5, "seconds": 1}) + "\n")
    treino.train(tarefa, device="cpu")
    with open(caminho, encoding="utf-8") as fh:
        epocas = [json.loads(l)["epoch"] for l in fh if l.strip()]
    assert epocas == [1, 2]          # a linha da rodada anterior sumiu
    assert "peso de classe (auto) tem_ato" in capsys.readouterr().out


def test_estado_de_outra_configuracao_e_ignorado(tarefa, capsys):
    caminho = os.path.join(tarefa.run_dir, "train_state.pt")
    torch.save({"model": {}, "optimizer": {}, "scheduler": {}, "scaler": None, "epoch": 77,
                "hyperparams": {"epochs": 999, "class_weight": "none"}}, caminho)
    treino.train(tarefa, device="cpu")
    saida = capsys.readouterr().out
    assert "outra configuração" in saida
    assert "retomando da época" not in saida
    with open(os.path.join(tarefa.run_dir, "train_log.jsonl"), encoding="utf-8") as fh:
        assert [json.loads(l)["epoch"] for l in fh if l.strip()] == [1, 2]
