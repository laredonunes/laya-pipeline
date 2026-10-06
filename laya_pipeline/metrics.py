"""Métricas de avaliação (só numpy, testáveis sem torch).

- accuracy: opção decidida do modelo == a do professor. A decisão é o argmax,
  salvo quando a pergunta declara `threshold` no task.yaml (pergunta `noul`):
  então `true` vale quando p(true) >= threshold. O erro caro de um portão é o
  falso negativo, e o corte é do dono do dado, não do motor;
- brier: erro quadrático entre a distribuição do modelo e a do professor;
- ece: erro de calibração — quando o modelo diz 80% de confiança, acerta ~80%?
- score_mae: para perguntas `score`, distância média entre o nível esperado
  do modelo e o do professor."""

import math
from typing import Dict, List, Optional, Sequence

import numpy as np


def ece_score(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    """Igual a `laya.common.ece_score`, para os números baterem com os do upstream."""
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(e)


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), p))


class QuestionMetrics:
    """Acumula predições e gabarito de uma pergunta.

    `threshold` é o ponto de operação da pergunta `noul` (chave `threshold` do
    task.yaml): a decisão passa a ser `true` quando p(true) >= threshold, em vez
    do argmax 0,5. Existe porque o erro caro da tarefa é o falso NEGATIVO e o
    argmax calibrado no prior do treino erra justamente por baixo. `None` (o
    padrão) mantém o argmax de sempre.
    """

    def __init__(self, qtype: str, keys: List[str], threshold: Optional[float] = None):
        self.qtype, self.keys = qtype, keys
        self.threshold = threshold
        self._true = keys.index("true") if "true" in keys else None
        self.pred: List[np.ndarray] = []
        self.gold: List[np.ndarray] = []

    def add(self, pred: Dict[str, float], gold: Dict[str, float]) -> None:
        self.pred.append(np.array([pred.get(k, 0.0) for k in self.keys], dtype=float))
        self.gold.append(np.array([gold.get(k, 0.0) for k in self.keys], dtype=float))

    def decision(self, p: np.ndarray) -> int:
        """Índice da opção decidida: limiar na pergunta `noul`, senão argmax."""
        if self.threshold is not None and self._true is not None:
            return self._true if p[self._true] >= self.threshold else 1 - self._true
        return int(p.argmax())

    def decide(self, pred: Dict[str, float]) -> str:
        """Chave da opção decidida (mesma regra do `summary`)."""
        return self.keys[self.decision(np.array([pred.get(k, 0.0) for k in self.keys], dtype=float))]

    def summary(self) -> Dict[str, float]:
        if not self.pred:
            return {"n": 0}
        p, g = np.stack(self.pred), np.stack(self.gold)
        decided = np.array([self.decision(x) for x in p])
        correct = (decided == g.argmax(1)).astype(float)
        out = {
            "n": int(len(p)),
            "accuracy": round(float(correct.mean()), 4),
            "brier": round(float(((p - g) ** 2).sum(1).mean()), 4),
            "ece": round(ece_score(p.max(1), correct), 4),
        }
        # O ECE continua sendo o da distribuição (confiança x acerto no ponto de
        # operação); o limiar fica registrado para o relatório se explicar.
        if self.threshold is not None:
            out["threshold"] = self.threshold
        if self.qtype == "score":
            levels = np.arange(len(self.keys))
            out["score_mae"] = round(float(np.abs(p @ levels - g @ levels).mean()), 4)
        return out

    def confusion(self) -> Dict[str, Dict[str, int]]:
        """confusion[professor][modelo] = contagem."""
        table = {k: {kk: 0 for kk in self.keys} for k in self.keys}
        for p, g in zip(self.pred, self.gold):
            table[self.keys[int(g.argmax())]][self.keys[self.decision(p)]] += 1
        return table


def overall(per_question: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Média ponderada pelo número de casos de cada pergunta."""
    total = sum(m.get("n", 0) for m in per_question.values())
    if not total:
        return {}
    out = {}
    for key in ("accuracy", "brier", "ece"):
        vals = [(m[key], m["n"]) for m in per_question.values() if key in m and not math.isnan(m[key])]
        if vals:
            out[key] = round(sum(v * n for v, n in vals) / sum(n for _, n in vals), 4)
    return out
