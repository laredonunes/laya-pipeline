# laya-pipeline

Motor de treinamento de modelos **Laya dedicados**: um classificador pequeno e
calibrado, especializado numa única tarefa, com contexto longo (até 2560
tokens por padrão) e em português.

O motor é um só. Cada modelo vive num repositório próprio, criado a partir do
[`laya-template`](https://github.com/laredonunes/laya-template), que contém apenas o que muda por tarefa.

Objetivo, decisões, medições e andamento do projeto: [`docs/`](docs/README.md).

```
laya-template  ──(clonar/usar como template)──►  laya-<tarefa>/   task.yaml, prompts/, reports/
                                                      │
                                                      │  laya-pipeline <comando>
                                                      ▼
laya-pipeline  (este repo: rotular, dividir, avaliar, treinar, publicar)
                                                      │
                                                      ▼
s3://hermes-files-…/models/laya/<tarefa>/<versão>/   ──►  provedor-ia monta o endpoint
```

## Fluxo

| # | Onde | Comando | Resultado |
|---|---|---|---|
| 0 | **humano** | preenche `FORMULARIO.md`, `data/documentos/` e `data/exemplos/<resposta>/` no repositório da tarefa | entrada |
| 1 | estação | agente escreve `task.yaml` e `prompts/` do formulário → `laya-pipeline validate` | especificação (aprovada pelo humano) |
| 2 | estação | `laya-pipeline ingest` (+ `gen-texts --n 3000` se faltarem textos) | `data/texts.jsonl`, `data/human.jsonl` |
| 3 | estação | `laya-pipeline label` | `runs/<v>/labeled.jsonl` (professor LLM, retomável) |
| 4 | estação | `laya-pipeline split` + `sample` | `train.jsonl`, `eval.jsonl`, `sample.md` (com a concordância professor x humano) |
| 5 | **humano** | revisar `sample.md` → `laya-pipeline approve data --by <nome>` | portão 1 |
| 6 | estação | `laya-pipeline sync push` | dados no S3 |
| 7 | GPU | `sync pull` → `eval` → `train` → `eval --trained` → `sync push` | checkpoint + `report.md` |
| 8 | **humano** | revisar `report.md` → `laya-pipeline approve report --by <nome>` | portão 2 |
| 9 | GPU | `laya-pipeline publish` | checkpoint + manifesto no S3, `reports/<v>.md` no repo da tarefa |

`laya-pipeline status` mostra o que já foi feito e qual é o próximo passo.

Os exemplos respondidos por uma pessoa (`data/exemplos/`) vão sempre para a
avaliação, nunca para o treino. Eles medem o professor antes do treino (na
`sample.md`) e o modelo depois (no `report.md`, "Contra respostas humanas").

Na GPU (passo 7), use `notebooks/studiolab.ipynb` (SageMaker Studio Lab, T4);
funciona igual no Kaggle ou em qualquer máquina com CUDA.

## Instalação

```bash
pip install -e .            # validar, rotular, dividir (sem torch)
pip install -e '.[train]'   # avaliar e treinar (torch + laya)
pip install -e '.[dev]' && pytest
```

Variáveis de ambiente: `HERMES_PROXY_URL` e `HERMES_TOKEN`. Por padrão, todo
acesso externo (LLM professor e S3) passa pelo hermes-usage-proxy do
`provedor-ia`, sem credenciais AWS nem chaves de LLM.

**Professor no Amazon Bedrock:** com `teacher.provider: bedrock` no
`task.yaml`, a rotulagem e a geração chamam o Bedrock direto (API Converse),
com as credenciais AWS do ambiente. Instale com `pip install -e '.[bedrock]'`.
Padrão: `model: deepseek.v3.2`, `region: sa-east-1`. Esse uso não passa pelo
`/usage` do proxy; os tokens ficam em `runs/<versão>/teacher_usage.json`.

## Como funciona o treino

Receita RLCD do upstream (`NandhaKishorM/laya`, Apache-2.0 — ver `NOTICE`):
o modelo aprende a **distribuição de probabilidade** do professor (não só o
rótulo), com recompensa por regras de pontuação próprias + entropia cruzada
suave; depois ajusta uma temperatura de calibração por tipo de pergunta numa
fatia separada antes do treino. Mudanças: contexto e hiperparâmetros do
`task.yaml`, acumulação de gradiente e retomada por época.

Base padrão: checkpoint `multilingual` (`jhu-clsp/mmBERT-base`, treinado com
1024 tokens; o encoder aceita 8192).

## Latência em CPU (produção serverless)

`bench/latency.py` mede o modelo base em CPU limitada (simula o SageMaker
Serverless). A latência cresce com **tokens × número de perguntas**: cada
pergunta é uma sequência separada, e o texto é codificado de novo para cada
uma. Um modelo dedicado com **uma pergunta** é o caso mais barato.

```bash
docker run --rm --entrypoint "" --cpus=2 --memory=3g -e LAYA_THREADS=2 \
  -v "$PWD/bench/latency.py:/bench.py:ro" hermes-laya:local python /bench.py
```

Fixe `LAYA_THREADS` no número de CPUs do limite: `os.cpu_count()` enxerga os
núcleos da máquina, e threads a mais derrubam o desempenho (6,4 s → 0,96 s
para 512 tokens com 2 CPUs).
