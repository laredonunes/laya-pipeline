# Arquitetura

## Visão geral

```
laya-template ──(Use this template)──► laya-<tarefa>/      task.yaml, prompts/, reports/
                                            │
                                            │ laya-pipeline <comando>
                                            ▼
laya-pipeline (motor) ── professor LLM e S3 via hermes-usage-proxy (provedor-ia)
                                            │
                                            ▼ publish
s3://hermes-files-<conta>/models/laya/<tarefa>/<versão>/
        checkpoint/  manifest.json  report.md
                                            │
                                            ▼
provedor-ia: imagem de inferência → endpoint hermes-laya-<tarefa> → /laya/<tarefa>/invocations
```

Todo acesso externo do pipeline (LLM professor e S3) passa pelo
hermes-usage-proxy. Só são necessárias as variáveis `HERMES_PROXY_URL` e
`HERMES_TOKEN`. Não há credenciais AWS nem chaves de LLM no pipeline.

## Etapas

`laya-pipeline status` mostra o que já foi feito e o próximo passo.

| # | Onde | Etapa | Saída |
|---|---|---|---|
| 0 | **pessoa** | preenche `FORMULARIO.md`, `data/documentos/`, `data/exemplos/<resposta>/` e manda o link | entrada |
| 1 | agente | `task.yaml` e `prompts/` a partir do formulário, `validate` | especificação, aprovada pela pessoa |
| 2 | estação | `ingest` (+ `gen-texts` se faltarem textos) | `data/texts.jsonl`, `data/human.jsonl` |
| 3 | estação | `label` | `runs/<v>/labeled.jsonl` (retomável) |
| 4 | estação | `split` + `sample` | `train.jsonl`, `eval.jsonl`, `sample.md` |
| 5 | **pessoa** | `approve data` | portão 1 |
| 6 | estação | `sync push` | `runs/<v>` no S3 |
| 7 | GPU | `sync pull` → `eval` → `train` → `eval --trained` → `sync push` | checkpoint + `report.md` |
| 8 | **pessoa** | `approve report` | portão 2 |
| 9 | GPU | `publish` | checkpoint + manifesto no S3, `reports/<v>.md` no repo da tarefa |

## Entrada da pessoa (repositório da tarefa)

A pessoa não escreve YAML nem JSONL:

- **`FORMULARIO.md`**: a tarefa em perguntas e respostas (o que decide,
  opções, desempates, origem dos textos, meta). O agente deriva dele o
  `task.yaml` e os `prompts/`.
- **`data/documentos/`**: textos sem resposta (`.txt`/`.md`, `.csv` com
  coluna `texto`, ou `.jsonl`). O professor rotula.
- **`data/exemplos/<resposta>/`**: textos já respondidos pela pessoa. Ficam
  fora do treino e medem o professor e o modelo contra uma pessoa.

Arquivos `EXEMPLO-*` (os modelos que vêm no template) são ignorados.

## Módulos

| Módulo | Função | torch? |
|---|---|---|
| `task.py` | lê e valida o `task.yaml`, com defaults | não |
| `ingest.py` | `data/documentos/` e `data/exemplos/<resposta>/` → `texts.jsonl` e `human.jsonl` | não |
| `teacher.py` | gera textos e rotula com probabilidades via proxy | não |
| `proxy.py` | cliente do hermes-usage-proxy (LLM e URLs pré-assinadas do S3) | não |
| `dataset.py` | divisão treino/avaliação estável por hash do id (exemplos humanos sempre na avaliação), amostra, concordância professor x humano | não |
| `state.py` | etapas concluídas e portões de aprovação | não |
| `sync.py` | envia e baixa `runs/<v>` pelo S3 | não |
| `metrics.py` | acurácia, Brier, ECE e metas | não |
| `hf.py` | carregamento do mmBERT (correção de config do tokenizer) | sim |
| `evaluate.py` | baseline e modelo treinado, latência | sim |
| `train.py` | receita RLCD + calibração, acumulação de gradiente, retomada por época | sim |
| `publish.py` | checkpoint + manifesto no S3 | sim |

Os módulos sem torch rodam em ambiente mínimo, como uma futura Lambda.

## Contratos

Mudar qualquer um exige subir a versão do motor e criar a tag `vX.Y.Z`
(detalhes em `CLAUDE.md`):

1. chaves e defaults do `task.yaml`;
2. formato do dataset: `{id, state, questions, gold}`, igual ao do upstream;
3. layout no S3: `models/laya/<tarefa>/<versão>/checkpoint/` + `manifest.json`;
4. checkpoint carregável por `laya.load(<dir>)` do laya 0.3.5.

## Automação futura

Ainda não implementada (ver [04-andamento.md](04-andamento.md)):

```
Página (Lambda Function URL, com token)
  └─► Step Functions
        1. validar task.yaml do repositório
        2. gerar e rotular dados                 (job longo)
        3. ⏸ aprovação da amostra               (botão na página)
        4. baseline
        5. treino: SageMaker Training Job, spot, GPU
        6. avaliação → relatório
        7. ⏸ aprovação do relatório
        8. publicar + deploy do endpoint
```

A Lambda só dispara: tem limite de 15 min e não tem GPU. O Step Functions
conduz o processo e espera as aprovações de forma nativa.
