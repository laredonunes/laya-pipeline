# Medições

## Latência em CPU do modelo base (2026-09-30)

Checkpoint `multilingual` na imagem `hermes-laya:local`, Docker limitado a
2 vCPU e 3 GB, com `LAYA_THREADS=2`. Simula o SageMaker Serverless.
Script: `bench/latency.py`.

| max_len | perguntas | tokens codificados | mediana | máx. |
|---|---|---|---|---|
| 512 | 1 | 512 | 0,96 s | 0,98 s |
| 512 | 3 | 1536 | 3,14 s | 3,38 s |
| 1024 | 1 | 1024 | 2,43 s | 2,46 s |
| 1024 | 3 | 3072 | 7,27 s | 7,34 s |
| 2560 | 1 | 2560 | 10,56 s | 11,01 s |

- A latência cresce com **tokens × perguntas**.
- Com 2560 tokens: 2 perguntas ≈ 20 s e ~3 GB. Com 3, estoura a memória.
- O serverless corta a chamada em ~60 s, e o cold start atual é ~54 s. A
  primeira chamada depois de ociosidade continua sendo o ponto crítico.
- Fixar as threads no número de CPUs do limite é essencial. Sem isso,
  `os.cpu_count()` enxerga os núcleos da máquina: 512 tokens foi de 6,4 s
  para 0,96 s.

## Teste de fumaça do pipeline (2026-09-30)

Objetivo: provar que as etapas se encadeiam, não medir qualidade.

- **Configuração:** tarefa de exemplo do template (triagem de chamados,
  5 categorias), 60 textos, 1 época, `max_len` 384, CPU.
- **Resultado:** todas as etapas rodaram até o relatório. A época de treino
  levou 51 s e gerou um checkpoint de 644 MB.

| | casos | acurácia | Brier | ECE |
|---|---|---|---|---|
| baseline | 5 | 1,0 | 0,028 | 0,222 |
| treinado | 5 | 0,6 | 0,245 | 0,356 |

O modelo treinado piorou, o que é esperado com 51 exemplos e 1 época. Isso
também mostra que os portões funcionam: as metas do `task.yaml` não foram
atingidas e o `publish` recusaria o checkpoint.

## Testes automatizados

`pytest`: 10 testes, sem torch nem rede. Cobrem a validação do
`task.yaml`, o parse da resposta do professor, a estabilidade da divisão,
as métricas e os portões.
