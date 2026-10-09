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

## Ruído entre execuções idênticas e o modo determinístico (2026-10-07)

Duas execuções do **mesmo** treino — mesma entrega, mesmo `train.jsonl` (byte a byte),
mesma semente (`seed: 0`), mesma GPU (T4), sessões novas — divergiram em **20 dos 294
casos** de avaliação (6,8%). A acurácia de uma das réguas andou de 0,8696 para 0,9058,
**cruzando** a meta de 0,90: o número de uma execução isolada deixou de decidir portão.

Causa: o `train()` fixava a semente, mas treinava com `autocast` fp16 (o backward usa
somas atômicas) e sem nenhuma trava de kernel — o cuDNN escolhe algoritmo por heurística
e o cuBLAS por workspace disponível. A época 1 saía idêntica nas duas execuções; a
divergência começava no primeiro `optimizer.step()`.

Conserto: `train.deterministic: true` no `task.yaml` liga, **antes** de qualquer uso do
CUDA, `torch.use_deterministic_algorithms(True)`, `cudnn.deterministic = True`,
`cudnn.benchmark = False` e `CUBLAS_WORKSPACE_CONFIG=:4096:8`. O padrão continua `false`
— quem quer reprodutibilidade pede. Com `use_deterministic_algorithms(True)` estrito, uma
operação sem implementação determinística **derruba o treino nomeando a operação** (nos
primeiros segundos, porque todo o laço roda desde o primeiro micro-lote): é diagnóstico,
não defeito.

O custo em GPU ainda não foi medido — a conta do treino de 24 épocas era de ~145 s e o
modo determinístico só restringe a escolha de algoritmo; a primeira rodada com a trava
ligada dá o número.

## Onde o tempo de GPU vai (medido, não corrigido)

No mesmo treino de 24 épocas: **145 s** somados de conta (≈6 s/época) contra **23min42**
de parede numa execução e **33min25** na outra — a diferença é quase toda
`torch.save` do estado completo (modelo + otimizador + escalonador, ~3–4 GB) **ao fim de
cada época**, dentro do laço mas fora do cronômetro da época. Gravar o estado a cada N
épocas (com N > 1) encurtaria a rodada de 25 min para ~7 min ao custo de reprocessar até
N-1 épocas depois de uma queda. **Decisão pendente.**

## Testes automatizados

`pytest`: 24 testes — 19 em `tests/test_pipeline.py` (sem torch nem rede: validação do
`task.yaml`, parse da resposta do professor, estabilidade da divisão, métricas e portões) e
5 em `tests/test_fumaca_treino.py` (com torch em CPU, modelo e tokenizer falsos: o laço de
`train()` de verdade, retomada, log por rodada e o modo determinístico).
