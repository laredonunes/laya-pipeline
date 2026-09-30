# laya-pipeline — instruções para o agente

Motor único de treino de modelos Laya dedicados. **Não coloque nada
específico de uma tarefa aqui** — isso vai no repositório da tarefa
(criado do `laya-template`). Se uma tarefa precisa de algo que o motor não
faz, generalize a funcionalidade e exponha no `task.yaml`.

## Contratos que não podem quebrar sem subir a versão

1. **`task.yaml`** (`laya_pipeline/task.py`): chaves e defaults. Chave nova
   precisa de default que mantenha o comportamento antigo.
2. **Formato do dataset** (`{id, state, questions, gold}`, gold =
   `{qid: {probabilities: {opção: p}}}`) — o mesmo do upstream.
3. **Layout no S3** (`models/laya/<tarefa>/<versão>/checkpoint/...` +
   `manifest.json`): o `provedor-ia` monta a imagem de inferência a partir dele.
4. **Checkpoint carregável por `laya.load(<dir>)`** do laya 0.3.5, com
   `temperature_by_options` removido do config.

Mudou algum? Suba `__version__` + `pyproject.toml` e crie a tag `vX.Y.Z`;
os repositórios de tarefa fixam a tag no `requirements.txt`.

## Regras

- Etapas sem GPU (`task`, `teacher`, `dataset`, `state`, `sync`, `metrics`)
  não importam torch no topo do módulo: precisam rodar em ambiente mínimo
  (futura Lambda/Step Functions).
- Portões `data` e `report` são obrigatórios em `train` e `publish`; só o
  teste de fumaça (`train --max-steps`) passa sem o portão `data`, e o
  checkpoint dele é recusado pelo `publish`.
- A divisão treino/avaliação é por hash do id: nunca troque para aleatória
  (vazaria avaliação para o treino entre versões).
- Testes: `pytest` (sem torch nem rede). Teste de fumaça do treino em CPU:
  ver `bench/` e a skill do template.
- Código adaptado do upstream mantém a atribuição (NOTICE + docstring).
