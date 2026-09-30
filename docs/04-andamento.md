# Andamento

## Feito

- [x] Discussão e decisões ([01-objetivo-e-decisoes.md](01-objetivo-e-decisoes.md)) — 2026-09-30
- [x] Leitura do pacote `laya` 0.3.5 e da receita oficial de fine-tuning
- [x] Medição de latência em CPU com 512/1024/2560 tokens
- [x] `laya-pipeline` v0.1.0: CLI, módulos, notebook do Studio Lab, testes
- [x] `laya-template`: `task.yaml` de exemplo, prompts, skill `laya-task`
- [x] Teste de fumaça do fluxo inteiro em CPU
- [x] Repositórios no GitHub — 2026-09-30
- [x] v0.2.0: entrada pela pessoa (`FORMULARIO.md`, `data/documentos/`,
      `data/exemplos/<resposta>/`), comando `ingest`, exemplos humanos só na
      avaliação e concordância professor x humano na amostra e no relatório — 2026-09-30

- [x] v0.3.0: professor direto no Amazon Bedrock (`teacher.provider: bedrock`),
      uso de tokens em `teacher_usage.json`, geração mais robusta (JSON com
      quebras de linha ou aspas, limite de tokens maior, sem markdown) — 2026-09-30

## Modelo de teste em andamento

`laya-teste-tipo-documento` (privado): tipo de documento administrativo,
6 opções, 2560 tokens, 1.000 textos sintéticos, professor DeepSeek V3.2 no
Bedrock. Primeiros números (2026-09-30):

- documento de 1.444 palavras = 2.296 tokens (o limite do texto na janela);
- professor x 12 exemplos: 12/12;
- geração: ~2.300 tokens de saída por documento; rotulagem: ~1.150 tokens de
  entrada por texto curto.

## Próximos passos (1º modelo, à mão)

1. [ ] **Pessoa:** criar `laya-<tarefa>` do template, preencher o `FORMULARIO.md`,
       colocar os textos e os exemplos respondidos, e mandar o link.
2. [ ] Agente: `task.yaml` e `prompts/` a partir do formulário → aprovação da especificação.
3. [ ] `ingest` e teste do professor contra os exemplos humanos (meta ≥ ~85% de concordância).
4. [ ] Rotular tudo (+ sintéticos, se permitido). 5 a 10 mil exemplos cabem numa sessão de T4.
5. [ ] Aprovação da amostra.
6. [ ] Baseline, treino no Studio Lab e avaliação.
7. [ ] Aprovação do relatório.
8. [ ] Publicar no S3.
9. [ ] No `provedor-ia`: build da imagem a partir do S3, endpoint `hermes-laya-<tarefa>`, rota `/laya/<tarefa>/invocations`, skill e `/context`.
10. [ ] Medir a latência real do modelo treinado no serverless (cold start + 2560 tokens).

## Depois

- [ ] 2º modelo: novo repositório do template, mesmos comandos.
- [ ] Automação: integrar o pipeline a uma fila (página na Lambda + fila/Step Functions + SageMaker Training Job).
  - verificar a cota de `ml.g4dn.xlarge` para treino spot (costuma vir zerada);
  - verificar as permissões do usuário IAM que opera a conta.
- [ ] Decidir entre um endpoint por modelo ou um endpoint com vários checkpoints.

## Pendências e riscos

- **`laya-pipeline` público só temporariamente** (desde 2026-09-30), para o
  Studio Lab/Kaggle instalarem o motor por `pip install git+https` sem
  token durante o treino manual. **Voltar a privado** ao integrar a fila;
  a partir daí a instalação passa a usar token de leitura ou pacote interno.
  Enquanto público, não versionar aqui IDs de conta, URLs do proxy, nomes
  de bucket reais nem nada específico do TCE.
- **Cold start ~54 s** contra um limite de ~60 s por chamada, somado a ~10 s
  de inferência com 2560 tokens. Pode exigir concorrência provisionada, que
  tem custo fixo.
- **Qualidade do professor:** é o teto do modelo. A revisão da amostra é o
  que protege contra rótulos ruins.
