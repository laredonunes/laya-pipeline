# Objetivo e decisões

## Objetivo

Treinar **dois modelos Laya dedicados**, um de cada vez. Cada um faz uma
única tarefa de classificação em **português**, com:

- **contexto maior:** de ~500 tokens (checkpoint padrão) para ~2500;
- **acurácia melhor** que o Laya genérico na tarefa.

Os dados não são sensíveis.

O Laya (`convaiinnovations/laya`) é um encoder com uma cabeça de calibração.
Recebe `{state, questions}` e devolve uma probabilidade calibrada por opção.
Já roda em produção no endpoint serverless `hermes-laya-endpoint`, servido
pelo `provedor-ia` na rota `/laya/`.

## Decisões

### Modelo base: checkpoint `multilingual` (mmBERT-base)

- O checkpoint padrão é `ModernBERT-large`, só em inglês, com
  `max_len: 512`. É daí que vinham os ~500 tokens.
- O `multilingual` cobre português, foi treinado com 1024 tokens e o encoder
  aceita até 8192 posições. Também é menor, o que pesa na CPU do serverless
  e na cota de 3072 MB.
- Encoders só de português, como o BERTimbau, param em 512 tokens e não
  estão no formato que o `laya.load` carrega.

### Contexto de treino: `max_len` 2560

O limite de 500 tokens é configuração, não do modelo. Mas usar 2560 sem
treinar tende a piorar a acurácia, porque o modelo só viu até 1024. Treinar
já com 2560 resolve o contexto e a acurácia de uma vez.

### Dados por destilação

A receita de treino do upstream (RLCD) não usa só o rótulo: cada exemplo
precisa da **distribuição de probabilidade de um professor** em cada opção.

- **Professor:** um LLM que já temos no proxy (DeepSeek, Qwen ou
  OpenRouter). Ele rotula os textos com probabilidades.
- **Textos:** reais, se houver. Se não, sintéticos, gerados pelo próprio LLM
  (`gen-texts`).
- **Teto:** o modelo não fica melhor que o professor. O ganho é responder
  em fração de segundo e quase de graça, onde antes seria uma chamada a LLM.

Pela documentação do upstream, quase todo o ganho vem do fine-tuning: o
modelo base fica perto do acaso em tarefas novas (0,36) e o fine-tuned
chega a 0,77.

### Uma pergunta por modelo

Cada pergunta faz o Laya codificar o texto inteiro de novo. Com 2560 tokens
em CPU, uma pergunta leva ~10 s, e três estouram a memória do serverless
(ver [03-medicoes.md](03-medicoes.md)). Modelo dedicado = uma pergunta.

### Dois repositórios: motor e template

- **`laya-pipeline`** (este): o motor. Existe **uma cópia só**. Uma correção
  aqui chega a todos os modelos.
- **`laya-template`**: você cria **um repositório por modelo** a partir dele.
  Contém só o que muda por tarefa: `task.yaml`, prompts, relatórios e a
  skill do agente.

Copiar o código de treino para cada modelo faria as cópias divergirem.
Cada repositório de tarefa fixa a versão do motor por tag no
`requirements.txt`.

### Pipeline enxuto, operado pelo agente, com três aprovações humanas

Treinar é um ciclo que se repete: dados → treino → avaliação → ajuste. O
pipeline torna cada rodada repetível e rastreável. Não usamos MLflow,
Airflow nem orquestrador por enquanto.

O agente (Claude) opera as etapas. A pessoa responsável aprova em três
pontos:

1. a especificação da tarefa (`task.yaml`);
2. uma amostra dos dados rotulados (`approve data`), antes de gastar GPU;
3. o relatório de avaliação (`approve report`), antes de publicar.

### Onde treinar

| | Studio Lab | Kaggle | SageMaker Training Job |
|---|---|---|---|
| Custo | grátis | grátis | centavos a poucos US$ (spot) |
| GPU | 1× T4 | 2× T4 | a escolher |
| Sessão | poucas horas | ~9–12 h, cota semanal | sem limite prático |
| Quem opera | pessoa (login no navegador) | agente (CLI + chave API) | agente (AWS) |

- **Primeiros modelos:** Studio Lab, com checkpoint por época para retomar
  se a sessão cair. O mesmo notebook roda no Kaggle.
- **Automação futura:** Training Job, que encaixa direto no Step Functions.

### Provar antes de automatizar

Primeiro um modelo inteiro feito à mão. A página e o Step Functions vêm
depois, porque os números reais (latência, tempo de treino, custo) podem
mudar o desenho da automação.
