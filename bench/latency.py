"""Teste de viabilidade: checkpoint multilingual do Laya com max_len 512/1024/2560
em CPU limitada (simula o SageMaker Serverless 3072 MB)."""
import json, os, resource, statistics, time

os.environ.setdefault("USE_TF", "0")
t0 = time.time()
import torch
import laya
from transformers.initialization import no_init_weights

# os.cpu_count() enxerga os núcleos da máquina, não o limite do container/serverless;
# threads a mais que CPUs disponíveis derrubam o desempenho. Passe LAYA_THREADS.
N_THREADS = int(os.environ.get("LAYA_THREADS", os.cpu_count() or 1))
torch.set_num_threads(N_THREADS)
t_import = time.time() - t0

t0 = time.time()
with no_init_weights():
    agent = laya.load("convaiinnovations/laya", subfolder="multilingual", device="cpu")
t_load = time.time() - t0

PARAGRAFO = (
    "O Tribunal de Contas recebeu uma notificação do setor de tecnologia informando que, "
    "durante a madrugada, foram registradas diversas tentativas de acesso ao servidor de "
    "arquivos a partir de endereços externos. O analista de plantão verificou os registros "
    "de autenticação e identificou que uma conta de serviço teve a senha alterada sem "
    "abertura de chamado. Em seguida, houve cópia de planilhas da área de licitações para "
    "um diretório temporário e posterior transferência para um domínio desconhecido. "
)
tok = agent.tok
base_ids = tok(PARAGRAFO, add_special_tokens=False)["input_ids"]
texto_longo = PARAGRAFO * (3000 // len(base_ids) + 2)

QUESTOES = {
    "severidade": {"type": "score", "instructions": "Qual a severidade do incidente descrito?",
                   "criteria": ["baixa", "média", "alta", "crítica"]},
    "tipo": {"type": "choice", "instructions": "Qual o tipo principal do incidente?",
             "criteria": {"vazamento": "exfiltração ou vazamento de dados",
                          "malware": "infecção por software malicioso",
                          "phishing": "engenharia social por e-mail",
                          "acesso": "acesso indevido a contas ou sistemas"}},
    "escalar": {"type": "noul", "instructions": "O incidente deve ser escalado para a equipe de resposta?"},
}

resultados = {"cpus_visiveis": os.cpu_count(), "threads": N_THREADS, "import_s": round(t_import, 1), "load_s": round(t_load, 1), "runs": []}
# LAYA_RUNS="512:1,2560:3" escolhe os casos (padrão: todos).
CASOS = [tuple(map(int, c.split(":"))) for c in
         os.environ.get("LAYA_RUNS", "512:1,512:3,1024:1,1024:3,2560:1,2560:3").split(",")]
for max_len, nq in CASOS:
    agent.cfg["max_len"] = max_len
    if True:
        qs = dict(list(QUESTOES.items())[:nq])
        agent.system_one(texto_longo, qs)  # aquecimento
        tempos, n_tok = [], 0
        for _ in range(3):
            t0 = time.time()
            out = agent.system_one(texto_longo, qs)
            tempos.append(time.time() - t0)
            n_tok = out["usage"]["input_tokens"]
        r = {"max_len": max_len, "questoes": nq, "tokens": n_tok,
             "mediana_s": round(statistics.median(tempos), 2), "max_s": round(max(tempos), 2),
             "pico_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)}
        resultados["runs"].append(r)
        print(json.dumps(r), flush=True)

resultados["pico_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
print("RESULTADO " + json.dumps(resultados), flush=True)
