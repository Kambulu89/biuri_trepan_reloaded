# Estudo experimental: quando é que o enriquecimento OWL pode melhorar o BIURI/TREPAN Reloaded

**Estado: INTERMÉDIO.** Nada aqui altera o BIURI (core/, gui/), os gates ou os baselines. Execuções em Linux headless (offscreen), não Windows.

## Conteúdo
- `ontologies_real/` — Breast Cancer: v3 (entrega), v4 (EXPERIMENTAL), controlo negativo (rótulos permutados). Conhecimento de domínio citado de memória, **não verificado** (revisão humana pendente); estatuto por axioma: ver comentários `rdfs:comment`.
- `gen_synthetic.py`, `synthetic/` — datasets sintéticos S1 (grupos numéricos latentes) e S2 (taxonomia categórica) + ontologias `correct` / `perm` / `rand`. **Prova de capacidade do mecanismo, não evidência sobre dados reais.**
- `run_batch.sh`, `run_biuri_headless.py`, `parse_runs.py` — pipeline real da GUI em modo offscreen e extração dos logs.
- `sensitivity_offline.py` — análise de sensibilidade NÃO oficial (sklearn à parte), só diagnóstico.
- `logs_bc/`, `logs_synthetic/` — logs brutos (imutáveis). `logs_synthetic/TRUNCADOS_cap3000s/` = corridas cortadas pelo limite de espera do *meu* script (não do BIURI); **não usar como resultado**.

## Protocolo e limitações (honestas)
- O gerador sintético foi calibrado em 3 iterações usando **apenas** CV interna do treino numa análise offline (`sensitivity_offline.py`); nenhuma métrica de teste do BIURI foi usada.
- Execuções em paralelo reduzem os *trials* Optuna (limite de tempo): comparações entre braços correm sob orçamentos distintos → tratar como `observed_only`.
- Um split, poucas seeds: sem inferência estatística.
