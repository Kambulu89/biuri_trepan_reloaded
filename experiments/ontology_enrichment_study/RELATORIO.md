# Relatório — quando é que o enriquecimento OWL pode melhorar o BIURI/TREPAN Reloaded

Estatuto: **experimental, observed_only**. Execução em Linux headless (offscreen), **não Windows**. Código oficial do BIURI, gates, limiares, MLP e baselines **não foram alterados**. A única exceção é uma cópia temporária (patch de 1 linha, ver §3) usada numa análise de sensibilidade explicitamente **não oficial**.

## 1. Objetivo 1 — ontologias de datasets reais
Só o Breast Cancer Wisconsin estava disponível no ambiente (os ARFFs de heart/hepatitis/vehicle/ionosphere/mushroom/pima/sonar foram perdidos num reset; não os usei).
- `ontologies_real/breast_cancer_v3_ENRIQUECIDA.owl` (entrega): 30/30 colunas mapeadas, HermiT consistente, `VALID_DOMAIN_ONTOLOGY`; gate de utilidade `ACCEPT_PARTIAL_FEATURES` (OOF acc 96,3→97,3 %, ganho líquido +0,0115).
- `…v4_EXPERIMENTAL_extra_domains.owl`: mais domínios por propriedade (estatística, medida) → 72 candidatas; **mesma** única feature selecionada e mesmo ganho → mais estrutura não acrescentou informação.
- `…v3_CONTROLO_negativo_rotulos_permutados.owl` → gate de utilidade `REJECT_GAIN_BELOW_COMPLEXITY_COST` (controlo negativo funcionou ao nível do gate de utilidade).
- Estatuto do conhecimento: (i) 30 colunas = 10 medidas × {mean, error, worst}: **comprovado** pelo dicionário público do dataset (citado de memória, não verificado aqui); (ii) agrupamento em tamanho / indentação do contorno / regularidade / textura: **inferido** da descrição do dataset; (iii) limites ≥ 0: **comprovado** (grandezas geométricas não negativas). Nada é clínico-preditivo; revisão por especialista pendente. Rótulos e resultados de teste não foram usados para escolher axiomas.

### Achado importante (Breast Cancer)
O ganho aparente do Reloaded (94,15 % vs Original 90,06 %; C4.5 93,57 %; n=171) **não depende da ontologia**: com a ontologia permutada (gate de utilidade rejeitado) o Reloaded dá **exatamente** 94,15 %. Árvore Reloaded também maior (61 nós vs 39). Original < C4.5 → `HIERARCHY_INFEASIBLE`. Professor: `ZERO_ONTOLOGY_MASS_IN_SELECTED_HYBRID` (pesos 1,0/0/0): o MLP Original já tem 99 % e a seleção, só com treino, não deu peso à ontologia.

## 2. Objetivo 2 — prova de capacidade sintética (S1, seed 1)
S1: 250 amostras, 120 sensores = 20 subsistemas × 6; cada sensor = latente do subsistema + ruído; rótulo depende só de 2 subsistemas latentes. Ontologia correta escrita a partir do **desenho** do gerador (subsistema→classe de domínio), nunca a partir dos rótulos. Controlos: `perm` (colunas trocadas entre grupos, mesmos tamanhos), `rand` (partição aleatória de tamanhos arbitrários), `none`. O gerador foi calibrado em 3 iterações apenas com CV interna do treino (`sensitivity_offline.py`), sem métricas de teste do BIURI.

### Resultados (BIURI real; **com patch de sensibilidade, não oficial**, 1 seed, n_teste=75, validação interna do professor n=44)
| Braço | Professor | OOF acc base→OWL | Ganho líq. utilidade | Professor val. (n=44): acc MLP → híbrido | Árvore Original (teste) | Árvore Reloaded (teste) | C4.5 |
|---|---|---|---|---|---|---|---|
| ontologia correta | **ACEITE** (peso ont. do híbrido) | 77,9→80,9 | **+0,031** | 81,8→**93,2** (BA +10,6) | 72,0 % | **73,3 %** (53 nós) | 68,0 % |
| permutada | ACEITE | 77,9→79,4 | +0,019 | 81,8→86,4 (BA +4,6) | 72,0 % | 60,0 % | 68,0 % |
| aleatória | ACEITE | 77,9→78,6 | +0,011 | 81,8→86,4 (BA +4,6) | 72,0 % | 68,0 % | 68,0 % |
| sem ontologia | n/a | n/a | n/a | n/a | 54,7 %* | 54,7 %* | 68,0 % |
\* corrida `none` com orçamento Optuna diferente (4 workers em paralelo) → **não comparável** (MLP 86,7 % vs 90,7 %). Honestidade: só os 3 braços com ontologia partilham orçamento.

### Leitura
1. O mecanismo **representa e usa** a estrutura latente: com a ontologia correta o gate selecionou agregados de Flow, Density e Thermal; Flow e Thermal são exatamente os 2 subsistemas relevantes (Density é irrelevante). Seleção só com treino/OOF.
2. O ganho OOF é monótono: correta (+0,031) > permutada (+0,019) > aleatória (+0,011). **Mas** os controlos também passam o gate do professor: a média de colunas (mesmo mal agrupadas) reduz ruído, por isso o gate **não distingue** conhecimento correto de agregação genérica. Atribuição à ontologia só se sustentaria com ganho conjunto sobre controlos e vários seeds — **não demonstrado** (1 seed; diferença ≈ 2,5 amostras em 44).
3. Árvores (o que o TREPAN explica): Reloaded 73,3 % vs Original 72,0 % = **1 instância em 75**. Fidelidade do Reloaded é ao oráculo híbrido (correta 72,0 %, permutada 64,0 %, aleatória 70,7 %), do Original ao MLP Original (78,7 %): **oráculos diferentes → não comparável** (contrato exige mesmo oráculo). Hierarquia conjunta **não** satisfeita.
4. Conclusão sobre a hipótese do utilizador: **há evidência observada** de que com informação semântica real (grupos corretos) o professor pode ser aceite e o MLP melhorado (+10,6 pp BA numa validação de 44), **mas** não há evidência de que isso se traduza em árvores mais fiéis/precisas, e os controlos não ficam rejeitados.

## 3. Objetivo 3/4 — verificação dos professores e diagnóstico
**Oficial (sem patch): S1 com ontologia correta FALHA no treino** — `OntologySchemaError: O transform não reproduziu exatamente o schema ajustado no treino` (`logs_synthetic/OFICIAL_falha_schema/`).
Causa verificada no código (causa raiz = mecanismo de seleção/aplicação, não ontologia nem gates): `gui/biuri_app_complete.py:4265-4269` filtra `feature_specs_` mantendo a ordem original dos specs, mas fixa `output_features_` na ordem de `feature_names_aug` (ordem de seleção por ganho). Com ≥2 features semânticas selecionadas em ordem diferente da dos specs (aqui Flow, Density, Thermal vs Thermal…Density), `transform` produz colunas noutra ordem e a validação de esquema levanta o erro. Com 1 feature selecionada (Breast Cancer) não ocorre. É defeito geral (agnóstico ao dataset); **não foi corrigido no repositório** (restrição desta tarefa); `patch_sensibilidade_schema_order.diff` mostra a correção testada (ordenar os specs pela ordem de `feature_names_aug`), usada **só** na análise de sensibilidade.

Diagnóstico (Objetivo 4):
- Qualidade/relevância do conhecimento: relevante em S1 (seleciona subsistemas certos); irrelevante em Breast Cancer (colunas já completas, MLP a 99 %).
- Interpretação dos axiomas: OK (30/30 e 120/120 mapeados; ver `semantic_names`).
- Geração de atributos derivados: gera agregados corretos; **em Breast Cancer os derivados não acrescentam informação**.
- Seleção dos professores: **defeito de ordenação** (acima); e o gate aceita controlos (não discrimina semântica de média).
- Limiares: não alterados; sem evidência de que sejam a causa.
- Incapacidade das representações de acrescentar informação: sim em Breast Cancer; em S1 acrescentam, em parte por redução de ruído genérica.

## 4. Não feito / limitações
- S2 (taxonomia categórica) e seeds adicionais **não concluídos**: a máquina foi reciclada várias vezes e matou as corridas longas (~35–50 min cada); logs truncados em `logs_synthetic/TRUNCADOS_cap3000s/`.
- Só Breast Cancer entre os reais; ARFFs dos outros perdidos.
- 1 seed, 1 split, sem estatística; resultados `observed_only`.
- Orçamento Optuna limitado por tempo: comparações entre sessões não são idênticas.
- Conhecimento de domínio de Breast Cancer citado de memória.

## 5. Reprodução
`gen_synthetic.py <dir> <seed>` → `run_batch.sh`/`run_sensibilidade_patch.sh` (BIURI_ROOT aponta para cópia com o patch) → `parse_runs.py <logs>`.
Windows: carregar `s1_seed1.arff` + `s1_seed1_correct.owl` em 📁 Carregar dados e treinar (sem o patch falhará com o erro acima enquanto o defeito existir).
