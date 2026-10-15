# Doc. 26 — Avaliação do amostrador C (conjunta empírica + perturbação controlada)

Estado: **EXPERIMENTAL, desligado por omissão, DEV exploratório (`observed_only`)**. Confirmatório intocado. B1/B2-C/amostragem coerente **não promovidos**. Este documento começa pelo **pré-registo** (escrito e commitado ANTES de qualquer execução); resultados e análise serão acrescentados abaixo sem alterar o pré-registo.

## 0. Pré-registo

### 0.1 Contrato do amostrador (implementação: `core/query_sampling.py`)
- Usa **só** as linhas de treino passadas ao ajuste (global e locais por nó); não vê rótulos de teste; não conhece datasets (esquema inferido dos dados: binária, grupo one-hot por exclusão mútua com soma 1, discreta inteira nominal, contínua).
- Linha-base real + perturbação: contínuas com ruído gaussiano (0,25·DP local, piso 0,1·DP global) recortado a [lo−0,1·amp, hi+0,1·amp] do treino global; grupos one-hot reamostrados **conjuntamente** (p=0,2); discretas reamostradas do vocabulário observado (p=0,2); binárias trocadas (p=0,05); linhas que sairiam iguais à base sofrem perturbação forçada de uma unidade categórica.
- Determinístico: RNG = SeedSequence([random_state, réplica, 0xC5]); a **réplica isola a aleatoriedade do amostrador** sem alterar a do algoritmo. Parâmetros **fixados antes** (sem ajuste com resultados): `noise_scale=0.25, p_resample=0.2, p_flip=0.05, margin=0.1, scale_floor=0.1`.
- Respeita transformadores existentes: no Reloaded o projetor ontológico recalcula as derivadas depois do sorteio (coerência algébrica); rejeição por restrições do caminho e teste de adoção do modelo local **herdados** do Original.
- Ativação: só dentro de `joint_sampler_context(...)` e **igual no Original e no Reloaded** (o Original canónico não é modificado; fora do contexto nada muda). O tuning estrutural (checkpoint) nunca vê o amostrador C.

### 0.2 Braços (mesmos dados, partições, MLP/oráculo congelado, orçamento estrutural em checkpoint, parâmetros, ontologia)
| Id | Braço | Fonte |
|---|---|---|
| A | TREPAN Original histórico (amostrador marginal) | `trepan_original` (r=0, existente) + `sc_A_r1,2` (seed_offset 1000·r) |
| B | Original + amostrador C | `sc_B_r0..2` |
| C | Reloaded OWL histórico (amostrador marginal) | `reloaded_owl_full` (r=0) + `cx_h_on_n_r1,2` (Doc. 23; digits só r=0,1) |
| D | Reloaded OWL + amostrador C | `sc_D_r0..2` |
| E | Reloaded sem OWL, amostrador marginal | `reloaded_core` (r=0) + `sc_E_r1,2` |
| F | Reloaded sem OWL + amostrador C | `sc_F_r0..2` |
Sementes: **treino/partição** = semente mestra {42,7,123,2024,11 (digits: 42,7,123)} (MLP e divisão); **algoritmo** = semente mestra (fixa nos braços B/D/F; nas réplicas dos braços com amostrador marginal A/C/E o RNG do amostrador e o do algoritmo são o mesmo fluxo, logo a réplica usa `seed_offset`=1000·r — limitação registada); **amostragem do amostrador C** = réplica r∈{0,1,2}. Réplicas dentro de uma unidade **não** são unidades independentes: cada unidade (dataset, semente mestra) usa a média das réplicas; unidades do mesmo dataset não são datasets independentes (reporta-se também a média macro por dataset, n=4).

### 0.3 Contrastes e hipóteses (primário ΔF = fidelidade ao oráculo; secundário ΔAcc; terciários nós, profundidade, consultas, custo)
- **EQ-hist** A vs E; **EQ-C** B vs F: esperada igualdade **exata** (fidelidade, accuracy, nós, fluxo de consultas) em todas as unidades/réplicas. Qualquer diferença ⇒ violação do contrato EQ, resultados do Reloaded sob C suspensos.
- Família inferencial (5 contrastes, Holm): **B−A** (efeito genérico do amostrador no Original), **D−B** (efeito do Reloaded enriquecido sob C: contraste ontológico principal; D−F equivale-se a D−B se EQ-C se verifica), **D−C** (efeito de C no Reloaded), **C−A** (efeito ontológico no contexto histórico), **(D−B)−(C−A)** (modificação do efeito ontológico pelo amostrador). Inferência exploratória: t emparelhado por unidade (18 unidades), EP entre unidades, + média macro por dataset; não se declara significância confirmatória.
- Critérios (declarados antes): **«C melhora o Original»** só se B−A > 0 em ΔF e ΔAcc ≥ 0 com |ΔF| > 2 EP. **«Ganho ontológico sob C»** só se D−B > 0 em ΔF **e** ΔAcc, ambos > 2 EP, no mesmo sentido. A plausibilidade melhorada **não** basta para declarar sucesso. Hierarquia (`Acc_D > Acc_B > Acc_C4.5`; `Fid_D > Fid_B`) reportada como `HIERARCHY_SATISFIED|NOT_SATISFIED|INFEASIBLE|INSUFFICIENT_EVIDENCE` (contagem por unidade; sem impor).
- Também por réplica: desvio entre réplicas dentro de célula (RNG do amostrador) vs efeito.
- Métricas de consulta efetivas (por árvore): fora de intervalo, KS por coluna, cópias exatas de linhas de treino, unicidade, razão NN consultas→treino, diversidade, violações de one-hot/vocabulário/binárias, violação algébrica das derivadas, fração perto da fronteira e cobertura de fronteira (margem do MLP), TV/entropia de rótulos. Custo: tempo de treino da árvore, geração sintética, consultas.

### 0.4 Controlo computacional (teto: estimativa inicial ≈ 1,5 CPU-h; se o piloto projetar mais, apresenta-se antes de expandir)
Verificações antes: testes sintéticos (determinismo, réplica, one-hot/vocabulário/cópias, EQ B≡F sintético) · piloto funcional iris/42 (custo por braço, retoma de checkpoint: `tuning_fits_resumed` = nº total de ajustes; fingerprint v2 inalterado; referências r=0 idênticas ao benchmark anterior) · só depois a campanha, uma unidade por processo, retoma por unidade, sem duplicar benchmarks.

### 0.5 Datasets categóricos reais (P1) — pré-registo (escrito antes de correr qualquer unidade categórica)
- **German Credit (UCI Statlog, 1000×20, 13 nominais + 7 numéricos/contagens; ficheiro local do repositório; já exposto, inelegível para confirmatório — Doc. 8; não é reserva)**. Pré-processamento genérico (`validation/benchmark/datasets_categorical.py`): numéricas inalteradas; nominais → one-hot com níveis ordenados (61 colunas); categorias desconhecidas → linha de zeros (testado); ordinais/contagens (4 atributos) declaradas nos metadados e usadas **só** pelo amostrador C (±1 passo). Sem lógica por dataset em `core/`.
- **Ontologia**: TBox de domínio congelada **antes** de resultados (`validation/benchmark/ontologies/german_credit.owl`, sem instâncias): 5 grupos temáticos definidos pelo significado público dos atributos (AccountStatus, CreditRequest, EmploymentStability, PersonalProfile, DebtAndAssets). O matching, o reasoner e o quality gate decidem se o Reloaded fica S1 (rejeitada), S2 ou S3; o resultado é reportado qualquer que seja.
- **Unidades e braços**: 5 sementes mestras {42,7,123,2024,11}; os mesmos braços A–F (A,C,E: réplica r=0 + `seed_offset` 1000·r em r=1,2; B,D,F: réplica de amostragem r=0,1,2). Todos os braços sem o amostrador C usam o `FeatureDistributionModel` marginal do Original (que trata as one-hot como colunas discretas independentes — comportamento histórico, não alterado).
- **Verificações pré-registadas** (descritivas por unidade; sem inferência entre datasets — é UM dataset): equivalência EQ-hist (A≡E) e EQ-C (B≡F); exclusividade one-hot, vocabulário, binárias e violação algébrica das derivadas nas consultas EFETIVAS (marginal vs C); cópias; fora de intervalo; diversidade; cobertura de fronteira; qualidade do matching (cobertura de features, estado S0–S3, nº de derivadas); tratamento das derivadas sobre indicadores one-hot (padronização de indicadores); fidelidade/accuracy/complexidade/custo.
- **Orçamento**: o tuning estrutural é novo (sem checkpoint) — piloto de uma unidade para medir o custo; se a projeção das 5 unidades exceder o teto restante desta fase (≈1,5 CPU-h no conjunto), apresenta-se a estimativa antes de expandir.
- Adult Census: só existe no repositório como ficheiro de instâncias OWL (`data/adult_census_full_ontology.owl`, 48 842 indivíduos); reconstrução tabular implementada (`adult_frame`), mas um subconjunto estratificado + TBox + tuning novos excedem esta fase; **não é executado** sem decisão (ver relatório).

## 1. Implementação e testes
- `core/query_sampling.py` (módulo novo; `core/trepan_original.py` e `core/trepan_reloaded_historical.py` **não** foram modificados): `JointEmpiricalDistributionModel` (mesma interface do `FeatureDistributionModel`), `infer_schema` (esquema genérico), `joint_sampler_context` (ativação explícita, **desligada por omissão**, igual no Original e no Reloaded; restaura os nomes ao sair, mesmo com exceção).
- Testes (`tests/test_joint_sampler.py`, 17): desligado por omissão e restaurado; determinismo e isolamento da réplica; difere do marginal; **Reloaded sem ontologia ≡ Original sob C** (assinatura E1–E9 e fluxo de consultas ao oráculo, binário e misto, 2 sementes); estrutura categórica preservada (one-hot, vocabulário, binárias, sem cópias, limites do treino); dados só-categóricos; restrições do caminho e partilha de esquema entre modelo global e locais; picklable; grupos one-hot recuperados com níveis raros de atributos diferentes; binárias raras independentes não agrupadas. `tests/test_categorical_datasets.py` (4) para a codificação genérica e o German.
- Agnosticismo: `scripts/check_dataset_agnosticism.py` OK; nenhum nome de dataset em `core/`.

## 2. Execução e controlo computacional (factos)
- 18 unidades DEV (iris, wine, breast_cancer ×5 sementes mestras; digits ×3), braços A–F × 3 réplicas (A,C,E r>0 por `seed_offset`; B,D,F por réplica de amostragem).
- **Checkpoints**: `tuning_fits_resumed` = nº total de ajustes em **todas** as 18 unidades (iris 120, wine/bc 150, digits 240), `tuning_execution_count`=1; fingerprint v2 inalterado (o amostrador C não está nas dependências efetivas do tuning: só é ativado dentro do ajuste final do braço).
- **Integridade**: 360/360 comparações dos braços de referência r=0 (Original, Reloaded-core, Reloaded OWL, C4.5: fidelidade, accuracy, nós, profundidade, consultas) idênticas ao benchmark anterior; réplicas de C vindas das células `cx_h_on_n` do Doc. 23 verificadas idênticas a `sc_C_r1,2` em iris/42.
- **Equivalência**: **EQ-hist A≡E: 54/54 idênticas; EQ-C B≡F: 54/54 idênticas** (fidelidade, accuracy, nós, profundidade, nº de consultas), em todas as unidades e réplicas.
- **Custo**: soma dos tempos das unidades DEV ≈ 1,66 h (3 unidades em paralelo) + piloto German 0,32 h ⇒ **≈ 2,0 CPU-h**, ~30 % acima da estimativa inicial (1,5 CPU-h); o excesso vem dos ajustes do Reloaded OWL (~50 s por ajuste, ×3 réplicas ×D) e de digits (21–26 min por unidade). As restantes 4 unidades do German **não** foram corridas (estimativa: ≈ 1,35 CPU-h; ver §6).

## 3. Resultados DEV numérico (exploratórios; 18 unidades; EP entre unidades; Holm sobre a família de 5 contrastes)
### 3.1 Médias descritivas (média das unidades; cada unidade = média das 3 réplicas)
| Braço | Fidelidade | Accuracy | Nós | Profundidade | Consultas | Tempo de ajuste (s) |
|---|---|---|---|---|---|---|
| A Original (marginal) | 0,9160 | 0,9118 | 67,6 | 8,6 | 42 824 | 5,6 |
| B Original + C | **0,9327** | **0,9349** | 69,6 | 8,7 | 42 449 | 5,6 |
| C Reloaded OWL (marginal) | 0,9132 | 0,9105 | 68,9 | 9,0 | 91 499 | 52,9 |
| D Reloaded OWL + C | 0,9229 | 0,9287 | 68,8 | 8,8 | 90 893 | 49,1 |
| E Reloaded sem OWL (marginal) = A | 0,9160 | 0,9118 | 67,6 | 8,6 | 42 824 | 5,6 |
| F Reloaded sem OWL + C = B | 0,9327 | 0,9349 | 69,6 | 8,7 | 42 449 | 5,8 |
| C4.5 canónico | 0,9185 | 0,9239 | 37,3 | 6,1 | — | — |
### 3.2 Contrastes (Δ = primeiro − segundo)
| Contraste | ΔFidelidade (EP; melhor/pior; Holm) | ΔAccuracy (EP; melhor/pior; Holm) |
|---|---|---|
| **B−A** efeito de C no Original | **+0,0167** (0,0062; 14/3; p_Holm 0,080) | **+0,0231** (0,0060; 14/2; p_Holm 0,006) |
| **D−B** efeito do Reloaded enriquecido sob C | −0,0098 (0,0058; 3/10; 0,43) | −0,0063 (0,0053; 4/9; 0,77) |
| **D−C** efeito de C no Reloaded | +0,0097 (0,0062; 12/5; 0,43) | **+0,0181** (0,0063; 13/3; p_Holm 0,040) |
| **C−A** efeito ontológico, amostrador marginal | −0,0028 (0,0039; 7/7; 0,68) | −0,0013 (0,0047; 9/6; 1,0) |
| **(D−B)−(C−A)** interação | −0,0071 (0,0072; 8/8; 0,68) | −0,0050 (0,0074; 7/10; 1,0) |
| E−A, F−B (EQ) | 0 | 0 |
Por dataset (média das unidades): B−A fidelidade/accuracy — digits +0,043/+0,049, wine +0,022/+0,030, iris +0,009/+0,019, breast_cancer +0,004/+0,005 (4/4 positivos); D−B — todos negativos em fidelidade (iris −0,012, wine −0,018, bc −0,003, digits −0,004). B−A estável entre réplicas de amostragem (fidelidade +0,014, +0,014, +0,022; accuracy +0,026, +0,020, +0,024), ao contrário do D−C do Doc. 23. Desvio-padrão entre réplicas (dentro da unidade): A 0,018 · B 0,012 · C 0,020 · D 0,017.
Complexidade e custo: nós B−A +2,0 (EP 2,7), profundidade B−A +0,1 (n.s.); consultas ao oráculo: B−A −375 (n.s.), **D−B +48 444** (18/0) e C−A +48 675 (18/0) — o Reloaded OWL gasta ≈ 2,1× as consultas do Original em ambos os amostradores (mecanismo ativo existente, não do amostrador C); tempo de ajuste D−B +43 s (18/0); o amostrador C não acrescenta custo mensurável ao Original (B−A +0,07 s).
### 3.3 Critérios pré-registados (§0.3)
- «C melhora o Original» (B−A: ΔF > 2 EP e ΔAcc ≥ 0): **cumprido** (ΔF = 2,7 EP; ΔAcc = 3,9 EP; Holm: accuracy 0,006, fidelidade 0,080 — exploratório).
- «Ganho ontológico sob C» (D−B: ΔF e ΔAcc > 2 EP, mesmo sentido): **NÃO cumprido** (D−B negativo: −1,7 EP e −1,2 EP).
- Hierarquia por unidade (médias das réplicas): Acc_D > Acc_B em 4/18; Acc_B > Acc_C4.5 em 9/18; Fid_D > Fid_B em 3/18; as três em 1/18 ⇒ **`HIERARCHY_NOT_SATISFIED`** (não imposta; reportada).

## 4. Plausibilidade das consultas EFETIVAS das árvores (médias das unidades; Doc. 24 §6 medido agora dentro do TREPAN)
| Medida | A marginal | B Original+C | C Reloaded marginal | D Reloaded+C |
|---|---|---|---|---|
| fora do intervalo do treino | 0,45 | 0,12 | 0,49 | 0,10 |
| razão NN consultas→treino (1 = como o treino) | 1,76 | 0,63 | 1,65 | 0,61 |
| diversidade (razão NN entre consultas) | 1,37 | 0,54 | 1,31 | 0,50 |
| violação de vocabulário (discretas inteiras) | 0,16 | 0 | 0,28 | 0 |
| violação algébrica das derivadas | — | — | 0 | 0 |
| cobertura de pontos de fronteira reais | 0,25 | 0,63 | 0,14 | 0,68 |
| fração de consultas junto à fronteira (margem<0,2) | 0,043 | 0,039 | 0,068 | 0,053 |
| KS médio (originais vs treino) | 0,185 | 0,214 | 0,189 | 0,235 |
| cópias exatas de linhas de treino | 0 | ≈2e-5 | 0 | ≈1e-4 |
Leitura: C aproxima as consultas do suporte real (menos extrapolação, vocabulário respeitado, melhor cobertura da fronteira) à custa de **menor diversidade** e de marginais mais afastadas das do treino (KS maior): trade-off reportado, não otimizado. Plausibilidade melhor **não** implicou, por si, melhor D (D−B < 0).

## 5. P1 — German Credit (DEV categórico real; piloto de 1 unidade, semente 42; descritivo)
- Matching/ontologia: TBox de 5 grupos; `ontology_effectively_used`; 61 features mapeadas (cobertura 1,0); 5 derivadas (médias padronizadas de indicadores one-hot/numéricos de atributos distintos: algebricamente coerentes, **semanticamente fracas**).
- Equivalência: A≡E e B≡F idênticas nas 3 réplicas.
- **Estrutura categórica (consultas efetivas)**: o amostrador marginal histórico (Original e Reloaded) viola a exclusividade one-hot em **99,9 %/99,6 %** das consultas (ativa mais de uma categoria, ou nenhuma, num atributo) e gera valores fora do vocabulário (3,5 %); com C a violação one-hot e de vocabulário é **0** (B e D). Fora de intervalo 0,175→0,068 (B); cobertura da fronteira 0,19→0,81 (B) e 0,23→0,70 (D).
| Braço (média das 3 réplicas; C só r=0) | Fidelidade | Accuracy | Nós | Consultas | Tempo (s) |
|---|---|---|---|---|---|
| A / E | 0,861 | 0,724 | 42,3 | 28 298 | 5,0 |
| B / F | 0,843 | 0,727 | 58,3 | 28 461 | 5,1 |
| C (r=0) | 0,856 | 0,740 | 43,0 | 65 756 | 75,6 |
| D | 0,860 | 0,720 | 60,3 | 65 836 | 72,9 |
| C4.5 / MLP | fid. 0,748 / — | acc. 0,680 / 0,772 | 149 | — | — |
Uma só unidade; desvio entre réplicas ≈ 0,03 em fidelidade (≥ qualquer diferença entre braços): **sem evidência de efeito em fidelidade/accuracy**; o resultado robusto é estrutural (exclusividade/vocabulário/cobertura). Limitações: níveis de uma variável nominal codificados com todos os níveis do conjunto completo (vocabulário público; categorias desconhecidas tratadas por zeros — testado); ordinais declaradas por metadados (4); ontologia manual; 1 dataset.

## 6. Interpretação, riscos e pendentes
1. O amostrador C melhora o **Original** (B−A) de forma consistente nos 4 DEV numéricos e nas 3 réplicas — é um efeito **genérico, não ontológico** (contrato EQ: aplicado igualmente a Original e Reloaded sem OWL, B≡F). Candidato legítimo a variante experimental do Original (decisão do utilizador), **não** promovido.
2. **A ontologia verdadeira não demonstra ganho sob C**: D−B < 0 em fidelidade e accuracy (não significativo), D−C só melhora em accuracy porque C melhora tudo; a interação é ≈ 0. A hierarquia `Acc_Reloaded_OWL > Acc_Original > Acc_C4.5` não se verifica (9/18 e 4/18).
3. Riscos/limitações: tuning estrutural feito com o amostrador marginal (C herda a configuração); parâmetros de C fixados a priori (sem ajuste; nenhuma análise de sensibilidade feita para não selecionar com resultados); B−A é dominado em magnitude por digits e wine; 4 datasets, 3 sementes em digits; unidades do mesmo dataset não são independentes (macro por dataset reportado: todos positivos); o excesso de consultas do Reloaded OWL (2,1×) confunde comparações de custo; no German só 1 unidade.
4. Pendentes (precisam de decisão): correr as 4 unidades restantes do German (≈ 1,35 CPU-h, acima da estimativa; **não executado**); Adult Census (reconstrução implementada, requer subamostra + TBox; não executado); promover C como variante do Original; análise de sensibilidade de C (exploratória, rotulada).

## Alterações de protocolo
**A1 (registada ANTES de qualquer resultado analisado; aplicada a todos os braços/datasets).** Depois do piloto funcional (iris/42: só verificação de integridade, equivalência A≡E / B≡F, custo e identidade de C com as células `cx_h_on_n`; os valores de fidelidade/accuracy desse piloto foram vistos) e **antes** de analisar qualquer campanha, um teste estrutural com o German Credit mostrou que o contrato do §0.1 não estava cumprido em atributos categóricos reais, por duas razões de desenho (não de resultados):
1. **Deteção de grupos one-hot**: a versão inicial usava componentes conexas do grafo de exclusão mútua; níveis raros de atributos diferentes que nunca coincidem por acaso fundiam componentes e o grupo era rejeitado (German: 1 grupo reconhecido em 13; as restantes binárias seriam trocadas independentemente, violando a exclusividade). Substituída por **partição exata gulosa** (suportes disjuntos cuja soma cobre todas as linhas); recupera 13/13 grupos no German e continua a rejeitar binárias raras independentes (testes novos).
2. **Reamostragem de discretas** pelo vocabulário **uniforme** destruía a frequência empírica (ex.: pixels inteiros). Passa a copiar o valor de **outra linha-base do nó** (frequência empírica local; vocabulário preservado).
Os restantes parâmetros (ruído, p_resample, p_flip, margem, piso) **não** mudaram. O piloto e a campanha parcial (apenas iris, que não tem binárias nem discretas, logo nenhum braço de iris muda de comportamento) foram descartados e a campanha foi relançada de raiz com o código alterado. Estes resultados DEV numéricos não influenciaram a alteração.
