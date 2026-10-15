# Documento 2 — Protocolo experimental (PROPOSTA, revista pela correção arquitetural)

> **Revisão:** o Reloaded é o Original + extensão ontológica (contrato de equivalência EQ, `07_CONTRATO_EQUIVALENCIA.md`). Neste protocolo `R_OWL` = Reloaded com ontologia efetiva; `R∅` = Reloaded sem ontologia (controlo EQ, tem de ≡ Original); o ganho é sempre medido contra o **Original equivalente** nas mesmas condições. Os seis grupos de comparação (A C4.5, B Original, C R∅, D R_OWL, E controlo permutado/aleatório, F engenharia convencional) estão em `04_MATRIZ_ABLACAO.md` §1.

Nada aqui foi executado. O protocolo só é válido depois de **aprovado e congelado** num `BENCHMARK_MANIFEST_v2.json` (hash do código, datasets, ontologias,
seeds, grelhas, regras de decisão) **antes** de qualquer execução confirmatória. TREPAN Original e C4.5 mantêm-se congelados (hashes do manifest v1).

## 1. Estatuto das evidências (regra de ouro)
| Conjunto | Datasets | Para que serve | Proibido |
|---|---|---|---|
| **DEV** | iris, wine, breast_cancer, digits, synthetic_* (+ ontologias de breast_cancer, sonar, hepatitis, balance-scale já analisadas) | depurar, testes unitários/metamórficos, medir custo, verificar que o código faz o que a especificação diz | declarar superioridade; escolher λ/limiares/variante «pelo resultado no teste» |
| **SELECT** (interno a cada unidade) | treino de cada unidade, CV interna com MLP reajustado por dobra | escolher a configuração do Reloaded (λ, variante) | qualquer uso do teste externo |
| **CONFIRM** | ≥ 12 datasets **novos** (nunca usados em resultados) | única fonte de afirmações de superioridade | ajuste posterior a ver resultados; trocar datasets |

Os 4 datasets sklearn já foram observados repetidamente; contam como DEV mesmo que se use outra seed.

## 2. Datasets CONFIRM (critérios genéricos; a lista concreta é aprovada por ti antes de correr)
* Classificação tabular, 150 ≤ n ≤ 20 000, 4 ≤ p ≤ 200, 2–10 classes; mistura de: só numéricas / mistas / com valores em falta; binário e multiclasse;
  com e sem estrutura de «famílias de medições»; desequilibrados e equilibrados; pelo menos 3 domínios distintos.
* Origem pública e versionada (UCI/OpenML), hash do ficheiro no manifest. Sem pré-visualização de resultados para escolher.
* Acrescentar dataset = **dados + `DatasetSpec` + ontologia** (nunca lógica).
* **Ontologias CONFIRM:** escritas por procedimento documentado a partir de descrições de atributos (metadados), **sem** consultar o desempenho de qualquer modelo,
  congeladas por `sha256` no manifest; ontologia «independente» quando o autor não é quem ajustou o algoritmo (ideal: o próprio utilizador/colaborador).
  Passam o quality gate existente; as rejeitadas ficam registadas (resultado, não erro).
* **Cenário nomes opacos:** subconjunto de ≥ 4 datasets repetido com colunas renomeadas `f1…fn`. Sem mapeamento explícito a ontologia é rejeitada pelo gate
  (esperado → mede o fallback). Com mapeamento explícito (decisão D-6) mede-se o valor do conhecimento puro.

## 3. Unidades, partições e seeds
* Esquema CONFIRM: `repeated_cv` 5-fold × 2 repetições (10 unidades por dataset), estratificado, `base_seed` e lista de repetições **fixados no manifest**;
  as mesmas partições para todos os braços (pareamento). Com 12 datasets = **120 unidades por braço**.
* Papéis: `TRAIN` (treino externo) ⊃ `VALIDATION` (dobras internas) ; `TEST` externo lido **uma vez** por unidade (`EvaluationProtocolGuard`).
* Seeds nunca escolhidas depois de ver resultados; sem repetição de unidades «azaradas».

## 4. Oráculos e pistas
* **Pista T (enriquecimento da árvore) — primária:** oráculo = `MLP Original` congelado (`freeze_oracle`), **o mesmo** para Original, R∅, R_OWL (todas as variantes), controlos e C4.5-fidelidade. R_OWL no espaço enriquecido
  consulta o mesmo oráculo via `OriginalOracleProjection`. Toda a fidelidade «A–F» pertence a esta pista.
* **Pista M (enriquecimento do MLP) — separada:** oráculo = `MLP Ontológico` aceite pelo gate interno; compara-se Original(repr. OWL) vs R_OWL **consultando o mesmo oráculo MLP_onto**. Fidelidades das pistas T e M nunca se misturam.

## 5. Fluxo por unidade (sem contaminar o teste)
```
Treino externo (n_tr) ──► MLP original (oráculo externo O_ext)  [para o teste final]
        │
        ├─► Estrutura comum (tuning ontology-blind, existente, só Original + O_ext)      → budget comum C
        │
        └─► SELEÇÃO do R_OWL (nova; **só existe se Ω efetiva** — sem ontologia ou ontologia rejeitada o Reloaded ≡ Original e não há seleção):  para cada dobra interna k=1..K do treino:
                 MLP_k reajustado só em inner-train_k  →  oráculo O_k congelado
                 Original_k, C4.5_k, Reloaded_k(config c ∈ GRELHA) treinados com O_k / rótulos de inner-train_k, orçamento C
                 medir em inner-val_k (rótulos reais e O_k): A_R,A_O,A_C,F_R,F_O, nós, queries, estabilidade
             agregar OOF → escolher c* pela regra lexicográfica do Documento 5 (congelada)
        │
        ▼
 Refit Reloaded(c*) com O_ext no treino externo completo → avaliação ÚNICA no teste externo
```
`K = 3` dobras internas (como `BenchmarkConfig.inner_cv_splits`). A grelha é **pré-declarada e pequena** (Documento 3 §4); λ nunca é escolhido no teste.

## 6. Hipóteses e endpoints
| Id | Hipótese (unilateral, pré-registada) | Endpoint primário |
|---|---|---|
| **H0** | **EQ:** `R∅ ≡ O` (exata, não estatística) | igualdade de E1–E9 em **todas** as unidades; uma violação = protocolo suspenso para essa variante |
| H1 | `A_R_OWL > A_O` | Δacc_real = acc(R_OWL*) − acc(O), pareado por unidade |
| H2 | `A_O > A_C` | Δacc_real = acc(O) − acc(C4.5) (propriedade dos baselines; **não** é alvo de otimização) |
| H3 | `F_R_OWL > F_O` | Δfid = fid(R_OWL*) − fid(O) ao mesmo oráculo |
| H4 | ganho atribuível à ontologia | Δ_conhecimento = R_OWL − R_OWL~perm/~rand; Δ_convencional = O(repr. OWL) − O(repr. conv.); decomposição Δ_total = Δ_repr + Δ_guia (Doc. 7 §7) |
Secundários: macro-F1, balanced accuracy, nº de nós/folhas/profundidade, comprimento médio das regras, nº de splits m-of-n, nº de queries (igual por
construção de orçamento), tempo, estabilidade entre repetições (DP entre unidades do mesmo dataset), taxa de aceitação/rejeição de splits semânticos.
Não-inferioridade (restrições): nós ≤ 1,25× Original; fidelidade ≥ Original − δ_F (δ_F pré-fixado = 0,01).

## 7. Estatística
* Dentro de cada dataset: t reamostrado corrigido de Nadeau–Bengio (10 unidades correlacionadas) + IC bootstrap pareado.
* Entre datasets (unidade = dataset, diferenças médias): Wilcoxon de ordens com sinais (unilateral) **e** IC bootstrap sobre datasets; correção de Holm sobre a
  família {H1, H2, H3}; H4 corrigida separadamente (família de controlos).
* Equivalência/não-inferioridade: TOST com margem ±1 pp (pré-fixada).
* Tamanhos de efeito (dz de Cohen, Cliff's δ); todos os resultados **sem arredondamento** nas comparações; arredondar só na apresentação.
* **Potência** (aprox., t pareado entre datasets, 80 %, α=0,05 bilateral; MDE = 2,8·σ_d/√N): 

| σ_d (DP das diferenças por dataset) | N=12 | N=16 | N=20 |
|---|---|---|---|
| 1,5 pp | 1,2 pp | 1,1 pp | 0,9 pp |
| 2,0 pp | 1,6 pp | 1,4 pp | 1,3 pp |
| 3,0 pp | 2,4 pp | 2,1 pp | 1,9 pp |
Pressupostos de σ_d vêm de resultados exploratórios (DP por unidade ≈ 2,5–2,9 pp) e **não são garantidos**; com N=12 só se detetam efeitos ≳ 1,6 pp. Efeitos de ~0,5 pp
(ordem observada no MLP) ficam **fora de alcance**: o protocolo declara-o em vez de o esconder.
* Resultados por dataset reportados todos (incluindo negativos); nenhum dataset excluído depois de correr.

## 8. Variantes e controlos (ver Documento 4)
Todas as variantes de R_OWL usam **o mesmo** orçamento de candidatos/queries que o Original; os controlos igualam número de candidatos m-of-n avaliados e oportunidades de seleção. Nenhuma variante
usa rótulos reais na construção da árvore (a linha «híbrida/Modo B» saiu do âmbito: Doc. 7 §8).

## 9. Reprodutibilidade e integridade
* `BENCHMARK_MANIFEST_v2.json` + `manifest_sha256`, `frozen_code_sha256` (inclui os novos módulos), versões de bibliotecas, ambiente.
* Resultados brutos imutáveis (`validation/benchmark/raw_store.py`), tabelas reconstruídas só dos brutos; `equivalence_report.py` para qualquer otimização de desempenho.
* Um mundo OWL isolado por unidade (PR #9). Execução por unidade retomável (checkpoint).
* Falhas de braço registadas (`skipped`), nunca omitidas; contagem de unidades válidas por braço reportada.

## 10. Regras de paragem e de honestidade
* Se H1 não for sustentada, publica-se **«não sustentada»** com IC e potência; não se alteram variantes depois de ver o CONFIRM.
* Nova hipótese/variante ⇒ novo manifest, **novos** datasets CONFIRM.
* `HIERARCHY_*` é um relatório (Documento 5), nunca um critério de exclusão de unidades.


## 12. Revisão 2 — requisito científico obrigatório (hierarquia conjunta)
Endpoint de aceitação adicional **H-conj** (unilateral, pré-registado): `A_R_OWL > A_O`, `A_O > A_C`, `A_R_OWL > A_C`, `F_R_OWL > F_O`, `F_R_OWL > F_C`, **em conjunto**, nas mesmas instâncias e contra o mesmo MLP (ver `28_REQUISITO_HIERARQUIA_CONJUNTA.md`; `core/hierarchy_contract.py`). Compromissos (`TRADEOFF_*`) e ganhos unilaterais não são sucesso. Accuracy (rótulos reais) é a métrica principal; `precision_macro_real_labels` e `macro_f1_real_labels` são complementares e distintas. Seleção só em treino/validação interna (elegibilidade conjunta ΔA_val ≥ 0 ∧ ΔF_val ≥ 0 face ao Original equivalente); teste externo lido uma vez; baselines congelados; unidades `INFEASIBLE` (Original ≤ C4.5) reportam-se sem tratamento. Os endpoints H1–H3 acima mantêm-se como análises individuais, mas a **declaração de sucesso** de uma intervenção exige H-conj.
