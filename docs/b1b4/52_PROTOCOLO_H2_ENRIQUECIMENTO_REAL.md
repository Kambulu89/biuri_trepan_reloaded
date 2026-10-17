# Doc. 52 — Protocolo H2: benefício de semântica OWL expressiva (enriquecimento ontológico real)

Estado: **protocolo em rascunho — `frozen = false`, NÃO executado, NÃO autorizado a executar.** Só `validation/`. `B34 = DESIGN_ONLY / NOT_EXECUTED`; `M0_M6_CORE_INTEGRATION = NOT_AUTHORIZED`. Contrato de mecanismos: Doc. 51; seleção de ontologia: Doc. 50; qualidade: Doc. 45; revisão da B3.4: Docs. 48–49.
Objetivo central inalterado: `Reloaded(None) ≡ Original`; com enriquecimento efetivo, `Acc(Reloaded_OWL) > Acc(Original) > Acc(C4.5)` e `Fid(Reloaded_OWL) > Fid(Original)`, `Fid(Reloaded_OWL) > Fid(C4.5)`, em simultâneo, no mesmo MLP e nas mesmas instâncias — como **objetivo experimental**, não como consequência garantida da qualidade da ontologia.

## 1. Hipóteses
**H2 — semântica OWL expressiva.** Relações de domínio verdadeiras, axiomas e inferências, corretamente modelados e usados por um mecanismo efetivo, melhoram a extração de regras m-of-n **para além** do benefício de uma simples partição de atributos.
* **H2a (valor da expressividade):** F > D e F > E (OWL expressiva vs partição equivalente, com e sem OWL).
* **H2b (valor das relações verdadeiras):** F > G e F > H (neutralizada; permutada ou relações aleatórias controladas).
* **H2c (relevância para a hierarquia global):** F vs A, B, C e F vs J (C4.5) segundo o requisito conjunto.
* **H2d (descritiva):** D vs E mede o efeito da representação OWL plana (igualdade **não** refuta H2).
**Refutável:** H2a/H2b falham se, no regime estudado, F não superar D/E (resp. G/H) em accuracy **e** fidelidade com a evidência prevista. **Pré-condição** (`H2_MECHANISM_PRECONDITION`): se F e D produzirem traços de candidatos idênticos (nenhum literal derivado entra nos candidatos), o contraste é degenerado e H2 fica `UNTESTABLE`, não refutada. Resultados do estudo H1 (B3.4) **não** sustentam H2.

## 2. Fases (cada uma com porta de passagem)
| Fase | Conteúdo | Dados | Estado |
|---|---|---|---|
| S — semântica | testes de correção semântica do contrato e do protótipo (`tests/test_h2_contract_and_mechanisms.py`, 7 testes) | sintético/OWL pequenos | **feito** (sem benchmark) |
| P — mecanismo em sintético | gerador com escalas ordinais latentes e verdade conhecida; verificar que o mecanismo T1/T2 produz candidatos/decisões como o contrato diz e que as ablações atribuem decisões a axiomas | sintético | desenho; **não executar** |
| R — exploratório real | German Credit (DEV, exposto), ontologia O2 da proposta `v0_draft` **após revisão humana**; 1 dataset ⇒ unidade de inferência = 1 | DEV | desenho; **não executar** |
A fase R só começa com: (i) fichas de axioma revistas pelos três papéis (ou declaração explícita do responsável de que corre como «rascunho de IA não validado», com o rótulo no relatório); (ii) fontes verificadas; (iii) mecanismo ligado ao arnês com rastreio por literal; (iv) pré-registo congelado em commit separado.

## 3. Braços (matriz A–J) e implementação em `validation/`
| Braço | Definição | Informação prévia | Implementação |
|---|---|---|---|
| A | TREPAN Original canónico | nenhuma | `TrepanOriginalClassifier` |
| B | Original WIDE (`max_features_per_node = p`) | nenhuma | idem |
| C | Pesquisa estatística genérica de interações (rastreio por pares com controlo de comparações múltiplas) | nenhuma | `pair_screening_units` adaptado a literais binários |
| D | Original com partição correta de atributos **sem OWL** | partição | unidades de pesquisa B3.1 (lista) |
| E | Ontologia plana equivalente (O1) | partição (OWL) | registo B3.1 sobre O1 |
| F | **Ontologia expressiva verdadeira (O2)** | partição + axiomas | O1-unidades + literais derivados T1/T2 (Doc. 51) |
| G | Ontologia expressiva **neutralizada** (O3) | parcial | classes primitivas, mesma estrutura |
| H | Ontologia **permutada** (ordem/composição trocadas) ou **relações aleatórias** estruturalmente comparáveis (I) | informação errada | OWL H e I do `v0_draft` |
| I′ | Engenharia convencional: literais cumulativos com a ordem dos níveis aprendida da taxa de classe **no treino** (dados), e agrupamento por correlação | derivada dos dados | `control_*` + ordenação por treino |
| J | C4.5 canónico | nenhuma | `C45Classifier` |
Todos os braços usam o **mesmo MLP/oráculo congelado, as mesmas instâncias e partições** e a mesma semente. Contrastes essenciais: **F vs A/B/C**; **F vs D/E**; **F vs G/H**; **D vs E**; **F vs J**. O conhecimento adicional fornecido a F (os axiomas) é **contabilizado**: nº de fichas, nº de literais derivados, bits de informação estrutural quando aplicável (Doc. 48 §3) e fração que depende do reasoner.

## 4. Orçamentos comparáveis (cada modalidade separada)
Como cada literal derivado é um teste `1-de-n` que o Original já pode formar (Doc. 51 §1.5), F **não** alarga o espaço de hipóteses: modifica a **pesquisa**. Por isso: (i) **candidatos equiparados** — WIDE com `beam_width` calibrado (só recursos) a igualar candidatos avaliados de F e, em separado, CPU de ajuste; (ii) controlos H/I com **o mesmo n.º e tamanhos de literais derivados** (equivale a mesmo orçamento de candidatos); (iii) consultas ao oráculo (`max_queries`/`min_sample` iguais, contagens reais reportadas); (iv) literais por pesquisa (`pool_parity`); (v) custo de compilação OWL e do reasoner **incluído** no custo total; (vi) limites de árvore iguais (`max_nodes`, `max_depth`, `max_n`) e complexidade reportada; (vii) memória (`tracemalloc`) em subconjunto. Nenhuma igualdade global é afirmada; benefício de informação externa relata-se à parte da eficiência.
Nota sobre mecanismo de fidelidade: com **limite de nós**, um literal cumulativo exprime uma dependência monótona com um único split onde o Original precisaria de vários; esse é o canal plausível de ganho de fidelidade/accuracy a complexidade igual — hipótese a testar, não pressuposto. Se o ganho em accuracy não vier com fidelidade, o estado é `TRADEOFF`, nunca sucesso.

## 5. Métricas
`accuracy_real_labels` (principal), `fidelity_to_oracle` (mesmo MLP/instâncias), `precision_macro`, `macro_f1`, recuperação de regra (quando há verdade), nós, profundidade, condições m-of-n, candidatos gerados/avaliados, consultas, CPU, memória, estabilidade entre sementes e divisões, diferença face a G/H/I′ e face aos genéricos, **e por relação**: estados `AVAILABLE…BENEFIT_OBSERVED` (Doc. 45 Q6) com evidência observacional (literais realmente gerados; escolhas de split que mudam sob ablação).

## 6. Atribuição ontológica (dois níveis, sempre separados)
* **Nível relação (mecanismo):** para cada axioma/literal derivado, ablação (classe passa a primitiva, o resto igual): a mudança de escolha de split atribui-se a esse axioma se o literal derivado foi o escolhido e a ablação muda a decisão; compara-se com a mesma ablação nos controlos H/I para separar efeito do mecanismo de efeito do conhecimento. Medida: n.º de decisões atribuídas por axioma; nunca inferida do nome do braço.
* **Nível estudo (ganho):** `ONTOLOGY_ATTRIBUTED_GAIN` exige, em simultâneo: (1) ganho conjunto accuracy **e** fidelidade de F sobre A; (2) F > D e F > E (para H2a) ou o estado `UNTESTABLE`/`NOT_ADDITIVE` explícito; (3) F > G e F > H/I; (4) F > pelo menos um genérico forte de recursos comparáveis (B, C, I′, K1/K2), reportando `ALL_GENERICS` vs `SOME_GENERICS`; (5) controlos negativos sem vantagem espúria; (6) estabilidade entre divisões/sementes independentes; (7) custo e complexidade documentados; (8) conhecimento anterior e independente da avaliação (fichas, hashes, declaração) e ressalva da revisão humana (`*_REVIEW_PENDING` até haver aprovações verificáveis).
Estados: `RULE_RECOVERY_GAIN`, `FIDELITY_GAIN`, `ACCURACY_GAIN`, `JOINT_GAIN`, `ONTOLOGY_ATTRIBUTED_GAIN`, `NO_ONTOLOGICAL_ADVANTAGE`, `INSUFFICIENT_EVIDENCE`, e para H2 também `UNTESTABLE` e `NOT_ADDITIVE_TO_PARTITION`. Evidência: `observed_only` (um dataset) vs `statistically_sustained` (≥ vários datasets independentes com ontologias independentes — hoje inexistente). Unidade de inferência = dataset ⇒ o estudo R é **exploratório**.

## 7. Controlos negativos e validade
G, H, I (ontologia), N_* (irrelevante, incompleta, enganadora), e **ontologia lógica correta mas sem consequências para o rótulo** (axiomas verdadeiros sobre variáveis irrelevantes): não pode haver ganho atribuído. Controlos sem dados de teste; nenhuma reescolha de axiomas por resultados; H e I construídos com sementes fixadas antes; deteção de coincidência com O2 (se o conteúdo de um literal do controlo coincidir com o verdadeiro, volta a sortear-se segundo regra pré-fixada e regista-se).

## 8. Fase P (sintético, mecanismo com verdade conhecida) — desenho
Variáveis categóricas ordinais com `m` níveis; rótulo = função monótona do nível latente com ruído; partição recuperável **ou não** pela **porta de recuperabilidade** (Doc. 49); ontologia O2 gerada **só** do mapa conceito/níveis/ordem do domínio (mesma garantia T-IND-1: invariante a mudanças do alvo); braços A–J. Mede-se: o mecanismo gera os literais corretos; as ablações atribuem a decisão ao axioma; em condições onde existe efeito, F > D/E/G/H; onde não existe, sem vantagem. O objetivo é **validar o mecanismo**, não demonstrar superioridade em dados reais.

## 9. Sementes, estatística, paragem
Sementes novas e disjuntas (faixa 8000+; calibração 8000–8019, avaliação 8100+, estabilidade 8200+), fixadas antes de qualquer execução e nunca repetidas. Diferenças emparelhadas por semente/divisão; bootstrap; correção de Holm quando houver várias células; sem paragem opcional. Paragem: hipóteses, braços, métricas e sementes congeladas antes; sem novas variantes depois de ver resultados; se H2 não for sustentada, reporta-se e recomenda-se a revisão do enriquecimento (Doc. 44 §15) sem propor automaticamente novos estudos.

## 10. Plano de testes pequenos de correção semântica (feitos e pendentes)
Feitos (Doc. 51 §4): extensões exatas e aninhadas; dependência do reasoner; O3 sem efeito; permutada com perfil de tamanhos igual e conteúdo diferente; ablação remove exatamente o literal; mundo aberto (disjunção não valida linhas; linhas não removidas); fichas bloqueiam validação por IA e derivação de dados; O2 do `v0_draft` válida, 61/61 mapeada, 55 inferências, 10 literais. Pendentes: ligação ao arnês com rastreio por literal; teste de que a decisão atribuída muda sob ablação num caso construído; T-IND análogo para o gerador P; verificador OWL 2 DL formal.

## 11. Orçamento estimado (não autorizado a executar)
Base: medições da B3.3 (≈ 40–45 s por conjunto de sementes × 13 braços; 2,26 CPU-h para 180 conjuntos). Fase P: 4 células × 30 sementes × 11 braços ≈ 1,5 CPU-h; calibração (portas) ≈ 0,5; fase R (German Credit, 61 colunas + 10 literais, 20 divisões × 10 braços, incluindo o reasoner HermiT ≈ 1–3 s por compilação) ≈ 0,4; estabilidade/memória ≈ 0,3. **Total ≈ 2,7 CPU-h; limite duro proposto 4 CPU-h.** O esforço dominante é **humano** (aquisição de fontes, revisão) e **de engenharia** (ligação ao arnês e rastreio), não estimados em CPU.

## 12. Limitações
Um dataset real; ontologia de rascunho de IA sem revisão; fontes por verificar; o mecanismo só reorienta pesquisa dentro do espaço do Original; a fidelidade está limitada pelo MLP; a accuracy do MLP no German Credit é moderada (≈ 0,76 em estudo anterior), pelo que a margem de ganho em fidelidade pode ser pequena.
