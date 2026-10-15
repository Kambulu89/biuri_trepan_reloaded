# Doc. 28 — Requisito científico obrigatório: hierarquia conjunta em dois níveis

Estado: **norma do projeto** (CLAUDE.md §2). Contrato executável `core/hierarchy_contract.py` (13 testes, incluindo a verificação exaustiva dos 3⁶ = 729 padrões de ordem) e relatório retroativo só de leitura `validation/benchmark/hierarchy_joint_report.py`. **Decisão do utilizador (Opção 1)**: todos os datasets e sementes permanecem na avaliação; nenhuma unidade é excluída por `Original ≤ C4.5`; os baselines, o MLP e os resultados históricos não são alterados.

## 1. Requisito final (inalterado)
`Accuracy_Reloaded_OWL > Accuracy_Original > Accuracy_C4.5` **e** `Fidelity_Reloaded_OWL > Fidelity_Original` **e** `Fidelity_Reloaded_OWL > Fidelity_C4.5`, em conjunto, nas **mesmas instâncias** e, na fidelidade, contra o **mesmo MLP/oráculo** (FrozenOracle, mesmo `oracle_id`). Melhorar uma métrica perdendo na outra é `TRADEOFF`, não sucesso conjunto. **Accuracy ≠ precision**: a métrica preditiva principal é `accuracy_real_labels`; `precision_macro_real_labels` e `macro_f1_real_labels` são complementares (só relatório).

## 2. Dois níveis de avaliação (obrigatórios, reportados separadamente)
**Nível 1 — ganho ontológico direto** (`evaluate_direct_gain`): `A_R` vs `A_Oeq` e `F_R` vs `F_Oeq`, com o **Original equivalente** nas mesmas condições (amostrador, oráculo, partições, orçamento, representação quando se isola a orientação semântica). Estados: `ONTOLOGICAL_JOINT_GAIN`, `TRADEOFF`, `ACCURACY_ONLY_GAIN`, `FIDELITY_ONLY_GAIN`, `EQUIVALENT` (igualdade exata), `NO_GAIN`. Além disso (`attribute_to_ontology`) o ganho só se **atribui** à ontologia (`ATTRIBUTED`) se houver ganho conjunto estrito também sobre os controlos exigidos — ontologia permutada, grupos aleatórios, engenharia convencional de features — e Original/Reloaded com o mesmo amostrador; controlo em falta ⇒ `NOT_ASSESSABLE` (nunca se atribui por omissão); ganho que resulte só de alteração genérica do amostrador não é atribuído à ontologia.
**Nível 2 — superioridade global** (`evaluate_global`): cadeia U1 `A_R>A_O`, U2 `A_O>A_C`, U3 `A_R>A_C`, U4 `F_R>F_O`, U5 `F_R>F_C` (estritas). Estado por unidade (precedência): `INSUFFICIENT_EVIDENCE` › `JOINT_HIERARCHY_SATISFIED` (U1–U5) › `HIERARCHY_INFEASIBLE` (U2 falsa: `Accuracy_Original ≤ Accuracy_C4.5` **nessa unidade, com os resultados fixados dos baselines** — não é impossibilidade universal e a unidade **não** é excluída) › `TRADEOFF` › `NOT_SATISFIED`. Sinalizadores **ortogonais** reportados em separado: estado do ganho direto, superioridade de accuracy (U1∧U2∧U3), superioridade de fidelidade (U4∧U5), compromisso face ao Original, compromisso face ao C4.5, inviabilidade pelos baselines. O Nível 1 **nunca** substitui o Nível 2 nem autoriza declarar sucesso global.
**Mesmas instâncias e mesmo MLP**: `evaluate_*` são funções puras sobre números; a garantia está em `check_comparability`, aplicada pelo relatório a **todas** as linhas usadas (cada réplica e cada braço): `split_hash` e `n_test` iguais e `oracle_id` das árvores igual ao oráculo contra o qual a fidelidade do C4.5 foi medida (`fidelity_oracle_id`). Qualquer falha ⇒ o veredito passa a `INSUFFICIENT_EVIDENCE`.

## 3. Estado lógico ≠ evidência estatística
`classify_evidence` é separada das desigualdades: `OBSERVED_JOINT_GAIN` (médias entre datasets positivas em accuracy **e** fidelidade), `SUPPORTED_JOINT_GAIN` (adicionalmente: avaliação **confirmatória**, ≥ 12 datasets independentes — Doc. 2 §7 — e limite inferior dos IC95 (t e bootstrap entre datasets) > 0 em ambas), `TRADEOFF`, `NOT_SATISFIED`, `INSUFFICIENT_EVIDENCE`. A unidade de inferência é o **dataset**: sementes/réplicas são médias dentro do dataset (variabilidade intra-dataset reportada à parte) e **nunca** substituem datasets independentes (200 unidades em 4 datasets ⇒ no máximo `OBSERVED`; testado). Reporta-se: diferenças pareadas, IC entre datasets, DP entre datasets e DP intra-dataset. DEV já observado e seleção retroativa de configurações **nunca** são confirmatórios.

## 4. Seleção e aceitação de candidatos (treino/validação interna apenas)
1. **Elegibilidade (não-inferioridade)**: `A_R ≥ A_O` e `F_R ≥ F_O` na validação interna face ao Original equivalente (`non_inferior`; tolerâncias ν só se pré-registadas). É um filtro, **não** superioridade.
2. **Preferência**: entre elegíveis privilegiam-se os candidatos com ganho **simultâneo estrito** (`A_R > A_O` e `F_R > F_O`), ponderando fidelidade, accuracy, complexidade (nós/profundidade) e estabilidade (variância entre dobras/réplicas); em empate de ganho prefere-se menor complexidade. Se não houver elegível com ganho, mantém-se a configuração Reloaded ≡ Original (EQ).
3. **Declaração**: superioridade observada exige `A_R > A_O` e `F_R > F_O`; superioridade conjunta completa exige também as condições face ao C4.5 (U2, U3, U5). Igualdade não é superioridade. O teste nunca autoriza retroativamente uma configuração; o teste externo é lido uma vez com a configuração congelada.

## 5. Estado atual da evidência (retroativo, só leitura; DEV, 18 unidades, TODAS incluídas; exploratório, nunca confirmatório)
Relatórios: `docs/evidence/b1b4/hierarchy_joint/two_levels/*.json`. Comparabilidade (mesmas instâncias, mesmo MLP): **18/18 unidades comparáveis em todas as especificações**. Os ficheiros `docs/evidence/b1b4/hierarchy_joint/*.json` da versão anterior (estados `HIERARCHY_*` do contrato de nível único) mantêm-se como histórico e ficam **substituídos** por estes.

| Especificação (R vs Original equivalente) | Nível 1: estados por unidade | elegíveis (não-inf.) / ganho conjunto estrito | Evidência (datasets) | Atribuição à ontologia |
|---|---|---|---|---|
| Doc. 26 **D vs B** (OWL+C vs Original+C: mesmo amostrador) | NO_GAIN 10, TRADEOFF 4, JOINT_GAIN 2, EQUIVALENT 2 | 4 / 2 de 18 | `NOT_SATISFIED` (ΔA −0,0067, ΔF −0,0092; IC boot. ΔA [−0,012; −0,002], ΔF [−0,015; −0,003]) | `NOT_ASSESSABLE` (sem controlos sob C) |
| Doc. 26 C vs A (histórico) | NO_GAIN 6, JOINT_GAIN 6, TRADEOFF 3, EQUIV. 2, ACC_ONLY 1 | 9 / 6 | `NOT_SATISFIED` (ΔA −0,0008, ΔF −0,0021) | `NOT_ASSESSABLE` |
| Doc. 22 `coh_full` vs `trepan_original_enr` (mesma representação) | JOINT_GAIN 5, NO_GAIN 6, EQUIV. 5, TRADEOFF 1, FID_ONLY 1 | 11 / 5 | `OBSERVED_JOINT_GAIN` (ΔA +0,0009 [−0,004; 0,006], ΔF +0,0071 [0,003; 0,013]) | controlos: permutada **melhor** que a real (ΔA −0,014, ΔF −0,006) ⇒ não atribuído; convencional `TRADEOFF`; grupos aleatórios em falta ⇒ `NOT_ASSESSABLE` (unidades: 13 `NOT_ATTRIBUTED`, 5 `NOT_ASSESSABLE`) |
| Doc. 22 `coh_full` vs Original canónico | JOINT_GAIN 6, NO_GAIN 7, ACC_ONLY 2, EQUIV. 3 | 11 / 6 | `NOT_SATISFIED` (ΔA −0,0038, ΔF −0,0018) | idem |
| Histórico `reloaded_owl_full` vs Original (controlo permutado) | JOINT_GAIN 5, NO_GAIN 8, TRADEOFF 3, ACC_ONLY 1, EQUIV. 1 | 7 / 5 | `NOT_SATISFIED` (ΔA −0,0073, ΔF −0,013) | permutada melhor que a real ⇒ não atribuído |

| Nível 2 (todas as 18 unidades) | JOINT_HIERARCHY_SATISFIED | HIERARCHY_INFEASIBLE | TRADEOFF | NOT_SATISFIED | superioridade de accuracy / de fidelidade (unidades) |
|---|---|---|---|---|---|
| D vs B | 1 | 9 | 1 | 7 | 1 / 3 |
| C vs A | 1 | 12 | 1 | 4 | 2 / 3 |
| `coh_full` vs `trepan_original_enr` | 0 | 12 | 1 | 5 | 0 / 3 |
| `coh_full` vs Original | 0 | 13 | 1 | 4 | 0 / 1 |
| histórico vs Original | 0 | 13 | 0 | 5 | 0 / 1 |
Compromissos face ao Original: D vs B 4/18; face ao C4.5: 3/18. `Original > C4.5` em accuracy só em 9/18 (B), 6/18 (A) e 5/18 (Original canónico) unidades — `HIERARCHY_INFEASIBLE` nas restantes, **reportado, não corrigido, sem exclusão** (percentagens sempre sobre as 18 unidades).

**Respostas às três perguntas (estado atual, DEV, exploratório):**
1. *O Reloaded com ontologia melhora simultaneamente accuracy e fidelidade face ao Original equivalente?* **Não demonstrado.** O único sinal `OBSERVED_JOINT_GAIN` (coerente vs Original na mesma representação) tem accuracy ≈ 0 (IC inclui 0); sob o mesmo amostrador (D vs B) o contraste é negativo.
2. *O efeito é causado pelo conhecimento ontológico?* **Não atribuído**: a ontologia permutada iguala ou supera a real; o controlo de grupos aleatórios ainda não foi executado.
3. *O Reloaded com ontologia cumpre a hierarquia global?* **Não**: 0–1 de 18 unidades por especificação, e em 9–13 unidades a cadeia é inviável com os baselines fixados.
Uma resposta positiva à 1 não implicaria a 2 nem a 3; o objetivo final continua a ser alcançar as três.

## 6. Consequências para o plano
Critérios de aceitação das intervenções (manifest v3 draft; Docs. 26–27) incluem os dois níveis, a atribuição por controlos e o nível de evidência. B3-a (Doc. 27) será avaliado por `ontology_vs_permuted` e `random_groups`/`conventional` **e** pelos dois níveis. Unidades `HIERARCHY_INFEASIBLE` mostram sempre, em paralelo, o Nível 1 (contribuição ontológica). Nenhuma mudança de estado das intervenções (B1/B2-C não promovidos; amostragem coerente experimental; amostrador C melhoria genérica sob investigação; B2-A/B não autorizados; B3-a desenho; B3-b desenho; B3-c/d adiados; B4 não implementado) decorre desta formalização.
