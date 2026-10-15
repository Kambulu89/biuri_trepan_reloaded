# Doc. 28 — Requisito científico obrigatório: hierarquia conjunta (accuracy E fidelidade)

Estado: **norma do projeto** (CLAUDE.md §2), com contrato executável `core/hierarchy_contract.py` (7 testes) e relatório retroativo só de leitura `validation/benchmark/hierarchy_joint_report.py`. Não altera baselines, resultados históricos nem seleção. Substitui, quando mais estrito, as formulações anteriores dos Docs. 5 e 2 (que ficam como histórico, com notas de revisão).

## 1. Requisito
O TREPAN Reloaded + OWL deve ser otimizado e avaliado para alcançar **simultaneamente**:
| Métrica | Condições |
|---|---|
| **Accuracy** (rótulos reais) | U1 `A_R > A_O` · U2 `A_O > A_C` · U3 `A_R > A_C` |
| **Fidelidade ao MLP** (`fidelity_to_oracle`) | U4 `F_R > F_O` · U5 `F_R > F_C` |
Mesmas instâncias de teste para todos os modelos; a fidelidade usa o **mesmo MLP/oráculo congelado** (mesmo `oracle_id`) para Original, Reloaded e C4.5 (C4.5 treina nos rótulos reais, a sua fidelidade mede-se contra esse MLP, sem o usar).
**Sucesso científico de uma intervenção = U1–U5 em conjunto.** Melhorar a accuracy e perder fidelidade (ou o inverso) é `TRADEOFF_*` e **não** conta como sucesso; ganhar ao Original sem ganhar ao C4.5 também não. Nenhuma superioridade se declara sem evidência experimental (`observed_only` vs `statistically_sustained`).

## 2. Accuracy não é precision
- Métrica preditiva **principal**: `accuracy_real_labels` (acertos / total, rótulos reais).
- **Complementares, sempre reportadas ao lado e nunca usadas para declarar a hierarquia**: `precision_macro_real_labels`, `macro_f1_real_labels` (e `balanced_accuracy`, recall/`minority_recall`, `weighted_f1`).
- Todos os quadros/relatórios têm de nomear a métrica exata; «precisão» em português **não** pode designar a accuracy.

## 3. Estados e modos (`evaluate_joint`)
Estados por unidade: `HIERARCHY_SATISFIED` (U1–U5) · `HIERARCHY_NOT_SATISFIED` · `HIERARCHY_INFEASIBLE` (U2 falha: o Original não supera o C4.5 em accuracy — propriedade dos baselines congelados; **reporta-se, não se corrige**) · `INSUFFICIENT_EVIDENCE` (dados em falta). Modos: `JOINT_GAIN`, `TRADEOFF_ACCURACY_UP_FIDELITY_DOWN`, `TRADEOFF_FIDELITY_UP_ACCURACY_DOWN`, `ACCURACY_ONLY_GAIN`, `FIDELITY_ONLY_GAIN`, `NO_GAIN_OVER_ORIGINAL`, `GAIN_OVER_ORIGINAL_BUT_NOT_OVER_C45`, `ORIGINAL_NOT_ABOVE_C45_IN_ACCURACY`. Comparações estritas (empates não contam); margens opcionais `delta_acc`, `delta_fid` pré-registadas.
Agregação (dataset/portfólio): contagens por estado/modo, contrastes médios com EP entre unidades, macro por dataset; `statistically_sustained` só com o critério pré-registado do protocolo (Doc. 2 §6). Réplicas dentro de uma unidade são médias, não unidades.

## 4. Regra de otimização/seleção (proposta; implementação só com autorização)
Seleção e tuning usam **apenas treino e validação interna** (OOF). Um candidato do Reloaded+OWL é elegível se, na validação interna, face ao **Original equivalente** (mesmo oráculo, representação, orçamento, semente): `ΔA_val ≥ 0` **e** `ΔF_val ≥ 0` (sem compromisso) e a estrutura ontológica tem controlo permutado/convencional computado. Entre elegíveis maximiza-se `min(ΔA_val/EP_A, ΔF_val/EP_F)` (melhora o pior dos dois). Se nenhum for elegível, mantém-se a configuração Reloaded ≡ Original (EQ); **nunca** se escolhe por resultado de teste, e nunca se piora um baseline. O teste externo avalia-se uma só vez com a configuração congelada. As intervenções B1–B4 e a amostragem têm de ser desenhadas, ablacionadas e relatadas com este objetivo conjunto, mais os controlos que separam efeito ontológico de efeito genérico (Original com a mesma amostragem/representação).

## 5. Estado atual da evidência (relatório retroativo; DEV, 18 unidades; observado, não sustentado)
Mesmas instâncias, mesmo MLP; média das réplicas por unidade (`docs/evidence/b1b4/hierarchy_joint/`).
| Comparação (Reloaded+OWL vs Original equivalente) | SATISFIED | NOT_SATISFIED | INFEASIBLE | ΔA(R−O) | ΔF(R−O) | ΔA(R−C4.5) | ΔF(R−C4.5) |
|---|---|---|---|---|---|---|---|
| Doc. 26: D vs B (OWL + amostrador C vs Original + C) | 1 | 8 | 9 | −0,0063 | −0,0098 | +0,0047 | +0,0043 |
| Doc. 26: C vs A (OWL histórico vs Original histórico) | 1 | 5 | 12 | −0,0013 | −0,0028 | −0,0134 | −0,0053 |
| Doc. 22: Reloaded OWL histórico vs Original | 0 | 5 | 13 | −0,0081 | −0,0140 | −0,0175 | −0,0122 |
| Doc. 22: coerente (`coh_full`) vs Original | 0 | 5 | 13 | −0,0042 | −0,0019 | −0,0136 | −0,0001 |
Leituras: (i) **o requisito conjunto não está satisfeito por nenhuma intervenção testada**; (ii) `Original > C4.5` em accuracy só se verifica em metade ou menos das unidades (B: 9/18; A: 6/18): a cadeia é `INFEASIBLE` nessas unidades com os baselines congelados — isto limita o que qualquer mecanismo ontológico pode demonstrar e deve ser reportado como achado, **não** corrigido (o C4.5 não pode ser enfraquecido, o Original não pode ser alterado); (iii) o amostrador C melhora o Original (Doc. 26: B−A) mas isso é efeito genérico não ontológico; (iv) as precisões e macro-F1 acompanham a accuracy (complementares: R 0,931/0,927, O 0,936/0,934, C4.5 0,925/0,922 em D vs B).

## 6. Consequências para o plano
1. Critérios de aceitação das intervenções (manifest v3 draft, Docs. 26–27) passam a incluir U1–U5 em conjunto, além dos critérios pré-registados específicos de cada documento (que se mantêm).
2. B3-a (Doc. 27) é avaliado por `ontology_vs_permuted` **e** pelo veredito conjunto face ao Original equivalente; ganho unilateral não basta.
3. Para unidades `INFEASIBLE`, o relatório mostra separadamente (a) o contraste Reloaded−Original (contribuição ontológica) e (b) a condição U2 como propriedade dos baselines.
4. Qualquer afirmação de superioridade exige: controlos ontológicos favoráveis, equivalência EQ verificada, mesmo MLP, teste externo lido uma vez, e número de comparações/unidades declarado.
