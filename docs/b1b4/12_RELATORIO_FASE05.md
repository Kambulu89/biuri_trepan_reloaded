# Documento 12 — Relatório da Fase 0.5 (para decisão antes da Fase 1a)

Base: `main` @ `1789c17`; documentação sobre a branch `claude/repository-architecture-analysis-s1vl5a`. **Não foi executada nenhuma avaliação confirmatória. Nenhum baseline foi modificado. B1–B4 não foram implementados. Não houve merge.**
Execuções feitas: apenas **medições de custo e de equivalência em datasets DEV** (iris, wine, breast_cancer) e a sonda de equivalência do Doc. 7.

## 1. Estado dos 10 entregáveis
| # | Entregável | Estado | Onde |
|---|---|---|---|
| 1 | Manifest v2 (hashes e identificadores experimentais) | **Rascunho** `frozen=false`; hashes do código, regra de seeds, datasets, grelhas, margens, braços, hashes dos documentos; `PENDING` onde depende de dados/ontologias/decisões | `docs/b1b4/manifest_v2/BENCHMARK_MANIFEST_v2.DRAFT.json` (+ `validation/confirm/build_manifest_v2_draft.py`, `tests/test_manifest_v2_draft.py`) |
| 2 | Contrato de equivalência E1–E9 consolidado | ✔ | Doc. 7 |
| 3 | Diagnóstico e plano da V1 | ✔ (+ quantificação no protocolo real) | Doc. 11 |
| 4 | Matriz de elegibilidade dos 12 | ✔ **documental/parcial** (UCI inacessível: só EC1 verificado) | Doc. 8 |
| 5 | Datasets de reserva | ✔ (8, ordem fixa) | Doc. 8 §4 |
| 6 | Protocolo de autoria independente de ontologias | ✔ | Doc. 9 |
| 7 | Contrato de congelamento OWL | ✔ | Doc. 9 §5 |
| 8 | Grelhas e margens fundamentadas | ✔ (revisão matemática; **decisões D-8 e D-14 pendentes**) | Doc. 10 |
| 9 | Estimativa de custo computacional | ✔ **com achado de inexequibilidade** | Doc. 10 §4 |
| 10 | Critérios de entrada e conclusão da Fase 1a | ✔ | Doc. 11 §5–6 |

## 2. Achados novos (por ordem de importância)
1. **Custo (P0 de planeamento).** O protocolo completo (Estágio 1, seleção aninhada B3) custa **≈ 1 990 CPU-horas** (otimista) ou ≫ 10 000 h (pessimista) para os 10 datasets elegíveis; só 6 datasets com n ≤ 5 000 custam ≈ 300–900 h. Medido: unidade v1 em breast_cancer = 334 s, 79 % em tuning estrutural. Ver Doc. 10 §4. **Decisão D-14.**
2. **Deriva do código «congelado» desde o v1.** Seis ficheiros do `frozen_code_sha256` do manifest v1 mudaram: `core/trepan_original.py`, `core/trepan_scientific_tuning.py`, `core/benchmark/{runner,semantic,preprocessing,timing}.py` (otimizações «exatas» de engenharia, checkpoint, contabilidade do oráculo; `semantic.py` também pelo isolamento OWL do PR #9).
   `c45_j48_tree.py`, `scientific_experiment_contract.py` e `trepan_reloaded_historical.py` **não** mudaram. O código do manifest v1 (`code_commit 5788562`) não tem relatório de equivalência versionado: `validation/benchmark/manifest.py` procura `EQUIVALENCE_REPORT_v1_to_v2.json`, que **não existe no repositório**.
   Resultado da verificação feita agora (DEV): **EQV_RESULT**. **Decisão D-13.**
3. **German Credit está exposto (EC1)** — `counterfactuals/experimentos/german_credit/training_summary.json` regista `trepan_fidelity=0,994` e `trepan_reloaded_fidelity=0,927` (implementação anterior) e o dataset está no registo de investigação contrafactual. Inelegível; substituído pela reserva R1/R2 por ordem. (Isto é um resultado histórico em que o Reloaded ficou abaixo do Original em fidelidade; não é usado para afinar nada.)
4. **Occupancy Detection é série temporal (EC4)** — inelegível por critério estrutural, antes de qualquer resultado.
5. **UCI inacessível a partir do ambiente (403 do proxy)** — nenhum dado foi descarregado nem inspecionado; as dimensões/licenças são «esperado, não verificado». Para desbloquear: *Allowed domains* += `archive.ics.uci.edu` (Doc. 8 §7) ou correres `validation/confirm/inspect_candidates.py` localmente.
6. **O runner só aceita matrizes numéricas** (`encoding: none`): Car, Shoppers, Mushroom, Bank exigem um estágio determinístico de codificação por dicionário (EC11) no *loader* — **dependência D-12**, a desenhar na Fase 1b, fora do núcleo.
7. **V1 no protocolo real:** Reloaded sem ontologia consulta o oráculo 2,05× (iris) / 2,15× (breast_cancer) o que o Original consulta (9 896 vs 20 272; 15 054 vs 32 426), tempo 2,7× / 1,3×, mesma árvore. Reforça o Doc. 7/11.
8. **Nomenclatura do manifest:** `validation/benchmark/manifest.py` já designa «v2» uma versão de *engenharia* (`MANIFEST_VERSION="2"`, `ENGINEERING_CHANGES_V2`), ainda não gerada. O v2 científico proposto **estende** esse (mesmo ficheiro, gerado só na entrada da Fase 1a); o rascunho está fora de `validation/benchmark/` para o `latest_manifest_path()` não o tratar como congelado.

## 3. Decisões
**Resolvidas por ti:** contrato EQ (allowlist vazia) · V1 é não-conformidade a corrigir na Fase 1a · arquitetura «Reloaded = Original + extensão ontológica» · B1–B4 só ontológicos · sem rótulos reais / híbrido só com protocolo separado · hierarquia como objetivo pré-registado · margens `δ_A=δ_F=ν_F=0` · seeds DEV 5, CONFIRM 10.
**Pendentes (precisam de ti):**
| Id | Pergunta | Recomendação |
|---|---|---|
| D-7 | Confirmar a lista após inspeção (2 substituições previstas: Occupancy, German Credit → R1 Phishing, R2 MAGIC) e quem escreve as ontologias | desbloquear UCI ou correr a inspeção; ontologias escritas por humano ≠ desenvolvedor |
| D-8 | Interpretação de `ν_F=0` (≥, não-estrita) e grelha `lambda_cap` `[0,.05,.10,.20,.40,.80]` | confirmar; revisão matemática no Doc. 10 §1–2 |
| D-12 | Estágio de codificação determinística (EC11) no *loader* | aprovar o desenho na Fase 1b |
| D-13 | Deriva de hashes do Original/tuning desde o v1 | re-congelar hashes **só** com relatório de equivalência v1→atual versionado (Doc. 12 §2.2); Digits fica por verificar |
| D-14 | Orçamento/escopo de custo (Doc. 10 §4) | teto global de dimensão + aceleração exata + faseamento |
| D-15 | Atributos de *reporting* (`semantic_split_audit_`, `error_region_audit_` vazios sem Ω) fora da allowlist? | tratá-los como fora de E1–E9, sem allowlist (confirmar) |
| D-16 | Observação V2: `mirror_when_no_semantic_effect=True` duplica consultas ao oráculo | avaliar na Fase 1a; não é V1 |
| D-17 | `n_eff` do B2: só linhas reais para aceitar | confirmar (Doc. 3 §10, D-5) |

## 4. Plano exato para corrigir a V1 (resumo; detalhe no Doc. 11)
(0) âncora dourada do comportamento com Ω efetiva em `main@1789c17` → (1) predicado único `_ontology_effective()` = `reload_extension_active_` → (2) despacho no topo dos ganchos do Reloaded para o Original quando `Ω=∅` (sem `predict_proba` extra) → (3) testes unitário/integração/propriedade com oráculo-espião que hash o fluxo de entradas →
(4) verificar Ω≠∅ idêntico à âncora → (5) custo `reloaded_core/original ≤ 1,1` → (6) agnosticismo + suite completa + CI Windows 3.11/3.12 no commit exato. Só em `core/trepan_reloaded_historical.py` e testes. B1–B4 só depois.

## 5. Integridade científica — declaração
Nenhum resultado confirmatório observado · nenhum dataset CONFIRM descarregado · baselines intocados (verificação por hash no manifest rascunho) · medições em DEV apenas para custo/equivalência · decisões de elegibilidade (EC1–EC11, reservas e substituição) fixadas **antes** de qualquer avaliação, por critérios estruturais e não por desempenho.
