# Relatório de validação semântica

Versão do pipeline semântico: `9.3.0`. Gerado por `scripts/render_semantic_report.py` a partir de `scripts/run_semantic_validation.py`; todos os números vêm do JSON da execução.

**Protocolo:** só dados de desenvolvimento (nenhum teste externo); `OntologyProcessor` refeito em cada fold só com o treino do fold; MLP base e MLP+OWL otimizados em separado com a mesma lista de candidatos, folds internos e semente; utilidade ponderada e IC bootstrap emparelhado (95%) definidos antes de correr; limiares não foram ajustados depois de ver resultados.

## 1. Cobertura OWL e estado da ontologia

| Dataset | Estado | Mapeadas | Ambíguas | Colisões | Entropia | Profundidade | ABox | Reasoner | Riqueza |
|---|---|---|---|---|---|---|---|---|---|
| breast_cancer | VALID_DOMAIN_ONTOLOGY | 30/30 (1.00) | 0 | 0 | 1.00 | 2 | SAFE | consistente (0.64 s, 0 inferidos) | limited (statistic_roles) |
| digits | VALID_DOMAIN_ONTOLOGY | 64/64 (1.00) | 0 | 0 | 1.00 | 2 | SAFE | consistente (1.10 s, 0 inferidos) | poor (só taxonomia) |
| iris | VALID_DOMAIN_ONTOLOGY | 4/4 (1.00) | 0 | 0 | 1.00 | 2 | SAFE | consistente (0.94 s, 0 inferidos) | poor (só taxonomia) |
| wine | VALID_DOMAIN_ONTOLOGY | 13/13 (1.00) | 0 | 0 | 1.00 | 2 | SAFE | consistente (0.92 s, 0 inferidos) | poor (só taxonomia) |

## 2. Features geradas, estáveis e selecionadas

| Dataset | Semente | Geradas | Retidas (novidade) | Estáveis | Selecionadas | Decisão |
|---|---|---|---|---|---|---|
| breast_cancer | 42 | 53 | 53 | 9 | 0 | `REJECT_NO_INFORMATIONAL_GAIN` |
| breast_cancer | 43 | 53 | 53 | 8 | 0 | `REJECT_NO_INFORMATIONAL_GAIN` |
| breast_cancer | 44 | 53 | 53 | 9 | 9 | `ACCEPT_PARTIAL_FEATURE_SET` |
| digits | 42 | 3 | 3 | 2 | 0 | `REJECT_NO_INFORMATIONAL_GAIN` |
| iris | 42 | 2 | 2 | 2 | 0 | `REJECT_DEGRADATION` |
| iris | 43 | 2 | 2 | 2 | 2 | `ACCEPT_NON_INFERIOR_WITH_SECONDARY_GAIN` |
| iris | 44 | 2 | 2 | 2 | 2 | `ACCEPT_NON_INFERIOR_WITH_SECONDARY_GAIN` |
| wine | 42 | 3 | 3 | 2 | 0 | `REJECT_DEGRADATION` |
| wine | 43 | 3 | 3 | 2 | 0 | `REJECT_NO_INFORMATIONAL_GAIN` |
| wine | 44 | 3 | 3 | 2 | 0 | `REJECT_NO_INFORMATIONAL_GAIN` |

## 3. MLP base vs MLP + OWL (OOF, desenvolvimento)

| Dataset | Sem. | Balanced Acc. base | + OWL | Macro-F1 base | + OWL | Ganho de utilidade | IC 95% | Evidência | MLP aceite | TREPAN disponível |
|---|---|---|---|---|---|---|---|---|---|---|
| breast_cancer | 42 | 0.967 | 0.972 | 0.968 | 0.974 | 0.0007 | [-0.017; 0.016] | n/a | False | True |
| breast_cancer | 43 | 0.969 | 0.973 | 0.972 | 0.974 | -0.0007 | [-0.012; 0.011] | n/a | False | True |
| breast_cancer | 44 | 0.968 | 0.980 | 0.970 | 0.981 | 0.0081 | [-0.003; 0.020] | fraca | True | True |
| digits | 42 | 0.957 | 0.953 | 0.957 | 0.953 | 0.0003 | [-0.020; 0.016] | n/a | False | True |
| iris | 42 | 0.960 | 0.947 | 0.960 | 0.947 | -0.0240 | [-0.051; -0.007] | n/a | False | True |
| iris | 43 | 0.940 | 0.960 | 0.940 | 0.960 | 0.0162 | [-0.007; 0.045] | fraca | True | True |
| iris | 44 | 0.960 | 0.967 | 0.960 | 0.967 | 0.0020 | [-0.007; 0.020] | fraca | True | True |
| wine | 42 | 0.984 | 0.979 | 0.983 | 0.977 | -0.0093 | [-0.024; -0.003] | n/a | False | True |
| wine | 43 | 0.984 | 0.986 | 0.983 | 0.983 | -0.0038 | [-0.021; 0.019] | n/a | False | True |
| wine | 44 | 0.973 | 0.979 | 0.972 | 0.977 | 0.0020 | [-0.017; 0.023] | n/a | False | True |

Distribuição das decisões: `ACCEPT_NON_INFERIOR_WITH_SECONDARY_GAIN`=2, `ACCEPT_PARTIAL_FEATURE_SET`=1, `REJECT_DEGRADATION`=2, `REJECT_NO_INFORMATIONAL_GAIN`=5.

## 4. Ablação por componente

Cada linha restringe as features semânticas a um tipo. `NO_NOVEL_FEATURES` significa que a OWL desse dataset não produz features desse tipo (não é uma falha do pipeline).

| Dataset | Sem. | Agregados | Relacionais | Restrições/categóricas | Inferidas pelo reasoner | Todas |
|---|---|---|---|---|---|---|
| breast_cancer | 42 | `R_NO_INFORMATIONAL_GAIN` (0.003) | `A_PARTIAL_FEATURE_SET` (0.008) | `R_NO_NOVEL_FEATURES` (—) | `R_NO_NOVEL_FEATURES` (—) | `R_NO_INFORMATIONAL_GAIN` (0.001) |
| breast_cancer | 43 | `R_NO_INFORMATIONAL_GAIN` (-0.004) | `R_NO_INFORMATIONAL_GAIN` (-0.001) | `R_NO_NOVEL_FEATURES` (—) | `R_NO_NOVEL_FEATURES` (—) | `R_NO_INFORMATIONAL_GAIN` (-0.001) |
| breast_cancer | 44 | `A_PARTIAL_FEATURE_SET` (0.010) | `A_PARTIAL_FEATURE_SET` (0.006) | `R_NO_NOVEL_FEATURES` (—) | `R_NO_NOVEL_FEATURES` (—) | `A_PARTIAL_FEATURE_SET` (0.008) |
| digits | 42 | `R_NO_INFORMATIONAL_GAIN` (0.000) | `R_NO_NOVEL_FEATURES` (—) | `R_NO_NOVEL_FEATURES` (—) | `R_NO_NOVEL_FEATURES` (—) | `R_NO_INFORMATIONAL_GAIN` (0.000) |
| iris | 42 | `R_DEGRADATION` (-0.024) | `R_NO_NOVEL_FEATURES` (—) | `R_NO_NOVEL_FEATURES` (—) | `R_NO_NOVEL_FEATURES` (—) | `R_DEGRADATION` (-0.024) |
| iris | 43 | `A_NON_INFERIOR_WITH_SECONDARY_GAIN` (0.016) | `R_NO_NOVEL_FEATURES` (—) | `R_NO_NOVEL_FEATURES` (—) | `R_NO_NOVEL_FEATURES` (—) | `A_NON_INFERIOR_WITH_SECONDARY_GAIN` (0.016) |
| iris | 44 | `A_NON_INFERIOR_WITH_SECONDARY_GAIN` (0.002) | `R_NO_NOVEL_FEATURES` (—) | `R_NO_NOVEL_FEATURES` (—) | `R_NO_NOVEL_FEATURES` (—) | `A_NON_INFERIOR_WITH_SECONDARY_GAIN` (0.002) |
| wine | 42 | `R_DEGRADATION` (-0.009) | `R_NO_NOVEL_FEATURES` (—) | `R_NO_NOVEL_FEATURES` (—) | `R_NO_NOVEL_FEATURES` (—) | `R_DEGRADATION` (-0.009) |
| wine | 43 | `R_NO_INFORMATIONAL_GAIN` (-0.004) | `R_NO_NOVEL_FEATURES` (—) | `R_NO_NOVEL_FEATURES` (—) | `R_NO_NOVEL_FEATURES` (—) | `R_NO_INFORMATIONAL_GAIN` (-0.004) |
| wine | 44 | `R_NO_INFORMATIONAL_GAIN` (0.002) | `R_NO_NOVEL_FEATURES` (—) | `R_NO_NOVEL_FEATURES` (—) | `R_NO_NOVEL_FEATURES` (—) | `R_NO_INFORMATIONAL_GAIN` (0.002) |

## 5. Controlo negativo: semântica baralhada

A mesma ontologia, mas com a atribuição feature→entidade baralhada. Se o ganho real fosse conhecimento semântico, os controlos deveriam ficar sistematicamente abaixo do real.

| Dataset | Sem. | Ganho real | Ganhos dos controlos | Controlos ≥ real |
|---|---|---|---|---|
| breast_cancer | 42 | 0.0007 | 0.0018, -0.0028, -0.0116 | 1/3 |
| breast_cancer | 43 | -0.0007 | -0.0101, 0.0021, 0.0011 | 2/3 |
| breast_cancer | 44 | 0.0081 | 0.0078, 0.0001, 0.0021 | 0/3 |
| digits | 42 | 0.0003 | 0.0052, 0.0046, -0.0018 | 2/3 |
| iris | 42 | -0.0240 | -0.0067, -0.0240, -0.0153 | 3/3 |
| iris | 43 | 0.0162 | 0.0162, 0.0162, 0.0162 | 3/3 |
| iris | 44 | 0.0020 | 0.0020, 0.0020, -0.0067 | 2/3 |
| wine | 42 | -0.0093 | -0.0091, -0.0091, -0.0139 | 2/3 |
| wine | 43 | -0.0038 | -0.0082, -0.0040, -0.0027 | 1/3 |
| wine | 44 | 0.0020 | -0.0043, -0.0111, 0.0020 | 1/3 |

No total, 17 de 30 controlos com semântica baralhada tiveram ganho ≥ ao da ontologia real.

## 6. Leitura (calculada a partir dos resultados acima)

- Execuções: 10; MLP+OWL aceite em 3 (0 com evidência forte, i.e. IC da utilidade acima de zero; 3 com evidência fraca).
- Controlos com semântica baralhada com ganho ≥ ao real: 17/30 — mais de metade: os ganhos observados não se distinguem do ruído.
- Datasets cuja OWL gera features relacionais (as únicas não lineares): breast_cancer.
- A semântica continua disponível para o TREPAN em todas as execuções (`semantic_trepan_available`), mesmo quando o enriquecimento do MLP é rejeitado: os dois estados são independentes.
- **Limites:** poucas sementes e amostras pequenas; um único protocolo de validação cruzada; ontologias de benchmark sem propriedades de objeto, restrições nem limites declarados (ver `docs/SEMANTIC_PIPELINE_AUDIT.md`); nos datasets em que o baralhamento preserva a estrutura de grupos (ex. iris, 2 grupos) o controlo pode coincidir com o real. Nenhum limiar foi ajustado depois de ver estes resultados. Este relatório **não** demonstra superioridade do MLP ontológico nem do TREPAN Reloaded.
