# OWL enriquecida — Breast Cancer Wisconsin Diagnostic

`data/breast_cancer_enriched.owl`, gerada por `scripts/build_breast_cancer_enriched_owl.py` a partir da OWL original
(mesmas entidades, labels e aliases; só TBox, sem instâncias nem limiares aprendidos).

## O que acrescenta
| Conhecimento (da descrição do dataset, Street et al. 1993) | Efeito no sistema |
|---|---|
| `measurementFamily` + `statisticRole` (mean / error / worst) nas 30 propriedades | 50 features relacionais (5 por família × 10 famílias) |
| Super-propriedade por família medida (radius, texture, …) | 10 agregados por família |
| Super-propriedade por conceito (tamanho, forma, textura) | 3 agregados por conceito |

A OWL original só tinha 3 agregados (mean / erro padrão / worst); a riqueza semântica passa de `limited` a `rich`.

## Resultado (pipeline de produção, 2 seeds, split de teste de 171 amostras)

| | OWL original | OWL enriquecida |
|---|---|---|
| Features geradas / estáveis | 3 / 2 | 63 / 8 |
| Ganho OOF do MLP (seed 42) | +0,0035, IC95 [−0,007; +0,017] | +0,0024, IC95 [−0,014; +0,018] |
| Ganho OOF do MLP (seed 7) | +0,0091, IC95 [−0,004; +0,022] | +0,0085, IC95 [−0,005; +0,023] |
| Decisão (42 / 7) | ACCEPT_PARTIAL (fraca) / ACCEPT_PARTIAL (fraca) | REJECT_NO_INFORMATIONAL_GAIN / ACCEPT_PARTIAL (fraca) |
| Professor do TREPAN | MLP Original (evidência fraca < forte) | MLP Original (idem) |

**A OWL enriquecida não deu ganho estatisticamente significativo.** Todos os intervalos de confiança incluem zero.
Mais features não melhoraram o ganho; com seed 42 a enriquecida foi até rejeitada (o MLP já usa estas combinações).

TREPAN com capacidade ajustada (63 nós, purity_epsilon 0,01, 31 000 queries): 51–59 nós, 26–30 folhas; a ontologia
influencia 26–29 de 31 splits (impacto na decisão 16–26%), mas as diferenças Reloaded vs Original são de ~1 amostra
de teste (1/171 ≈ 0,006) e mudam de sinal entre seeds. **Não há evidência de ganho do Reloaded devido à ontologia.**

## Âmbito
Esta OWL e o gerador são um **artefacto de teste específico deste dataset** (`data/` e `scripts/`); a aplicação não os referencia
e continua agnóstica ao dataset: o processador deriva features de qualquer OWL que declare hierarquia, família/papel ou limites.

## Limitações
Duas seeds e um split; dataset com MLP perto do teto (~96–97% OOF) e n=569, onde um ganho < ~1,5 pp não é detetável.
Os resultados da OWL enriquecida usam conhecimento de domínio declarado, não ajustado aos dados.
