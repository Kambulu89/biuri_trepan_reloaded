# Fase 11 — resultados: complexidade igual (P07) e consultas efetivas dos braços com mecanismos (P02)

Análise segundo `FASE11_PREREGISTO.md` (commit `1570f39`, anterior à campanha). Nenhuma hipótese, critério ou braço foi alterado. O núcleo não foi alterado. As fases 1 a 10 não foram alteradas. Dados em `evidencia/fase11/`.

## 1. Execução

- **Pipeline principal** (MLP selecionado só no treino): 45 de 45 unidades com código de saída 0. **Execução com o MLP por omissão** (pipeline da fase 6): 20 de 20 unidades. Nenhum braço descartado.
- Recalculei as métricas a partir das previsões guardadas: 9675 e 2860 valores, com diferença máxima de 0.
- **Integridade:** 910 de 910 previsões (MLP, C4.5, B, B7, R e R7) iguais às das fases 8 e 9 (pipeline principal) e da fase 6 (MLP por omissão).
- **Construção:** em todas as unidades, Bc e Rc ficam com no máximo N nós, e c45_d2 com no máximo 7.

## 2. Complexidade igual (P07)

### 2.1 TREPAN limitado ao tamanho do C4.5

N = min(nós do C4.5 canónico da unidade, teto do tuning). Diferenças em pontos percentuais, com IC a 95 %.

| Contraste (pp) | iris | wine | breast_cancer | digits | adult | german_credit | phys_bmi | phys_kinetic | phys_power |
|---|---|---|---|---|---|---|---|---|---|
| H1c accuracy: Rc − Bc | +0,53 [-0,70; +1,58] | -1,04 [-2,07; +0,00] | +0,56 [+0,28; +0,79] | +0,10 [-0,49; +0,49] | -0,04 [-0,37; +0,35] | +0,21 [-1,09; +1,41] | +1,47 [+0,13; +2,80] | +2,00 [+1,33; +2,73] | +2,00 [+0,40; +3,27] |
| H2c accuracy: Rc − C4.5 | -0,18 [-1,75; +1,40] | +2,22 [+0,29; +4,00] | +0,98 [-0,05; +2,10] | +1,76 [+0,18; +3,53] | -0,51 [-1,69; +0,85] | +1,25 [-1,60; +3,89] | +2,47 [+0,47; +4,40] | +3,67 [+1,93; +5,67] | +5,00 [+3,27; +6,87] |
| H3c accuracy: Bc − C4.5 | -0,70 [-2,98; +2,11] | +3,26 [+0,44; +5,93] | +0,42 [-0,65; +1,49] | +1,66 [-0,21; +3,56] | -0,47 [-1,57; +0,81] | +1,04 [-0,93; +3,01] | +1,00 [-0,53; +2,87] | +1,67 [+0,13; +3,60] | +3,00 [+1,00; +4,80] |
| H4c fidelidade: Rc − Bc | +0,53 [-0,53; +2,11] | -0,15 [-0,89; +0,59] | +0,28 [+0,00; +0,47] | +0,12 [-0,44; +0,65] | -0,12 [-0,63; +0,32] | -0,37 [-1,20; +0,32] | +0,80 [+0,13; +1,47] | +2,40 [+1,33; +4,13] | +1,73 [+1,00; +2,27] |
| H5c fidelidade: Rc − C4.5 | -0,53 [-0,88; -0,18] | +3,11 [+0,44; +5,19] | +0,05 [-0,61; +0,51] | +1,82 [+0,07; +3,78] | +1,07 [+0,51; +1,61] | -2,48 [-5,18; +0,21] | +2,87 [+1,73; +3,73] | +5,13 [+3,27; +7,00] | +7,40 [+4,87; +9,93] |

### 2.2 Sete nós: B7 e R7 contra o C4.5 de profundidade 2

| Contraste (pp) | iris | wine | breast_cancer | digits | adult | german_credit | phys_bmi | phys_kinetic | phys_power |
|---|---|---|---|---|---|---|---|---|---|
| H1 accuracy: R7 − B7 | +0,88 [+0,35; +1,40] | -1,93 [-2,96; -0,89] | -0,56 [-1,82; +0,51] | +0,39 [+0,01; +0,79] | -0,04 [-0,68; +0,55] | +1,17 [+0,61; +1,95] | +7,67 [+4,87; +9,60] | +7,73 [+5,93; +9,53] | +7,00 [+4,80; +9,40] |
| H2d accuracy: R7 − C4.5d2 | +0,35 [-1,75; +2,46] | +4,89 [+1,33; +8,89] | +0,79 [-0,09; +2,19] | +5,99 [+4,83; +7,64] | +2,61 [+1,84; +3,39] | +2,59 [+1,57; +3,60] | +14,27 [+12,07; +16,20] | +9,13 [+6,20; +11,47] | +13,53 [+8,20; +20,87] |
| H3d accuracy: B7 − C4.5d2 | -0,53 [-2,81; +1,75] | +6,81 [+2,67; +10,96] | +1,35 [+0,42; +2,28] | +5,60 [+4,52; +6,96] | +2,65 [+1,87; +3,52] | +1,41 [+0,11; +2,72] | +6,60 [+5,40; +7,73] | +1,40 [-0,53; +2,80] | +6,53 [+0,59; +14,67] |
| H4 fidelidade: R7 − B7 | +0,18 [-1,58; +2,11] | -1,04 [-1,78; -0,30] | -0,93 [-2,24; +0,09] | +0,33 [-0,01; +0,70] | -0,47 [-1,53; +0,44] | +0,32 [-0,32; +1,12] | +4,60 [+2,33; +7,00] | +6,40 [+5,60; +7,20] | +8,47 [+5,67; +12,47] |
| H5d fidelidade: R7 − C4.5d2 | +1,75 [-1,23; +4,39] | +5,78 [+1,63; +9,93] | +0,05 [-1,35; +1,26] | +5,91 [+4,92; +7,44] | +3,95 [+2,87; +5,04] | +4,19 [+0,93; +6,96] | +11,87 [+9,13; +15,13] | +8,20 [+5,67; +10,27] | +17,27 [+10,33; +24,13] |

### 2.3 Número de nós (médias)

| Dataset | Nós C4.5 | N (limite de Bc/Rc) | Nós Bc | Nós Rc | Nós B | Nós R | Nós c45_d2 | Nós B7 | Nós R7 |
|---|---|---|---|---|---|---|---|---|---|
| iris | 8,6 | 8,6 | 7,4 | 7,3 | 33,8 | 33,8 | 5,0 | 5,7 | 5,4 |
| wine | 9,8 | 9,8 | 9,5 | 9,5 | 23,5 | 18,5 | 5,8 | 6,9 | 7,0 |
| breast_cancer | 17,0 | 17,0 | 16,5 | 15,1 | 34,3 | 35,5 | 6,2 | 6,7 | 6,7 |
| digits | 169,4 | 146,6 | 146,3 | 145,9 | 177,1 | 176,6 | 7,0 | 7,0 | 7,0 |
| adult | 135,0 | 56,6 | 47,3 | 47,5 | 47,3 | 47,5 | 3,8 | 5,4 | 4,7 |
| german_credit | 172,2 | 56,6 | 50,7 | 50,6 | 50,7 | 50,6 | 3,4 | 5,9 | 6,3 |
| phys_bmi | 23,4 | 23,4 | 22,1 | 5,7 | 28,7 | 8,3 | 3,0 | 6,7 | 3,5 |
| phys_kinetic | 21,0 | 21,0 | 20,3 | 15,1 | 41,9 | 35,8 | 5,0 | 6,3 | 3,9 |
| phys_power | 23,0 | 23,0 | 21,5 | 15,3 | 28,9 | 18,5 | 5,0 | 6,6 | 4,6 |

No adult e no german_credit, N é o teto do tuning (56,6), bem abaixo do C4.5 (135 e 172 nós). Aí o TREPAN já era mais pequeno do que o C4.5, e Bc e Rc são iguais a B e R. Com 7 nós o limite é o mesmo para todos, mas o C4.5 de profundidade 2 fica com menos nós do que B7 e R7 (3,0 a 7,0 contra 3,5 a 7,0): **o limite é igual, o tamanho realizado não**.

### 2.4 Decisão (definições da fase 6; sem controlo de OWL permutada nesta fase)

| Orçamento | Dataset | Elegível | Hierarquia sustentada | Hierarquia observada | Original > C4.5 sustentado | Reloaded > C4.5 em acc. e fid. sustentado |
|---|---|---|---|---|---|---|
| tamanho do C4.5 | iris | não | não | não | não | não |
| tamanho do C4.5 | wine | sim | não | não | sim | sim |
| tamanho do C4.5 | breast_cancer | sim | não | sim | não | não |
| tamanho do C4.5 | digits | sim | não | sim | não | sim |
| tamanho do C4.5 | adult | não | não | não | não | não |
| tamanho do C4.5 | german_credit | sim | não | não | não | não |
| tamanho do C4.5 | phys_bmi | sim | não | sim | não | sim |
| tamanho do C4.5 | phys_kinetic | não | sim | sim | sim | sim |
| tamanho do C4.5 | phys_power | sim | sim | sim | sim | sim |
| 7 nós | iris | não | não | não | não | não |
| 7 nós | wine | sim | não | não | sim | sim |
| 7 nós | breast_cancer | sim | não | não | sim | não |
| 7 nós | digits | sim | não | sim | sim | sim |
| 7 nós | adult | não | não | não | sim | sim |
| 7 nós | german_credit | sim | não | sim | sim | sim |
| 7 nós | phys_bmi | sim | sim | sim | sim | sim |
| 7 nós | phys_kinetic | não | não | sim | não | sim |
| 7 nós | phys_power | sim | sim | sim | sim | sim |

## 3. Consultas efetivas (P02)

Unidades em que o braço recebeu no máximo 1,10 vezes os pontos novos do Original e nenhum ponto novo só por probabilidade (tabela completa em `fase11_tabelas.txt`):

| Execução | Braço | Unidades que cumprem | Exceções (dataset: unidades) |
|---|---|---|---|
| MLP por omissão (fase 6) | R | 19/20 | phys_bmi: 1 (razão 1,16) |
| MLP por omissão (fase 6) | R7 | 20/20 | — |
| MLP selecionado | R | 42/45 | iris: 1; wine: 2 (as da fase 10) |
| MLP selecionado | R7 | 45/45 | — |
| MLP selecionado | E (EFSR original) | 42/45 | iris: 1; wine: 2 |
| MLP selecionado | D (Reloaded completo) | 40/45 | iris: 1; wine: 2; breast_cancer: 1; phys_kinetic: 1 |
| MLP selecionado | A (consultas ativas) | 41/45 | wine: 3; phys_power: 1 |
| MLP selecionado | V (EFSR com lote novo) | 8/45 | o lote de validação é, por desenho, informação nova: 1,35 a 2,47 vezes os pontos novos nos reais |

- **Em todas as 65 unidades e todos os braços, 0 pontos novos chegaram só por probabilidade.** As linhas a mais contadas são sempre probabilidades de pontos conhecidos.
- **MLP por omissão:** Rnp ≡ R em 120 de 120 árvores. A conclusão da fase 10 vale também para as comparações da fase 6.
- **E, D e A usam essas probabilidades nas decisões** (EFSR, consultas ativas). Para eles, «o mesmo número de pontos novos» não significa a mesma informação: recebem também a confiança do MLP sobre pontos conhecidos, que o Original não usa.

## 4. Leitura

1. **Com complexidade igual, `Original > C4.5` sustenta-se em mais datasets do que nas fases 6 a 9.**
   - A 7 nós, contra o C4.5 de profundidade 2, sustenta-se em 7 de 9: wine, breast_cancer, digits, adult, german_credit, phys_bmi e phys_power. Nas fases 6 a 9, o C4.5 sem limite ganhava ao B7 por grande margem no digits (−49 pp).
   - Ao tamanho do C4.5, sustenta-se só em wine, phys_kinetic e phys_power.
2. **`Reloaded > C4.5` em accuracy e fidelidade** sustenta-se em 7 de 9 a 7 nós e em 5 de 9 ao tamanho do C4.5 (wine, digits, phys_bmi, phys_kinetic, phys_power).
3. **A hierarquia completa** (Reloaded > Original > C4.5 nas duas métricas) **sustenta-se em dois datasets elegíveis, ambos controlos positivos sintéticos**: phys_bmi e phys_power a 7 nós, e phys_power ao tamanho do C4.5. Também se sustenta no phys_kinetic, que não é elegível, ao tamanho do C4.5. Em dados reais continua por sustentar. O obstáculo é `Reloaded > Original` (H1 e H4), não o C4.5:
   - a hierarquia é observada (todas as médias positivas) no breast_cancer e no digits ao tamanho do C4.5, e no digits e no german_credit a 7 nós;
   - em dados reais, H1 (accuracy, Reloaded − Original) tem intervalo acima de 0 no iris (+0,88 pp), no digits (+0,39 pp) e no german_credit (+1,17 pp) a 7 nós, e no breast_cancer (+0,56 pp) ao tamanho do C4.5. H4 (fidelidade) não está sustentado em nenhum dataset real: o melhor é o breast_cancer, +0,28 pp com limite inferior 0,00.
4. **Consultas:** para R e R7 a ressalva fica resolvida também com o MLP por omissão (fase 6). Para E, D e A os pontos novos são iguais na grande maioria das unidades, mas as probabilidades sobre pontos conhecidos entram nas decisões: a ressalva muda de «mais consultas» para «informação de confiança do MLP que o Original não usa».

## 5. Limitações

- A 7 nós, o limite é igual mas o C4.5 de profundidade 2 é em média mais pequeno do que B7 e R7. Uma comparação com o mesmo número de nós realizado exigiria um C4.5 com limite exato de nós, que não existe na implementação canónica.
- Ao tamanho do C4.5, N depende do C4.5 de cada unidade (ajustado só no treino).
- Não há controlo de OWL permutada nesta fase: «atribuível» não se aplica.
- n = 5 sementes, sem correção para comparações múltiplas; leitura exploratória.

Não avancei para nenhuma fase seguinte.
