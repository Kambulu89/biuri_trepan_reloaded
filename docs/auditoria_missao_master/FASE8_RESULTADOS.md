# Fase 8 — resultados: pipeline melhorado em dados reais

Análise segundo `FASE8_PREREGISTO.md` (commit `74eb6ff`, anterior à campanha). Nenhuma configuração, hipótese ou critério foi alterado durante a campanha. As fases 1 a 7 e os seus resultados não foram alterados. Dados em `evidencia/fase8/`.

Pipeline avaliado (congelado nas fases 5 e 7): MLP otimizado por seleção só no treino como oráculo, Original (B) e Reloaded O (R, atributos da ontologia sem mecanismos), amostrador conjunto, 3 réplicas, dois orçamentos (estrutura do tuning e máximo de 7 nós), controlo de ontologia permutada (Rq).

## 1. Execução

- 25 de 25 unidades com código de saída 0 (5 datasets × 5 sementes; 6 braços TREPAN × 3 réplicas, mais MLP e C4.5); nenhum braço saltado; nenhuma unidade excedeu o limite de 7200 s.
- Verificação independente das métricas a partir das previsões guardadas: 3425 valores, diferença máxima 0 (`fase8_verificacao_metricas.log`). 575 linhas brutas em `fase8_linhas_brutas.csv`.
- **MLP original** reajustado no treino de cada unidade (`fase8_mlp_original.csv`): reproduz as previsões do baseline da fase 1 em 20 de 20 unidades onde o baseline existe; o Adult não tem baseline (5 unidades sem comparação).
- Elegibilidade por validação cruzada aninhada só no treino: 25 de 25 unidades.

### Incidentes (sem efeito nos resultados, registados)

1. **Reinício do contentor** a meio da campanha, com 19 de 25 unidades concluídas (`fase8_supervisor.log`, marca de 2026-11-18T23:44:55Z). As 6 unidades em falta foram executadas **do início**, com o mesmo executor e a mesma configuração; nenhuma unidade concluída foi repetida ou substituída.
2. **Elegibilidade paralelizada** em 4 processos (por grupos de datasets), cada um a ler as mesmas unidades e a executar o mesmo procedimento da fase 7 (`fase8_eligibility.py`). Os 4 resultados foram juntos sem alteração (`fase8_elegibilidade.json`, `fase8_elegibilidade.log`). Mudou só o tempo de execução, não o procedimento.

## 2. Tratamento estatístico

Igual à fase 7: a unidade é a semente; as 3 réplicas promediam-se dentro da unidade; cada contraste usa 5 diferenças emparelhadas e um intervalo a 95 % por bootstrap percentil (20 000 reamostragens). «Sustentada» exige o limite inferior **estritamente** acima de 0 (um limite igual a 0,000 não conta). Sem correção para comparações múltiplas: leitura exploratória por dataset. Quatro dos cinco datasets já tinham sido observados; nada aqui é confirmatório.

## 3. MLP otimizado contra MLP original e contra o C4.5 (M1)

Diferenças médias no teste, intervalo a 95 %:

| Contraste (accuracy) | iris | wine | breast_cancer | digits | adult |
|---|---|---|---|---|---|
| MLP otimizado − MLP original | −0,011 [−0,021; +0,000] | +0,009 [+0,000; +0,018] | +0,011 [−0,006; +0,028] | **+0,023 [+0,020; +0,024]** | −0,007 [−0,019; +0,006] |
| MLP otimizado − C4.5 | −0,016 [−0,037; +0,000] | **+0,049 [+0,027; +0,071]** | **+0,043 [+0,034; +0,052]** | **+0,129 [+0,115; +0,143]** | **−0,016 [−0,021; −0,012]** |
| MLP original − C4.5 | −0,005 [−0,016; +0,000] | **+0,040 [+0,013; +0,071]** | **+0,032 [+0,015; +0,053]** | **+0,107 [+0,094; +0,121]** | −0,009 [−0,019; +0,001] |

Macro-F1 e balanced accuracy seguem a mesma direção (tabela completa em `fase8_stats.log`); a única divergência é o Adult, onde o MLP otimizado tem balanced accuracy +0,011 [−0,004; +0,033] acima do original apesar de menos accuracy (troca a favor da classe minoritária, não sustentada).

- A otimização do MLP **só melhora de forma sustentada no digits** (+2,3 pontos). No wine o intervalo toca 0; no breast_cancer inclui 0; no iris e no Adult a média é negativa.
- Ao contrário dos controlos físicos da fase 7, o MLP original destes datasets reais não estava a memorizar ruído de forma prejudicial: o ganho da seleção é pequeno.
- No **Adult o MLP otimizado fica abaixo do C4.5** (−1,6 pontos, sustentado).

### Elegibilidade (só treino, margem 0,02)

| Dataset | MLP otimizado − C4.5 (CV aninhada no treino) | Elegível |
|---|---|---|
| iris | +0,012 | não |
| wine | +0,082 | sim |
| breast_cancer | +0,044 | sim |
| digits | +0,130 | sim |
| adult | −0,022 | não |

A decisão de elegibilidade tomada só no treino é coerente com o teste: os três elegíveis têm MLP otimizado acima do C4.5 por mais de 0,02 no teste; no iris o sinal muda (+0,012 no treino, −0,016 no teste), mas nenhum dos dois passa a margem.

## 4. Original contra Reloaded e contributo da ontologia (mesmo MLP otimizado)

### 4.1 Estrutura escolhida pelo tuning

| Contraste | iris | wine | breast_cancer | digits | adult |
|---|---|---|---|---|---|
| H1 accuracy: R − B | −0,002 [−0,011; +0,009] | −0,004 [−0,018; +0,009] | −0,001 [−0,009; +0,005] | +0,001 [−0,005; +0,004] | −0,000 [−0,004; +0,004] |
| H2 accuracy: R − C4.5 | −0,004 [−0,025; +0,018] | **+0,030 [+0,006; +0,049]** | +0,009 [−0,002; +0,022] | **+0,019 [+0,003; +0,037]** | −0,005 [−0,017; +0,009] |
| H3 accuracy: B − C4.5 | −0,002 [−0,019; +0,018] | **+0,034 [+0,006; +0,058]** | +0,010 [−0,000; +0,022] | +0,019 [−0,000; +0,038] | −0,005 [−0,016; +0,008] |
| H4 fidelidade: R − B | −0,005 [−0,014; +0,004] | +0,002 [−0,006; +0,012] | −0,005 [−0,013; +0,003] | +0,001 [−0,005; +0,006] | −0,001 [−0,006; +0,003] |
| H5 fidelidade: R − C4.5 | −0,004 [−0,012; +0,004] | **+0,036 [+0,009; +0,053]** | +0,001 [−0,008; +0,009] | **+0,019 [+0,002; +0,038]** | **+0,011 [+0,005; +0,016]** |
| AT accuracy: R − Rq | **+0,009 [+0,002; +0,018]** | +0,000 [−0,033; +0,033] | −0,001 [−0,015; +0,013] | **+0,003 [+0,001; +0,005]** | +0,006 [−0,001; +0,013] |
| AT fidelidade: R − Rq | +0,005 [−0,004; +0,016] | −0,007 [−0,030; +0,018] | −0,001 [−0,006; +0,005] | **+0,004 [+0,003; +0,005]** | +0,006 [−0,002; +0,013] |

### 4.2 Máximo de 7 nós

| Contraste | iris | wine | breast_cancer | digits | adult |
|---|---|---|---|---|---|
| H1 accuracy: R7 − B7 | **+0,009 [+0,004; +0,014]** | **−0,019 [−0,031; −0,009]** | −0,006 [−0,018; +0,005] | **+0,004 [+0,000; +0,008]** ¹ | −0,000 [−0,007; +0,005] |
| H2 accuracy: R7 − C4.5 | −0,002 [−0,019; +0,018] | +0,018 [−0,004; +0,036] | −0,002 [−0,019; +0,018] | **−0,488 [−0,501; −0,474]** | −0,001 [−0,007; +0,004] |
| H3 accuracy: B7 − C4.5 | −0,011 [−0,030; +0,011] | **+0,037 [+0,012; +0,058]** | +0,004 [−0,006; +0,015] | **−0,492 [−0,505; −0,476]** | −0,001 [−0,009; +0,008] |
| H4 fidelidade: R7 − B7 | +0,002 [−0,016; +0,021] | **−0,010 [−0,018; −0,003]** | −0,009 [−0,022; +0,001] | +0,003 [−0,000; +0,007] | −0,005 [−0,015; +0,004] |
| H5 fidelidade: R7 − C4.5 | −0,009 [−0,023; +0,000] | **+0,027 [+0,002; +0,046]** | −0,012 [−0,025; +0,000] | **−0,488 [−0,499; −0,475]** | −0,005 [−0,018; +0,007] |
| AT accuracy: R7 − Rq7 | +0,009 [+0,000; +0,023] | −0,007 [−0,033; +0,016] | −0,007 [−0,018; +0,004] | −0,000 [−0,001; +0,000] | −0,002 [−0,008; +0,002] |
| AT fidelidade: R7 − Rq7 | +0,002 [−0,005; +0,011] | −0,006 [−0,028; +0,016] | −0,004 [−0,015; +0,007] | −0,000 [−0,001; +0,000] | −0,003 [−0,011; +0,005] |

¹ Limite inferior +0,0001: sustentado por margem mínima.

No digits, 7 nós dão no máximo 4 folhas para 10 classes: B7 e R7 ficam em ≈ 0,36 de accuracy. O C4.5 não tem limite de nós (169 em média), por isso H2, H3 e H5 com 7 nós comparam árvores de tamanhos muito diferentes, como nas fases 6 e 7.

### 4.3 Decisão (critérios da fase 6)

| Orçamento | Dataset | Elegível | Sustentada | Observada | Atribuível |
|---|---|---|---|---|---|
| estrutura do tuning | iris | não | não | não | não |
| estrutura do tuning | wine | sim | não | não | não |
| estrutura do tuning | breast_cancer | sim | não | não | não |
| estrutura do tuning | digits | sim | não | **sim** | não |
| estrutura do tuning | adult | não | não | não | não |
| máx. 7 nós | iris | não | não | não | não |
| máx. 7 nós | wine | sim | não | não | não |
| máx. 7 nós | breast_cancer | sim | não | não | não |
| máx. 7 nós | digits | sim | não | não | não |
| máx. 7 nós | adult | não | não | não | não |

Unidades com as cinco condições em conjunto (H1–H5): digits 2/5, wine 1/5, outros 0/5 com a estrutura do tuning; breast_cancer 1/5, outros 0/5 com 7 nós (`fase8_stats.log`).

## 5. Leitura

- **Veredicto principal (datasets elegíveis: wine, breast_cancer, digits):** a hierarquia `Reloaded > Original > C4.5` **não se sustenta em nenhum** dataset elegível, em nenhum orçamento. Só é **observada** (todas as médias positivas, intervalos a incluir 0) no digits com a estrutura do tuning, e por diferenças R − B de +0,001.
- **`Original > C4.5`** em accuracy sustenta-se no wine (ambos os orçamentos), fica no limite no digits e no breast_cancer com a estrutura do tuning, e falha no iris e no Adult. Onde o MLP otimizado supera o C4.5 por margem larga, o Original tende a superá-lo; onde não supera (iris, Adult), não.
- **`Reloaded > Original`** (o ganho ontológico direto) não se sustenta em nenhum dataset elegível com a estrutura do tuning. Com 7 nós o Reloaded fica **abaixo** do Original no wine (accuracy −1,9 e fidelidade −1,0 pontos, sustentado) e acima no iris (+0,9, não elegível) e no digits (+0,4, margem mínima, sem fidelidade).
- **Ontologia contra ontologia permutada:** o digits mostra R > Rq sustentado em accuracy e fidelidade com a estrutura do tuning (+0,3 e +0,4 pontos), mas sem R > B; o iris mostra R > Rq em accuracy, também sem R > B. Isto é: a ontologia real é melhor do que a permutada, mas não melhor do que não usar ontologia. Não é ganho atribuível pelo critério pré-registado.
- **Adult:** o MLP otimizado não supera o C4.5; não é elegível. O Reloaded tem fidelidade sustentadamente superior ao C4.5 (+1,1 pontos) mas não accuracy: é uma diferença de **fidelidade, não de accuracy**, e não satisfaz a hierarquia.
- **Ontologias v2 com relações declaradas (iris, wine, breast_cancer):** sob o MLP otimizado, as relações declaradas continuam sem produzir ganho sobre o Original, como na fase 3. Nos dados reais, ao contrário dos controlos físicos da fase 7, não há uma relação declarada que defina a classe.

## 6. Limitações

- Quatro de cinco datasets já tinham sido observados; só o Adult é novo, e numa amostra fixa de 2000 de 48 842 linhas.
- n = 5 unidades por dataset; intervalos aproximados; muitas comparações sem correção.
- Com 5 datasets não há teste entre datasets com poder; a conclusão é por dataset.
- Com 7 nós o C4.5 não é restringido ao mesmo tamanho; H2, H3 e H5 com 7 nós não comparam complexidade igual.
- A ontologia do digits é a v1 (só agrupamentos); a do Adult é um agrupamento temático com uma única relação declarada (ganho − perda de capital). Uma ontologia mais rica poderia dar outro resultado; não foi testada.
- O EFSR com validação independente continua apenas proposto; não foi pré-registado nem testado.

## 7. Conclusões

1. Com o pipeline melhorado das fases 5 e 7, **a hipótese não se verifica em nenhum dos cinco datasets reais**, nem nos três elegíveis. O resultado mais próximo é o digits (hierarquia observada, não sustentada, 2/5 unidades).
2. A otimização do MLP **só melhora sustentadamente o digits**; nos restantes reais o efeito é pequeno ou negativo. O ganho de 5–6 pontos da fase 7 era específico dos controlos com ruído de rótulos.
3. `Original > C4.5` acompanha a vantagem do MLP sobre o C4.5: aparece onde ela é grande (wine, digits) e desaparece onde é nula ou negativa (iris, Adult).
4. Não há ganho de accuracy nem de fidelidade atribuível à ontologia em dados reais. A ontologia real supera a permutada em dois datasets, mas não supera o Original.
5. Juntando as fases 6 a 8: o mecanismo de atributos ontológicos produz ganho atribuível quando a ontologia declara uma relação verdadeira que determina a classe (controlos físicos); nas ontologias reais disponíveis, que descrevem sobretudo agrupamentos e relações não determinantes, não produz.

Não avancei para nenhuma fase seguinte.
