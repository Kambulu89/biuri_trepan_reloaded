# Análise estatística

Dados: `evidencia/baseline_por_unidade.csv` (20 unidades = 4 datasets × 5 sementes, 6 braços), gerado pelo harness do projeto e verificado de forma independente (700 valores recalculados, diferença máxima 0). Natureza: **DEV, exploratória**. Estes datasets já tinham sido observados pelo projeto; nada aqui é confirmatório.

Método: diferenças emparelhadas por unidade (mesma partição, mesmo MLP). Intervalo de confiança a 95 % por bootstrap percentil (20 000 reamostragens das 5 unidades de cada dataset). Efeito = diferença média / desvio-padrão das diferenças. Com 5 unidades por dataset os intervalos são largos e o bootstrap é pouco fiável; leio-os como ordem de grandeza.

## 1. Contrastes por dataset

| Contraste | Dataset | Diferença média | IC 95 % | Efeito (média/dp) | Unidades +/− |
|---|---|---|---|---|---|
| acc: Reloaded+OWL − Original | iris | -0,0053 | [-0,0368; +0,0263] | -0,12 | 3/2 |
| acc: Reloaded+OWL − Original | wine | -0,0267 | [-0,0578; +0,0044] | -0,67 | 1/3 |
| acc: Reloaded+OWL − Original | breast_cancer | +0,0028 | [-0,0098; +0,0126] | +0,19 | 3/1 |
| acc: Reloaded+OWL − Original | digits | +0,0009 | [-0,0156; +0,0178] | +0,04 | 2/3 |
| acc: Reloaded+OWL − C4,5 | iris | -0,0263 | [-0,0579; +0,0000] | -0,71 | 0/2 |
| acc: Reloaded+OWL − C4,5 | wine | -0,0267 | [-0,0844; +0,0222] | -0,39 | 2/3 |
| acc: Reloaded+OWL − C4,5 | breast_cancer | +0,0056 | [-0,0042; +0,0126] | +0,49 | 4/1 |
| acc: Reloaded+OWL − C4,5 | digits | -0,0293 | [-0,0436; -0,0102] | -1,35 | 1/4 |
| acc: Original − C4,5 | iris | -0,0211 | [-0,0263; -0,0105] | -1,79 | 0/4 |
| acc: Original − C4,5 | wine | -0,0000 | [-0,0444; +0,0444] | -0,00 | 3/2 |
| acc: Original − C4,5 | breast_cancer | +0,0028 | [-0,0112; +0,0168] | +0,15 | 2/2 |
| acc: Original − C4,5 | digits | -0,0302 | [-0,0516; -0,0102] | -1,11 | 0/5 |
| fid: Reloaded+OWL − Original | iris | -0,0053 | [-0,0316; +0,0211] | -0,15 | 2/2 |
| fid: Reloaded+OWL − Original | wine | -0,0311 | [-0,0667; +0,0044] | -0,68 | 1/3 |
| fid: Reloaded+OWL − Original | breast_cancer | -0,0112 | [-0,0238; +0,0042] | -0,64 | 1/3 |
| fid: Reloaded+OWL − Original | digits | -0,0000 | [-0,0164; +0,0182] | -0,00 | 3/2 |
| fid: Reloaded+OWL − C4,5 | iris | -0,0053 | [-0,0316; +0,0158] | -0,18 | 1/1 |
| fid: Reloaded+OWL − C4,5 | wine | -0,0267 | [-0,0711; +0,0178] | -0,46 | 2/3 |
| fid: Reloaded+OWL − C4,5 | breast_cancer | +0,0028 | [-0,0126; +0,0168] | +0,14 | 2/1 |
| fid: Reloaded+OWL − C4,5 | digits | -0,0204 | [-0,0351; -0,0058] | -1,03 | 1/4 |
| acc: Reloaded+OWL − permutada | iris | -0,0158 | [-0,0368; +0,0000] | -0,67 | 0/2 |
| acc: Reloaded+OWL − permutada | wine | -0,0311 | [-0,0711; -0,0044] | -0,68 | 0/3 |
| acc: Reloaded+OWL − permutada | breast_cancer | -0,0070 | [-0,0238; +0,0098] | -0,32 | 2/3 |
| acc: Reloaded+OWL − permutada | digits | -0,0067 | [-0,0200; +0,0058] | -0,40 | 2/3 |
| fid: Reloaded+OWL − permutada | iris | +0,0053 | [-0,0158; +0,0316] | +0,18 | 1/1 |
| fid: Reloaded+OWL − permutada | wine | -0,0356 | [-0,0756; -0,0089] | -0,82 | 0/4 |
| fid: Reloaded+OWL − permutada | breast_cancer | -0,0014 | [-0,0126; +0,0084] | -0,10 | 2/2 |
| fid: Reloaded+OWL − permutada | digits | -0,0080 | [-0,0231; +0,0049] | -0,44 | 1/4 |
| acc: MLP − C4,5 | iris | -0,0053 | [-0,0158; +0,0000] | -0,45 | 0/1 |
| acc: MLP − C4,5 | wine | +0,0400 | [+0,0133; +0,0711] | +1,10 | 4/0 |
| acc: MLP − C4,5 | breast_cancer | +0,0322 | [+0,0154; +0,0531] | +1,34 | 5/0 |
| acc: MLP − C4,5 | digits | +0,1067 | [+0,0924; +0,1213] | +5,93 | 5/0 |
| fid: Original − C4,5 | iris | +0,0000 | [-0,0368; +0,0474] | +0,00 | 2/3 |
| fid: Original − C4,5 | wine | +0,0044 | [-0,0311; +0,0356] | +0,10 | 2/1 |
| fid: Original − C4,5 | breast_cancer | +0,0140 | [-0,0014; +0,0308] | +0,69 | 3/1 |
| fid: Original − C4,5 | digits | -0,0204 | [-0,0493; +0,0076] | -0,60 | 1/4 |

Leitura: «acc» é accuracy face aos rótulos reais; «fid» é fidelidade ao MLP. «Unidades +/−» conta as unidades com diferença positiva e negativa (as restantes empatam).

## 2. Hierarquia conjunta

| Dataset | acc R>O | acc R>C4.5 | acc O>C4.5 | fid R>O | fid R>C4.5 | Conjunta |
|---|---|---|---|---|---|---|
| iris | 3/5 | 0/5 | 0/5 | 2/5 | 1/5 | 0/5 |
| wine | 1/5 | 2/5 | 3/5 | 1/5 | 2/5 | 0/5 |
| breast_cancer | 3/5 | 4/5 | 2/5 | 1/5 | 2/5 | 0/5 |
| digits | 2/5 | 1/5 | 0/5 | 3/5 | 1/5 | 0/5 |
| **Total** | 9/20 | 7/20 | 5/20 | 7/20 | 6/20 | **0/20** |

R = Reloaded + ontologia estrutural; O = Original. A hierarquia conjunta não se verificou em nenhuma das 20 unidades.

## 3. O que os dados sustentam

1. **Nenhum dos cinco contrastes da hipótese tem intervalo acima de zero em nenhum dataset.**
2. **Três contrastes têm intervalo abaixo de zero:** Original − C4.5 em accuracy no iris e no digits; Reloaded − C4.5 em accuracy e em fidelidade no digits.
3. **A ontologia real não supera a permutada.** Em accuracy a diferença média é negativa nos quatro datasets, com intervalo abaixo de zero no wine. Em fidelidade é negativa em três e o intervalo fica abaixo de zero no wine.
4. **O MLP supera o C4.5 em três datasets** (wine, breast_cancer, digits) e não o supera no iris. No iris a cadeia `Original > C4.5` em accuracy é inviável com este oráculo.
5. **O Reloaded sem ontologia é idêntico ao Original** nas 20 unidades (accuracy, fidelidade, nós). O contrato de equivalência verifica-se.
6. **Custo:** o Reloaded com ontologia usa 2,2 vezes mais amostras do oráculo (mediana) e 11,6 vezes mais tempo do que o Original, sem ganho.

## 4. Entre datasets

Sinal da diferença média por dataset (4 datasets):

| Contraste | Positivos | Negativos |
|---|---|---|
| acc: Reloaded+OWL − Original | 2 | 2 |
| acc: Reloaded+OWL − C4.5 | 1 | 3 |
| acc: Original − C4.5 | 1 | 3 |
| fid: Reloaded+OWL − Original | 0 | 4 (um deles −0,0000) |
| fid: Reloaded+OWL − C4.5 | 1 | 3 |

Com 4 datasets, um teste dos sinais bilateral não pode dar p abaixo de 0,125, mesmo com os quatro sinais iguais. Não reporto valores p: seriam decorativos.

## 5. Ontologia cientificamente enriquecida (Breast Cancer)

Execução separada com `scripts/run_benchmark.py`, ARFF com nomes UCI e a ontologia `breast_cancer_wisconsin_diagnostic_enriquecida_biuri.owl` (59 atributos derivados, cobertura 100 %), 5 sementes. Dados em `evidencia/bc_enriquecida/`.

| Contraste | Diferença média | IC 95 % | Unidades +/− |
|---|---|---|---|
| acc: Reloaded − Original | +0,0014 | [−0,0182; +0,0210] | 3/2 |
| acc: Reloaded − C4.5 | +0,0056 | [−0,0098; +0,0238] | 2/3 |
| acc: Original − C4.5 | +0,0042 | [−0,0140; +0,0224] | 2/2 |
| fid: Reloaded − Original | −0,0042 | [−0,0182; +0,0154] | 1/4 |
| fid: Reloaded − C4.5 | +0,0084 | [−0,0154; +0,0336] | 2/3 |
| acc: Reloaded − permutada | +0,0028 | [−0,0168; +0,0252] | 2/2 |
| fid: Reloaded − permutada | −0,0000 | [−0,0140; +0,0168] | 2/3 |

Todos os intervalos incluem zero. A ontologia enriquecida não se distingue da permutada.

## 6. Experiências exploratórias desta auditoria

Configuração fixa minha (sem o tuning estrutural do projeto), mesmo MLP e partições por unidade. Scripts e dados em `evidencia/exp_*.py|json`.

**6.1 Pré-seleção de atributos (P04).** Reescala equivalente das colunas: árvore diferente em 14 de 15 unidades com mais de 12 atributos. Usar todos os atributos em vez de 12: fidelidade média +0,010 no breast_cancer e +0,019 no digits (5 sementes), sem efeito consistente na accuracy.

**6.2 Capacidade.** Fidelidade média do Original (5 sementes) com 31 e 127 nós: iris 0,926 e 0,932; wine 0,929 e 0,929; breast_cancer 0,926 e 0,920; digits 0,739 e 0,826. Só no digits mais nós ajudam.

**6.3 Amostrador conjunto (`core/query_sampling.py`, desligado por omissão).** 5 sementes × 3 réplicas, 31 nós:

| Dataset | fid: conjunto − marginal | acc: conjunto − marginal | acc: Original (conjunto) − C4.5 |
|---|---|---|---|
| iris | +0,030 [+0,016; +0,047] | +0,023 [+0,002; +0,051] | −0,009 [−0,028; +0,007] |
| wine (todos os atributos) | +0,003 [−0,013; +0,016] | +0,009 [−0,006; +0,025] | +0,015 [+0,003; +0,034] |
| breast_cancer (todos os atributos) | +0,002 [−0,012; +0,015] | +0,010 [−0,001; +0,021] | +0,012 [+0,008; +0,019] |
| digits (todos os atributos) | +0,030 [+0,013; +0,045] | +0,036 [+0,022; +0,050] | −0,056 [−0,075; −0,034] |

Desvio-padrão entre réplicas do extrator, com tudo o resto fixo: 1,3 a 2,7 pontos em accuracy e 1,1 a 2,2 em fidelidade. Qualquer diferença desta ordem numa única execução não é interpretável.

## 7. Limites desta análise

- 5 unidades por dataset e 4 datasets: sem poder para confirmar efeitos pequenos.
- Uma única réplica do extrator por unidade no baseline oficial.
- A experiência 6.3 é exploratória e altera o baseline congelado; não pode ser somada ao baseline oficial.
- Coerência semântica e estabilidade entre execuções não foram medidas como métricas próprias (NÃO MEDIDA), para além do desvio entre réplicas em 6.3.
