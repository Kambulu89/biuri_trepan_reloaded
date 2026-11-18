# Fase 7 — auditoria e otimização do MLP: diagnóstico e pré-registo

Escrito e registado em commit **antes** da campanha de avaliação. O diagnóstico da secção 1 usou **só os dados de treino** das unidades da fase 6 (o conjunto de teste não foi lido; só os índices de teste, para obter o complemento). Os resultados irão para `FASE7_RESULTADOS.md`. As fases 1 a 6 e os seus resultados não são alterados.

## 1. Diagnóstico

### 1.1 Implementação auditada

O benchmark treina o MLP com `core.benchmark.runner.fit_mlp(..., trials=0)`, ou seja, com o perfil adaptativo de `core/mlp_factory.py`, sem pesquisa de hiperparâmetros:

- **Pipeline:** `StandardScaler` + `MLPClassifier`, ajustado só no treino. O escalonamento está dentro do pipeline, por isso não há fuga do teste.
- **Regra para menos de 500 linhas:** lbfgs, sem early stopping, `tol = 1e-5`, até 2000 iterações, ReLU, α = 0,001; camadas (h1, h1/2) com h1 = clip(8·√d, 8, 128).
- **Datasets físicos** (300 linhas, 5 atributos): camadas (18, 9), α = 0,001, lbfgs.
- **German Credit** (750 linhas, 61 atributos): adam com early stopping, camadas (62, 31).
- **Partições:** o MLP e o C4.5 são treinados nas mesmas linhas de treino e avaliados nas mesmas linhas de teste (verificado em todas as fases). Comparação justa.
- **Seleção de hiperparâmetros:** nenhuma no benchmark (`mlp_trials = 0`).

### 1.2 Medições (validação cruzada de 5 dobras só no treino, 5 unidades por dataset)

`evidencia/fase7/diag.csv`, script `evidencia/fase7/diag.py`.

| Dataset | Variante | Accuracy CV | Accuracy de treino | Convergiu |
|---|---|---|---|---|
| phys_bmi | perfil por omissão | 0,894 | 0,995 | 100 % |
| phys_bmi | α = 1 | 0,922 | 0,982 | 92 % |
| phys_bmi | adam + early stopping | 0,800 | 0,818 | 0 % |
| phys_bmi | C4.5 | 0,889 | 0,967 | — |
| phys_kinetic | perfil por omissão | 0,871 | 0,998 | 100 % |
| phys_kinetic | α = 1 | 0,895 | 0,982 | 88 % |
| phys_kinetic | C4.5 | 0,892 | 0,965 | — |
| phys_power | perfil por omissão | 0,845 | 0,997 | 100 % |
| phys_power | α = 1 | 0,883 | 0,979 | 88 % |
| phys_power | C4.5 | 0,873 | 0,949 | — |
| german_credit | perfil por omissão | 0,710 | 0,770 | 0 % (early stopping) |
| german_credit | adam + early stopping (val. 20 %) | 0,728 | 0,832 | 0 % |
| german_credit | C4.5 | 0,699 | 0,915 | — |

### 1.3 Causa

**Sobreajuste ao ruído dos rótulos, por regularização insuficiente.** Os datasets físicos têm 5 % de rótulos trocados, por isso o teto honesto de accuracy está perto de 95 %. O perfil por omissão chega a 99,5–100 % de accuracy de treino, ou seja, memoriza os rótulos errados: lbfgs até à convergência, sem early stopping, com α = 0,001 em 240 linhas. Com α = 1, a accuracy em validação cruzada sobe 2,4 a 3,8 pontos e passa a superar a do C4.5 nos três.

Descartados como causa: fuga de informação (não há), escalonamento (está no pipeline), não convergência (o perfil por omissão converge em 100 % das dobras), divisões diferentes (são as mesmas). O early stopping com adam também não serve: pára demasiado cedo e o modelo fica sub-ajustado.

No German Credit o perfil por omissão não sobreajusta (77 % de treino) e já supera ligeiramente o C4.5; a margem é pequena.

## 2. Hipóteses

| # | Hipótese |
|---|---|
| M1 | O MLP otimizado tem accuracy de teste superior à do MLP original, por dataset |
| M2 | Com o MLP otimizado, o MLP supera o C4.5 em mais de 2 pontos na validação cruzada aninhada só no treino (elegibilidade) |
| H1–H5 | Com o MLP otimizado como oráculo, a hierarquia da fase 6, para os dois orçamentos |
| AT | Reloaded > Reloaded com ontologia permutada, em accuracy e fidelidade |

## 3. Espaço de pesquisa e orçamento (congelados)

`validation/benchmark/mlp_selection_fase7.py`:

- **16 candidatos:** o perfil por omissão, exatamente o MLP original, e 15 configurações lbfgs (ReLU, 2000 iterações, `tol = 1e-5`): camadas {(16,), (64,), (32, 16)} × α {0,001; 0,01; 0,1; 1; 10}.
- **Seleção:** validação cruzada estratificada de 5 dobras **só no treino** da unidade, semente da unidade, critério balanced accuracy média. Empate (diferença ≤ 0,002): fica o α maior, depois o menor número de parâmetros.
- **Modelo final:** reajustado em todo o treino. Mesmo procedimento em todos os datasets.
- O espaço foi definido depois do diagnóstico da secção 1, que usou só dados de treino.

## 4. Campanha

- Executor `validation/benchmark/run_fase7.py`: o runner da fase 6 com o MLP substituído pelo MLP otimizado. É uma campanha nova, em diretório próprio; não substitui a fase 6.
- Datasets: phys_bmi, phys_kinetic, phys_power, german_credit. Sementes 42, 7, 123, 2024, 11; mesmas partições da fase 6.
- Braços: MLP otimizado (oráculo), C4.5, B e B7 (Original), R e R7 (Reloaded O congelado na fase 5), Rq e Rq7 (permutada); 3 réplicas. A estrutura das árvores é escolhida de novo pelo tuning do projeto, porque o oráculo muda.
- MLP original: as previsões da fase 6 (mesmas partições), para M1.

## 5. Métricas e estatística

- Accuracy, macro-F1 e balanced accuracy face aos rótulos reais; fidelidade ao oráculo da própria campanha.
- A fidelidade ao MLP otimizado não é comparável com a fidelidade ao MLP original: oráculos diferentes. Não compararei fidelidades entre campanhas.
- Medida por unidade: média das 3 réplicas. Intervalo a 95 % por bootstrap percentil (20 000 reamostragens das 5 unidades).
- **M1 sustentada:** intervalo de MLP otimizado − MLP original acima de zero em accuracy.
- **M2:** elegibilidade por validação cruzada aninhada no treino (5 dobras externas; em cada uma, a seleção da secção 3 com 5 dobras internas), margem 0,02; script `validation/benchmark/fase7_eligibility.py`.
- **Hierarquia sustentada, observada e atribuível:** definições da fase 6.
- Resultado negativo ou nulo é reportado como tal. Não altero o espaço, os datasets, as sementes nem os braços depois de ver resultados.

## 6. EFSR — análise separada (sem execução nesta fase)

O gate atual do EFSR (`_evaluate_error_focused_intervention`) aprova uma divisão proposta pela semântica se:

1. melhorar a fidelidade **na amostra de decisão do nó**, a mesma usada para escolher a divisão estatística concorrente; e
2. melhorar a fidelidade nas **linhas reais de treino que chegam ao nó**, que também fazem parte dessa amostra.

Há ainda uma tolerância de 1,5 pontos que aceita perdas quando a coerência semântica sobe. Os dois gates são, portanto, medidos dentro da amostra: favorecem propostas que se ajustam a essa amostra e não se generalizam, o que explica a perda medida na fase 5.

**Variante proposta (não implementada, não testada):** antes de comparar, sortear um lote **novo** de consultas ao oráculo na região do nó, com o mesmo amostrador e as mesmas restrições do caminho, não usado na escolha de nenhuma das duas divisões. A proposta semântica só é aceite se a sua fidelidade nesse lote superar a da divisão estatística com um teste unilateral emparelhado (McNemar, α = 0,05), sem tolerância. As consultas extra são contabilizadas no orçamento. Integrá-la exige um pré-registo próprio e avaliação em datasets diferentes dos usados para a desenhar.
