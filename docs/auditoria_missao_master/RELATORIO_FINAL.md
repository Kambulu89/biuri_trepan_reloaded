# Relatório final — auditoria do BIURI / TREPAN Reloaded

Ramo `auditoria/missao-master`, sobre o commit `df045ee`. Todos os números vêm de execuções feitas nesta auditoria; os registos estão em `evidencia/`.

## 1. Resposta à hipótese

**H1 (`TREPAN Reloaded > TREPAN Original > C4.5`) não foi confirmada.** A hierarquia conjunta (três condições de accuracy e duas de fidelidade) verificou-se em **0 de 20 unidades** (4 datasets × 5 sementes). Nenhum dos cinco contrastes tem intervalo de confiança acima de zero em nenhum dataset.

| Condição | Unidades em que se verificou |
|---|---|
| Accuracy: Reloaded + OWL > Original | 9 de 20 |
| Accuracy: Reloaded + OWL > C4.5 | 7 de 20 |
| Accuracy: Original > C4.5 | 5 de 20 |
| Fidelidade: Reloaded + OWL > Original | 7 de 20 |
| Fidelidade: Reloaded + OWL > C4.5 | 6 de 20 |
| As cinco em conjunto | 0 de 20 |

Onde falhou, por dataset (diferenças médias e intervalos em `ANALISE_ESTATISTICA.md`):

- **iris:** o MLP não supera o C4.5, por isso `Original > C4.5` é inviável; o Original fica 2,1 pontos abaixo.
- **wine:** Original e C4.5 empatam em média; o Reloaded com ontologia fica 2,7 pontos abaixo do Original.
- **breast_cancer:** as três árvores ficam a menos de 1 ponto umas das outras, todas 3 pontos abaixo do MLP.
- **digits:** Original e Reloaded ficam 3 pontos abaixo do C4.5.

A ontologia real não supera a ontologia permutada com intervalo acima de zero em nenhum dataset; em accuracy a diferença média é negativa nos quatro, e no wine o intervalo fica abaixo de zero. Com a única ontologia cientificamente enriquecida (Breast Cancer) o resultado é o mesmo: todos os intervalos incluem zero.

## 2. Matriz de comparação

Médias de 5 sementes. «Tempo» é o tempo do braço em segundos (para o MLP é só a avaliação; o treino é registado à parte pelo harness). Valores completos, com desvios e mais métricas, em `RESULTADOS_COMPARATIVOS.csv`.

| Dataset | Algoritmo | Accuracy | Macro F1 | Fidelidade MLP | Nº regras | Profundidade | Tempo (s) |
|---|---|---|---|---|---|---|---|
| iris | MLP | 0,953 | 0,953 | N/A | N/A | N/A | 0,009 |
| iris | C4.5 (rótulos reais) | 0,958 | 0,958 | 0,963 | 4,800 | 3,800 | 0,020 |
| iris | TREPAN Original | 0,937 | 0,937 | 0,963 | 18,400 | 8,400 | 0,265 |
| iris | TREPAN Reloaded sem ontologia | 0,937 | 0,937 | 0,963 | 18,400 | 8,400 | 0,246 |
| iris | TREPAN Reloaded + ontologia estrutural | 0,932 | 0,931 | 0,958 | 18,600 | 7,800 | 2,925 |
| iris | TREPAN Reloaded + ontologia permutada (controlo) | 0,947 | 0,948 | 0,953 | 17,800 | 9,000 | 2,546 |
| iris | TREPAN Reloaded + ontologia cientificamente enriquecida | NÃO MEDIDA | NÃO MEDIDA | NÃO MEDIDA | NÃO MEDIDA | NÃO MEDIDA | NÃO MEDIDA |
| iris | C4.5 treinado com rótulos do oráculo (variante separada) | 0,958 | NÃO MEDIDA | 0,963 | 5,000 | 3,800 | NÃO MEDIDA |
| wine | MLP | 0,969 | 0,969 | N/A | N/A | N/A | 0,012 |
| wine | C4.5 (rótulos reais) | 0,929 | 0,929 | 0,916 | 5,400 | 3,600 | 0,096 |
| wine | TREPAN Original | 0,929 | 0,929 | 0,920 | 19,600 | 9,200 | 1,017 |
| wine | TREPAN Reloaded sem ontologia | 0,929 | 0,929 | 0,920 | 19,600 | 9,200 | 1,045 |
| wine | TREPAN Reloaded + ontologia estrutural | 0,902 | 0,904 | 0,889 | 21,000 | 9,600 | 19,625 |
| wine | TREPAN Reloaded + ontologia permutada (controlo) | 0,933 | 0,935 | 0,924 | 21,600 | 10,000 | 20,668 |
| wine | TREPAN Reloaded + ontologia cientificamente enriquecida | NÃO MEDIDA | NÃO MEDIDA | NÃO MEDIDA | NÃO MEDIDA | NÃO MEDIDA | NÃO MEDIDA |
| wine | C4.5 treinado com rótulos do oráculo (variante separada) | 0,929 | NÃO MEDIDA | 0,916 | 5,400 | 3,600 | NÃO MEDIDA |
| breast_cancer | MLP | 0,962 | 0,960 | N/A | N/A | N/A | 0,012 |
| breast_cancer | C4.5 (rótulos reais) | 0,930 | 0,924 | 0,923 | 9,000 | 5,600 | 0,553 |
| breast_cancer | TREPAN Original | 0,933 | 0,928 | 0,937 | 16,600 | 7,000 | 2,856 |
| breast_cancer | TREPAN Reloaded sem ontologia | 0,933 | 0,928 | 0,937 | 16,600 | 7,000 | 2,882 |
| breast_cancer | TREPAN Reloaded + ontologia estrutural | 0,936 | 0,931 | 0,926 | 15,800 | 7,800 | 29,732 |
| breast_cancer | TREPAN Reloaded + ontologia permutada (controlo) | 0,943 | 0,939 | 0,927 | 15,600 | 8,000 | 18,983 |
| breast_cancer | TREPAN Reloaded + ontologia cientificamente enriquecida | 0,936 | 0,931 | 0,931 | 14,000 | 7,200 | 14,926 |
| breast_cancer | C4.5 treinado com rótulos do oráculo (variante separada) | 0,930 | NÃO MEDIDA | 0,923 | 9,000 | 5,600 | NÃO MEDIDA |
| digits | MLP | 0,960 | 0,960 | N/A | N/A | N/A | 0,011 |
| digits | C4.5 (rótulos reais) | 0,853 | 0,853 | 0,843 | 85,200 | 15,200 | 2,486 |
| digits | TREPAN Original | 0,823 | 0,821 | 0,823 | 116,400 | 12,400 | 14,089 |
| digits | TREPAN Reloaded sem ontologia | 0,823 | 0,821 | 0,823 | 116,400 | 12,400 | 13,688 |
| digits | TREPAN Reloaded + ontologia estrutural | 0,824 | 0,822 | 0,823 | 118,200 | 12,200 | 153,758 |
| digits | TREPAN Reloaded + ontologia permutada (controlo) | 0,830 | 0,830 | 0,831 | 117,000 | 12,000 | 144,795 |
| digits | TREPAN Reloaded + ontologia cientificamente enriquecida | NÃO MEDIDA | NÃO MEDIDA | NÃO MEDIDA | NÃO MEDIDA | NÃO MEDIDA | NÃO MEDIDA |
| digits | C4.5 treinado com rótulos do oráculo (variante separada) | 0,850 | NÃO MEDIDA | 0,843 | 84,800 | 16,000 | NÃO MEDIDA |

Notas:
- «Reloaded sem ontologia» é idêntico ao Original nas 20 unidades, como o contrato de equivalência exige.
- A linha da ontologia enriquecida vem de uma execução separada (partições e MLP próprios); compara-se com as linhas dessa execução, em `evidencia/bc_enriquecida/`, não diretamente com as restantes.
- A variante «C4.5 treinado com rótulos do oráculo» não é um baseline da hipótese. Dá quase o mesmo que o C4.5 normal porque o MLP reproduz os rótulos de treino.

## 3. Porque falha

1. **O TREPAN Original não imita o MLP melhor do que uma árvore treinada com rótulos reais.** A fidelidade do Original e a do C4.5 ao MLP diferem no máximo 2 pontos nos quatro datasets, e em três deles (wine, breast_cancer, digits) o MLP fica 3 a 14 pontos acima das árvores em accuracy. Sem fidelidade alta, a vantagem do MLP não chega à árvore.
2. **A informação nova que o TREPAN retira do MLP vem das consultas sintéticas, e o amostrador marginal gera-as fora da distribuição dos dados.** Trocar pelo amostrador conjunto que já existe no projeto melhorou o Original nos quatro datasets (secção 5).
3. **O Original só avalia 12 atributos por nó, escolhidos pela variância bruta.** O resultado depende das unidades de medida.
4. **Os mecanismos semânticos alteram decisões sem trazer informação.** As ontologias de benchmark só agrupam colunas; o peso dado às colunas derivadas não depende do significado. O Reloaded gasta 2,2 vezes mais consultas e não ganha.
5. **O MLP nem sempre supera o C4.5** (iris).

## 4. O que foi corrigido

| Commit | Correção |
|---|---|
| `af02d5d` | Coluna numérica de CSV com marcadores de ausente deixou de ser tratada como categórica |
| `16e6be8` | O gate de contribuição semântica deixou de declarar sucesso quando a ontologia não alterou nenhuma decisão, e passou a declarar o âmbito da evidência |

Suite de testes: 1732 aprovados antes, 1742 depois (10 testes novos), 40 ignorados nos dois casos.

## 5. O que encontrei que pode aproximar a hipótese

Experiência exploratória (5 sementes × 3 réplicas, 31 nós): com o amostrador conjunto e todos os atributos por nó, o **Original superou o C4.5 em accuracy no wine (+0,015 [+0,003; +0,034]) e no breast_cancer (+0,012 [+0,008; +0,019])**. Com a configuração atual não o supera em nenhum dataset. No iris continua inviável e no digits faltam nós.

Isto é uma melhoria não ontológica: ajuda o elo `Original > C4.5`, aplica-se igualmente ao Reloaded e não demonstra nenhum ganho da ontologia. Altera o baseline congelado, por isso só pode ser adotada com um manifest novo e uma campanha própria.

## 6. O que não fiz

- **Não implementei novos mecanismos no Reloaded.** O projeto já testou calibrações do peso semântico, desempate, amostragem coerente e controlos (documentos 17 a 42), com veredicto `NO_ONTOLOGICAL_ADVANTAGE`; o meu baseline confirma-o. Repetir variações do mesmo tipo sem corrigir primeiro a referência (secção 7) seria procurar um resultado, não testá-lo.
- **Não corri os controlos de grupos aleatórios nem de engenharia convencional.**
- **Não medi coerência semântica nem estabilidade como métricas próprias.**
- **Não validei a GUI manualmente, nem Windows, nem os contrafactuais.**
- **Não usei datasets novos.** Tudo é DEV; nada é confirmatório.

## 7. Recomendações, por ordem

1. Regenerar o manifest e fixar a referência (5 ficheiros «congelados» derivaram).
2. Adotar, num manifest v2, o Original com todos os atributos por nó e o amostrador conjunto, e repetir o baseline com réplicas. É o único caminho medido para `Original > C4.5`.
3. Igualar o orçamento de consultas entre Original e Reloaded e comparar a complexidade igual.
4. Só depois testar mecanismos ontológicos, contra o Original WIDE e os três controlos, com ontologias que contenham mais do que agrupamentos de colunas.
5. Alargar a pelo menos 6 datasets escolhidos a priori em que o MLP supere o C4.5; com 4 nenhum teste entre datasets pode ser significativo.

Detalhe em `PLANO_CORRECCAO.md` e `PROTOCOLO_EXPERIMENTAL.md`.

## 8. Estado final

| Fase da missão | Estado |
|---|---|
| 1. Diagnóstico | Concluído para algoritmos, dados, ontologias, gates e benchmark; parcial para GUI e contrafactuais |
| 2. Baseline | Concluído: 20 unidades, métricas verificadas |
| 3. Correção | 2 defeitos corrigidos; os restantes documentados com razão |
| 4. Validação | Suite completa antes e depois; pipeline de produção e HermiT executados |
| 5. Otimização | Não concluída: uma via medida de forma exploratória, nenhuma alteração algorítmica adotada |
| 6. Comparação | Concluída para o estado atual do código; hipótese não confirmada |
| 7. Consolidação | Onze documentos e evidência neste diretório |

## 9. Fases seguintes (depois deste relatório)

Este relatório descreve a fase 1. Oito fases posteriores, cada uma com pré-registo anterior à execução, estão documentadas à parte:

| Fase | O que testou | Resultado | Documentos |
|---|---|---|---|
| 2 | A hipótese dentro do harness com o amostrador conjunto em todos os braços, 3 réplicas e os quatro controlos (20 unidades) | Hierarquia sustentada em nenhum dataset; `Original > C4.5` sustentado no digits e observado no wine e no breast_cancer; Reloaded + OWL abaixo do Original nos quatro; nenhum ganho atribuível à ontologia | `FASE2_PREREGISTO.md`, `FASE2_RESULTADOS.md` |
| 3 | Um mecanismo ontológico novo (relações entre grandezas declaradas na ontologia) com ontologias v2 para iris, wine e breast_cancer, sob a referência com amostrador conjunto (15 unidades) | Hierarquia sustentada em nenhum dataset; as relações não ajudam segundo o critério pré-registado; a ontologia v2 não se distingue da permutada e fica abaixo do Original com as mesmas consultas | `FASE3_PREREGISTO.md`, `FASE3_RESULTADOS.md`, `REFERENCIA_V2_DRAFT.json` |
| 4 | Separar os atributos da ontologia v2 dos mecanismos do Reloaded (Original no espaço enriquecido, sem mecanismos, com controlos; 15 unidades) | Os atributos não mudam a árvore (O − B ≈ 0 nos três datasets); os mecanismos do Reloaded prejudicam no iris e no breast_cancer; nenhum efeito atribuível ao significado da ontologia | `FASE4_PREREGISTO.md`, `FASE4_RESULTADOS.md` |
| 5 | Ablação dos mecanismos do Reloaded e correção (15 unidades) | O peso e o ganho modificado não alteram nenhuma decisão sem o EFSR; a perda vem das decisões aprovadas pelo EFSR; as correções pré-registadas eram inertes; a regra escolheu O (atributos sem mecanismos) | `FASE5_PREREGISTO.md`, `FASE5_RESULTADOS.md` |
| 6 | O corrigido em datasets novos: German Credit e três controlos positivos físicos, com a estrutura do tuning e com máx. 7 nós (20 unidades) | Nenhum dataset elegível (MLP não supera o C4.5); nos controlos positivos, Reloaded > Original e > C4.5 com 7 nós, atribuível à ontologia; Original > C4.5 falha porque o MLP é pior do que o C4.5; German Credit sem ganho | `FASE6_PREREGISTO.md`, `FASE6_RESULTADOS.md` |
| 7 | Auditoria e otimização do MLP (seleção só no treino) e nova campanha com o MLP otimizado como oráculo (20 unidades) | O MLP original memorizava os rótulos com ruído; otimizado, ganha 5–6 pontos nos físicos e passa o C4.5; a hierarquia sustenta-se no phys_power (elegível, não atribuível) e no phys_kinetic (não elegível, atribuível); German Credit sem ganho | `FASE7_PREREGISTO.md`, `FASE7_RESULTADOS.md` |
| 8 | O pipeline melhorado (MLP otimizado, Reloaded O, amostrador conjunto) em dados reais: iris, wine, breast_cancer, digits e Adult (novo, amostra de 2000), dois orçamentos (25 unidades) | Elegíveis: wine, breast_cancer, digits. Hierarquia sustentada em nenhum; só observada no digits com a estrutura do tuning; `Reloaded > Original` não sustentado em nenhum elegível (abaixo no wine com 7 nós); a otimização do MLP só melhora sustentadamente o digits; o Adult não é elegível (MLP abaixo do C4.5); nenhum ganho atribuível à ontologia | `FASE8_PREREGISTO.md`, `FASE8_RESULTADOS.md` |
| 9 | O EFSR com gate validado num lote novo de consultas (variante V, isolada em `validation/`), contra Original, Reloaded O e EFSR original, em digits, German Credit, Adult e três controlos físicos (30 unidades) | O gate aprova 2–6× menos propostas; pela regra pré-registada é útil, só pelo German Credit (+1,1 pontos de accuracy face a R, sem ganho de fidelidade); hierarquia sustentada em nenhum elegível (perde a do phys_power que R tinha); o ganho atribuível nos físicos continua a vir dos atributos; não integrar V no núcleo | `FASE9_PREREGISTO.md`, `FASE9_RESULTADOS.md` |

Conclusão acumulada das nove fases: nos seis datasets reais testados (iris, wine, breast_cancer, digits, German Credit, Adult) a hipótese não se verifica, incluindo nos três elegíveis da fase 8 com o pipeline melhorado. Nos controlos positivos sintéticos, depois de corrigir a memorização de ruído do MLP, a hierarquia `Reloaded > Original > C4.5` sustenta-se num dataset elegível (sem atribuição completa à ontologia) e noutro não elegível (com atribuição). O mecanismo de atributos derivados da ontologia produz ganhos atribuíveis quando a ontologia declara uma relação verdadeira que determina a classe; com as ontologias reais disponíveis (agrupamentos e relações não determinantes) não produz. A fase 9 mostrou que um EFSR com validação em consultas novas não muda este quadro: o único sinal novo em dados reais é +1,1 pontos de accuracy no German Credit, sem ganho de fidelidade.
