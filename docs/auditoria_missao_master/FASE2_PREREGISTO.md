# Fase 2 — hipótese sob a referência com amostrador conjunto: pré-registo

Escrito e registado em commit **antes** de qualquer execução desta fase. Os resultados serão acrescentados num documento separado (`FASE2_RESULTADOS.md`) sem alterar este.

## 1. Motivação

A fase 1 mostrou que a hierarquia conjunta não se verifica (0 de 20 unidades) e que, numa experiência exploratória fora do harness, o amostrador conjunto do projeto melhora o TREPAN Original. Esta fase testa, **dentro do harness oficial**, com a estrutura escolhida pelo tuning do projeto e com réplicas, se a hipótese se verifica quando Original e Reloaded usam ambos o amostrador conjunto, e se um eventual ganho é atribuível à ontologia.

Não altero nenhum algoritmo. Uso os executores experimentais que já existem no repositório: `validation/benchmark/run_sampler_c.py` e `validation/benchmark/run_controls_m29.py`. O projeto já correu campanhas com estes executores em 18 unidades (documentos 26 e 30); esta é uma reprodução independente em 20 unidades, acrescida da comparação com o C4.5.

## 2. Desenho

- **Unidades:** as mesmas 20 da fase 1 (iris, wine, breast_cancer, digits × sementes 42, 7, 123, 2024, 11). Nenhuma é excluída.
- **Partições, MLP e estrutura das árvores:** os da fase 1 (pontos de controlo do tuning reutilizados; o harness verifica as chaves).
- **Réplicas:** 3 por braço e unidade (`--reps 3`, `--c-replicas all`).
- **Limite operacional:** 5400 s por unidade, igual para todos os datasets. Uma unidade que o exceda é reportada como `resource_limit_exceeded` e conta como não verificada.
- **Braços (campanha 1, `run_sampler_c`):**
  - C4.5 (rótulos reais);
  - A: Original com amostrador marginal (histórico);
  - B: Original + amostrador conjunto;
  - C: Reloaded + OWL, amostrador marginal (histórico);
  - D: Reloaded + OWL + amostrador conjunto;
  - E: Reloaded sem OWL, amostrador marginal;
  - F: Reloaded sem OWL + amostrador conjunto.
- **Braços (campanha 2, `run_controls_m29`, todos com amostrador conjunto):** ontologia permutada (P), grupos aleatórios (Dg), engenharia convencional de atributos (Ec), Original com consultas igualadas às do Reloaded + OWL (G).

## 3. Medida por unidade

Para cada braço com réplicas, a medida da unidade é a **média das 3 réplicas**. As comparações são emparelhadas por unidade.

## 4. Contrastes e critério

`X > Y` num dataset: diferença média emparelhada positiva **e** intervalo de confiança a 95 % (bootstrap percentil, 20 000 reamostragens das 5 unidades) acima de zero. Diferença positiva com intervalo a incluir zero é «observada, não sustentada».

Hierarquia sob a nova referência (cinco condições, por dataset):

| # | Condição |
|---|---|
| H1 | accuracy: D > B (Reloaded + OWL > Original) |
| H2 | accuracy: D > C4.5 |
| H3 | accuracy: B > C4.5 (Original > C4.5) |
| H4 | fidelidade: D > B |
| H5 | fidelidade: D > C4.5 |

Atribuição à ontologia (só interpretada se H1 e H4 se observarem): D > F, D > P, D > Dg, D > Ec e D > G, em accuracy e em fidelidade.

Efeito do amostrador (descritivo): B − A, D − C, F − E.

## 5. Regras de decisão

- **Hierarquia sustentada num dataset:** H1 a H5 com intervalo acima de zero.
- **Hierarquia observada:** as cinco diferenças médias positivas, alguma com intervalo a incluir zero.
- **Ganho atribuível à ontologia:** H1 e H4 sustentadas **e** D acima dos cinco controlos nas duas métricas.
- **Por unidade** (descritivo): contagem das unidades em que as cinco condições se verificam com as médias das réplicas.
- Entre datasets não é feito nenhum teste de significância: com 4 datasets o valor p mínimo de um teste dos sinais bilateral é 0,125.

## 6. O que esta fase não pode mostrar

- Não é confirmatória: os datasets já foram observados.
- O amostrador conjunto é uma alteração não ontológica à referência; mesmo que H3 se verifique, isso não é um ganho da ontologia.
- As ontologias são as estruturais do benchmark. A ontologia enriquecida do Breast Cancer não entra nesta fase.
- Se o resultado for negativo, reporto-o tal como sair.
