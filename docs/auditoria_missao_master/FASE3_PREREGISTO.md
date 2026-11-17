# Fase 3 — ontologias com relações declaradas, sob a referência v2: pré-registo

Escrito e registado em commit **antes** de qualquer execução desta fase. Os resultados irão para `FASE3_RESULTADOS.md`, sem alterar este documento.

## 1. O que muda nesta fase

**Referência v2 (proposta).** Descrita em `REFERENCIA_V2_DRAFT.json`: todos os braços TREPAN usam o amostrador conjunto do projeto, com 3 réplicas. Não substitui o manifest v1; é um rascunho para decisão do responsável do projeto.

**Mecanismo ontológico novo.** `core/ontology_processor.py` passa a ler a anotação `derivedRelation` numa propriedade: `"<operação>:<outra propriedade>[:<expoente esquerdo>:<expoente direito>]"`, com operação em `ratio`, `product`, `difference`, `sum`. A relação gera um atributo derivado só se as duas propriedades estiverem mapeadas em colunas numéricas. É genérico: nenhuma regra por dataset; sem a anotação nada muda (as ontologias v1 produzem exatamente os mesmos atributos). 12 testes novos em `tests/test_audit_declared_relations.py`.

**Ontologias v2** (`validation/benchmark/ontologies/v2/`, geradas por `build_v2.py`): as v1 mais relações declaradas.

| Dataset | Relações declaradas | Estatuto |
|---|---|---|
| iris | comprimento × largura e comprimento / largura, para sépala e pétala; pétala / sépala em comprimento e em largura | INFERIDO (geometria e morfometria elementares) |
| wine | flavonoides / fenóis totais; não flavonoides / fenóis totais; proantocianidinas / flavonoides | PROVADO (química dos compostos fenólicos; Ribéreau-Gayon et al., 2006) |
| wine | alcalinidade das cinzas / cinzas; magnésio / cinzas | INFERIDO |
| breast_cancer | perímetro² / área (média e pior) | PROVADO (definição de compacidade do dicionário de dados; Street, Wolberg, Mangasarian, 1993) |
| breast_cancer | perímetro / raio e área / raio² (média e pior) | INFERIDO (2π e π num círculo) |
| digits | nenhuma | Não tenho relações entre píxeis com fundamento de domínio que o mecanismo exprima |

As referências bibliográficas foram citadas de memória e não confirmadas online nesta sessão.

**Declaração de contaminação.** Escolhi as relações a partir de definições, sem consultar rótulos, importâncias ou resultados. Mas estes são datasets muito conhecidos e já vi os seus resultados agregados nas fases 1 e 2; não posso garantir que o meu conhecimento geral sobre eles não influenciou a escolha. Por isso esta fase é exploratória e um resultado positivo teria de ser repetido em datasets novos.

## 2. Desenho

- **Unidades:** iris, wine, breast_cancer × sementes 42, 7, 123, 2024, 11 (15 unidades). O digits não entra porque não tem ontologia v2; a razão está fixada acima, antes de qualquer resultado.
- **Partições, MLP e estrutura das árvores:** os das fases 1 e 2.
- **Executor:** `validation/benchmark/run_fase3.py`, que reutiliza `run_sampler_c` e `run_controls_m29` com as ontologias v2. 3 réplicas. Limite operacional de 5400 s por unidade.
- **Braços:** C4.5; B (Original + conjunto); D2 (Reloaded + OWL v2 + conjunto); F (Reloaded sem OWL + conjunto); controlos sobre a v2: permutada (P2), grupos aleatórios (Dg2), engenharia convencional (Ec2), Original com consultas igualadas (G2). D1 (Reloaded + OWL v1 + conjunto) vem da fase 2.
- **Verificação de integridade:** o braço B não depende da ontologia e tem de reproduzir os valores da fase 2. Se não reproduzir, a fase é inválida e reporto-o.

## 3. Contrastes e critério

Medida por unidade: média das 3 réplicas. `X > Y` num dataset: diferença média emparelhada positiva e intervalo a 95 % (bootstrap percentil, 20 000 reamostragens das 5 unidades) acima de zero.

| # | Condição |
|---|---|
| H1 | accuracy: D2 > B |
| H2 | accuracy: D2 > C4.5 |
| H3 | accuracy: B > C4.5 (já medida na fase 2; repetida como verificação) |
| H4 | fidelidade: D2 > B |
| H5 | fidelidade: D2 > C4.5 |
| K1 | D2 > D1, em accuracy e em fidelidade (as relações acrescentam algo às ontologias v1?) |

Atribuição (só interpretada se H1 e H4 se observarem): D2 > F, P2, Dg2, Ec2 e G2, nas duas métricas.

Limitação conhecida do controlo Ec2: gera médias de grupos escolhidos pelos dados, não razões nem produtos. O controlo que aplica as mesmas operações sem o significado é a ontologia permutada (P2).

## 4. Regras de decisão

- **Hierarquia sustentada num dataset:** H1 a H5 com intervalo acima de zero.
- **Hierarquia observada:** cinco médias positivas.
- **Ganho atribuível à ontologia:** H1 e H4 sustentadas e D2 acima dos cinco controlos nas duas métricas.
- **As relações ajudam:** K1 com intervalo acima de zero nas duas métricas.
- Resultado negativo ou nulo é reportado como tal. Não altero as relações declaradas, as sementes nem os datasets depois de ver resultados.
