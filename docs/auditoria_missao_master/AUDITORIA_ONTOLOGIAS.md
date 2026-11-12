# Auditoria das ontologias OWL

Método: cada ficheiro `.owl` do repositório foi carregado com owlready2 0.47 num processo isolado e submetido ao HermiT (via Java). Resultado completo por ficheiro em `evidencia/audit_ontologies.json`; script em `evidencia/audit_ontologies.py`. Estado: **EXECUTADO**.

## 1. Resultado formal

- 41 ficheiros analisados. Todos carregam. Todos são **consistentes no HermiT, sem classes insatisfazíveis**.
- Nenhum usa `owl:imports`.
- `xml:lang` só aparece em `data/adult_census_full_ontology.owl` (57 ocorrências), que mesmo assim é consistente.
- Ficheiros com indivíduos (ABox), que o quality gate rejeita por desenho:
  - `data/adult_census_full_ontology.owl`: 48 952 indivíduos (o dataset inteiro materializado);
  - `experiments/ontology_enrichment_study/synthetic/s2_*`: 24 indivíduos cada (6 ficheiros).

Limitação: não validei o perfil OWL 2 DL estrito com um validador de perfil; «consistente no HermiT» não é o mesmo.

## 2. Ontologias estruturais e enriquecidas

| Grupo | Ficheiros | Conteúdo | Classificação |
|---|---|---|---|
| Benchmark (`validation/benchmark/ontologies/*.owl`, cópias em `data/benchmark_ontologies*`) | iris, wine, breast_cancer, digits, sintéticos, german_credit | 3–6 classes, 1 disjunção, uma propriedade por coluna com `rdfs:domain` e `rdfs:label`; sem facetas, sem famílias de medida (o breast_cancer declara 2 propriedades de anotação) | **Estrutural**: só agrupa colunas |
| Breast Cancer enriquecida (`data/ontologias_demo/*.owl`, `experiments/.../breast_cancer_v3_ENRIQUECIDA.owl`) | 2 + variantes | 20–21 classes, 9 disjunções, 30 propriedades com faceta `minInclusive 0`, `measurementFamily` e `statisticRole` | **Enriquecida**; fonte declarada: dicionário de dados UCI |
| German Credit (`.../proposals/german_credit/v0_draft/`) | 5 | 41 classes | Rascunho; o próprio projeto declara os axiomas como não verificados |
| Adult (`data/adult_census_full_ontology.owl`) | 1 | ABox | Não utilizável como TBox |
| Sintéticas S1/S2 | 12 | correta, permutada, aleatória | Material de controlo |

Só existe uma ontologia cientificamente enriquecida para um dataset real (Breast Cancer). As conclusões sobre «ontologia enriquecida» assentam num único dataset.

## 3. Correspondência com os dados

- Nas ontologias de benchmark, o número de propriedades com `rdfs:label` coincide com o número de colunas de cada dataset (iris 4, wine 13, breast_cancer 30, digits 64). A cobertura efetiva medida pelo quality gate em execução está nos resultados por unidade (`mapping_rate`).
- A ontologia enriquecida do Breast Cancer usa os nomes UCI (`radius_mean`, ...), que coincidem com `data/ontologias_demo/breast_cancer_wisconsin_diagnostic_dados.arff`. O pipeline de produção aceitou-a com estado `VALID_DOMAIN_ONTOLOGY` em 3 execuções reais.
- Coluna-alvo e ordem das classes: declaradas por anotações `biuriColumnName`, `biuriIsTarget`, `biuriClassOrder` só na ontologia enriquecida.

## 4. O que o conhecimento consegue produzir

Com as ontologias estruturais, o único derivado possível é a média padronizada por grupo. No iris isso dá 2 atributos derivados. Com a enriquecida do Breast Cancer acrescentam-se as relações de família (pior − média, variação relativa, contraste, erro normalizado, razão erro/média). As 30 facetas `minInclusive 0` não geram nenhum atributo: nenhum valor é negativo e o indicador é constante.

## 5. Proveniência

- Benchmark: os comentários das ontologias declaram que a estrutura veio só da descrição das colunas, sem consultar rótulos nem resultados. Não é verificável a posteriori; registo a declaração.
- Breast Cancer enriquecida: `dct:source` aponta para a página UCI do dataset. O agrupamento em geometria, indentação, regularidade e textura é uma leitura do dicionário de dados, não um resultado clínico publicado.
- `vehicle_enriquecido_v3.owl` (produzida nesta sessão, fora do repositório): a ligação coluna → conceito é hipotética; ver a tabela de axiomas entregue com ela.
