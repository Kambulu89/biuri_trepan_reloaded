# Doc. 47 — Modelo de enriquecimento ontológico real: construção, validação e aplicação de ontologias expressivas

Estado: **proposta metodológica** (B3-Q). Nenhuma ontologia foi construída ou alterada; nenhuma revisão humana existe ou é declarada; os três papéis (domínio, engenharia de ontologias, metodologia) continuam **por designar**. Complementa o Doc. 45 (qualidade) e o Doc. 39 (pacotes de revisão). Restrições: só `validation/` e documentação; nada em `core/`.

## 0. Objetivo e princípio
Produzir ontologias de domínio **verdadeiras, documentadas, relevantes e rastreáveis**, para que uma futura avaliação teste **conhecimento de qualidade** e **mecanismos capazes de o usar** (H2, Doc. 48), em vez de repetir testes em que a ontologia é só uma partição de atributos.
Regras duras: (1) nenhuma relação de domínio é criada por observação de correlações nos dados; (2) nenhum axioma é escolhido, mantido ou removido com base em métricas de teste; (3) ontologias congeladas não se alteram — trabalha-se com **propostas versionadas**; (4) nenhuma ontologia é «independentemente validada» sem aprovações verificáveis de pessoas designadas.

## 1. Processo (fases E0–E9)

| Fase | Atividade | Saída (artefacto) | Quem |
|---|---|---|---|
| E0 | Âmbito e **perguntas de competência**: que decisões do classificador dependem de conhecimento externo? (p. ex. «que atributos medem a mesma grandeza?», «que medida limita outra?») | `scope.md`, lista de perguntas | metodologia + domínio |
| E1 | **Identificação de fontes** independentes dos dados de avaliação: normas, manuais, artigos de aquisição do dataset, ontologias de domínio reconhecidas, documentação do instrumento. Registar critério de independência | `sources.json` (id, citação, localizador, data, licença) | domínio |
| E2 | **Aquisição de conhecimento**: transcrever afirmações das fontes para **fichas de axioma** (§2). Proibido: derivar de correlações ou de resultados do benchmark | fichas `AX-nnn` | domínio |
| E3 | **Modelação OWL 2 DL**: converter fichas em axiomas; usar apenas construções com fonte (equivalência, subpropriedade, disjunção, restrições de valor/cardinalidade, tipos de dados com facetas, propriedades de objeto) — «mais axiomas» não é objetivo | `onto_vN.owl` | engenharia de ontologias |
| E4 | **Validação lógica**: perfil OWL 2 DL formal, consistência e satisfatibilidade (HermiT), ausência de *punning*; **testes de inferência com resultados esperados escritos antes de correr** | `validation_report.json` | engenharia de ontologias |
| E5 | **Matching** *features*→entidades por nome/rótulo (contrato do gate com P3); tabela de mapeamento revista; lista de não mapeados e consequências | `mapping.csv` + diagnósticos | engenharia + domínio |
| E6 | **Revisão independente** por três papéis, com formulários e assinatura **escrita pelos revisores** (a máquina nunca gera aprovações) | formulários assinados | humanos designados |
| E7 | **Rastreabilidade**: axioma → facto no registo → candidato → decisão → árvore (`semantic_trace_`); relatório Q1–Q6 (Doc. 45) **antes** do benchmark | `quality_report.json` | metodologia |
| E8 | **Versionamento e congelamento** (§4) | tag, hashes | metodologia |
| E9 | **Controlos** pré-registados (§5) | ontologias de controlo | metodologia |

## 2. Ficha de axioma (modelo)
`id` · `statement_nl` (afirmação em linguagem natural) · `owl_axiom` (sintaxe funcional) · `entities` (classes/propriedades e atributos mapeados) · `source` {citação, localizador, data} · `scope_and_validity` (condições em que é verdadeiro; exceções conhecidas; unidade/escala) · `universality` (`UNIVERSAL` / `CONDITIONAL` / `REGULARITY_ONLY`) · `falsifiability_check` (teste contra **fonte ou dados de treino de um dataset diferente do de avaliação**, definido antes; serve para **refutar**, nunca para criar) · `epistemic_status` (`LOGICALLY_ENTAILED` / `DOMAIN_DOCUMENTED` / `HYPOTHESIS`) · `permitted_use` (`NONE` / `PREFERENCE` / `HARD_CONSTRAINT`, só `HARD_CONSTRAINT` se `UNIVERSAL` e revisto) · `review` {domain, ontology, methodology: `*_REVIEW_PENDING`} · `independence_from_eval` (declaração). Exemplo de preenchimento (**não** é axioma aprovado): `wdbc_order_relations` — «o valor ‘worst’ de uma família de medidas é ≥ ao valor ‘mean’» (Doc. 39, hipótese até revisão de domínio).

## 3. Que conhecimento pode ser incorporado e que mecanismo o usa
Esta tabela impede a confusão entre conhecimento de grupos (H1) e semântica OWL expressiva (H2). **Hoje o motor só consome a partição.**

| Tipo de axioma | Informação | Mecanismo que o poderia usar | Estado atual |
|---|---|---|---|
| Agrupamento/taxonomia | partição e hierarquia de atributos | unidades de pesquisa, relatedness taxonómica (B3.1) | **implementado** (arnês, só `validation/`) |
| Disjunção entre conceitos | atributos de conceitos disjuntos não se combinam num mesmo teste m-of-n | restrição de candidatos | não implementado; ignorado pelo núcleo |
| Equivalência / subpropriedade | atributos intermutáveis ou especializações | colapso/ordem de candidatos, redução de redundância | não implementado |
| Limites e facetas de valor | domínio de cada atributo | restrição de limiares candidatos / amostragem plausível | parcial (limites explícitos) |
| Ordem entre medidas (`worst ≥ mean`) | relação de ordem | filtro de consultas plausíveis / restrição a literais | não implementado (hipótese, revisão pendente) |
| Restrições de cardinalidade, papéis | contagem/papéis | restrições de m-of-n (valor de *n*) | não implementado |
| Inferência de reasoner | consequências não declaradas | propagação a grupos/factos | não implementado (0 inferências nas DEV) |
**Condição de H2:** para cada tipo de axioma usado num estudo, deve existir um **mecanismo efetivo**, ativável por flag, auditável e com controlos (permutado, aleatório estruturalmente comparável, relações neutralizadas, engenharia convencional). Sem mecanismo, o axioma permanece `COMPILED` e não pode sustentar H2.

## 4. Versionamento, hashes e critérios de congelamento
* Layout: `validation/benchmark/ontologies/proposals/<dataset_id>/vN/{onto.owl, sources.json, axioms/*.json, mapping.csv, validation_report.json, quality_report.json, reviews/}`. As ontologias atuais em `validation/benchmark/ontologies/*.owl` **não se tocam**.
* Cada versão tem SHA-256 de cada ficheiro e da árvore; `CHANGELOG` por versão; versões antigas nunca se apagam.
* **Critérios de congelamento** (todos): (i) todas as fichas com fonte e localizador; (ii) Q1 sem violações; (iii) testes de inferência esperados cumpridos; (iv) matching revisto e sem ambiguidades por resolver; (v) relatório Q1–Q6 gerado e arquivado **antes** de qualquer benchmark; (vi) controlos e o pré-registo escritos antes de se ver qualquer resultado; (vii) estado das revisões humanas registado tal como está (`PENDING` ou assinado); (viii) commit separado dos experimentos.
* Estado epistémico por versão: `DRAFT` → `FROZEN_REVIEW_PENDING` → `FROZEN_REVIEWED` (só com aprovações verificáveis). Uma ontologia revista **depois** de observar benchmarks DEV só pode usar-se em estudos DEV exploratórios.

## 5. Controlos obrigatórios por ontologia enriquecida
Todos construídos pela **mesma** rotina, com os mesmos tamanhos e a mesma cobertura, antes da avaliação:
1. **Permutada ao nível da relação** (as relações trocam de atributos, preservando a estrutura);
2. **Neutralizada** (as relações relevantes removidas, o resto igual);
3. **Aleatória estruturalmente comparável** (mesma distribuição de graus/tipos de axioma);
4. **Plana equivalente** (mesmos grupos, sem a expressividade — separa H1 de H2);
5. **Sem OWL** (os mesmos grupos como lista simples);
6. **Engenharia convencional** de *features* (agrupamento por dados) e **pesquisa genérica de interações**.

## 6. Matching, inferência e rastreabilidade
Matching com o contrato do gate corrigido (P3): empates exatos entre entidades elegíveis rejeitam-se com diagnóstico; prioridade de propriedade de dados sobre classe com margem; colisões rejeitadas. Para cada axioma, o relatório regista a cadeia `AVAILABLE→…→BENEFIT_OBSERVED` (Doc. 45 Q6), nunca saltando estados. Testes de inferência: para cada axioma que deve gerar consequência, escreve-se a consequência esperada **antes** de correr o reasoner.

## 7. Regras de uso com datasets reais
Usar apenas documentação de domínio e dados de **treino** autorizados; não usar datasets confirmatórios; não usar resultados de teste para escolher axiomas; ontologias feitas para Iris, Wine, Breast Cancer ou Digits depois de conhecidas as suas métricas não são prova confirmatória independente.

## 8. Revisão humana
Formulários dos Docs. 39 (domínio, ontologia, metodologia) a preencher por pessoas reais; a máquina só verifica presença de fonte, localizador, âmbito e hashes (`check_package`); **nunca** declara um revisor, uma assinatura ou uma aprovação. Enquanto não existirem designações, todos os estados de revisão ficam `*_REVIEW_PENDING`.

## 9. Esforço e custo (estimativa de planeamento)
Trabalho humano dominante (aquisição de fontes e revisão); computacional desprezável (reasoner + relatório Q1–Q6: segundos por ontologia). O custo de CPU de um futuro estudo H2 depende dos mecanismos a implementar (fora de âmbito: requer autorização e não está estimado aqui).
