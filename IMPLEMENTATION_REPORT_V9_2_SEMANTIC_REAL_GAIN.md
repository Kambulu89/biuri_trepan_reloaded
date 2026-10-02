# IMPLEMENTATION REPORT — V9.2 Semantic Real-Gain Contract

## Objectivo

Fechar os defeitos observados numa execução real em que a ontologia estava válida e
100% mapeada, mas o relatório mostrava zero relações semânticas, zero regiões EFSR
elegíveis e uma árvore Reloaded diferente do Original apesar de impacto semântico
reportado como zero.

A implementação permanece dataset-agnostic, sem CART, com modo Científico obrigatório,
mesmo MLP/orçamento/capacidade entre TREPAN Original e Reloaded e sem uso do teste
externo na selecção.

## Correcções implementadas

### 1. Grafo OWL validado passa a ser a fonte semântica efectiva

`core/ontology_semantic_graph.py` e `core/trepan_reloaded_extractor.py` foram integrados
para que os matches aprovados pelo quality gate sejam promovidos para um grafo com:

- classes;
- datatype/object/annotation properties;
- `subClassOf`/`subPropertyOf`;
- equivalências;
- domain/range;
- inverse properties;
- ancestors materializados quando disponíveis;
- grupos semânticos por feature;
- matriz de relatedness entre features.

O extractor reaplica o grafo aprovado depois dos caminhos legados de extracção de
conhecimento, impedindo que `feature_semantics` seja sobrescrito por uma versão pobre.

### 2. EFSR deixa de produzir `regiões avaliadas > 0 / elegíveis = 0` por threshold rígido

Cada região calcula uma prioridade composta de `reach`, discordância MLP↔surrogate,
incerteza e oportunidade semântica. Além do threshold normal, existe um fallback top-k
controlado que permite refinar um número mínimo de regiões com erro real quando a
fidelidade ainda não é perfeita.

O fallback não cria erro nem usa `y_test`; apenas decide onde gastar a avaliação
semântica dentro do treino.

### 3. Gate obrigatório em linhas reais de treino

Uma intervenção EFSR precisa agora passar dois gates:

1. melhorar a fidelidade no decision sample (reais + membership queries);
2. melhorar também a fidelidade nas linhas REAIS de treino que chegaram ao nó.

Se a proposta OWL melhorar apenas as queries sintéticas e piorar/não melhorar os dados
reais internos, ela é rejeitada com:

`semantic_candidate_fails_real_training_gate`.

Isso ataca directamente o padrão observado de alta fidelidade sintética e fidelidade
real inferior.

### 4. Espelhamento exacto quando não existe contribuição semântica mensurável

Se a OWL estiver presente, mas no fim ocorrer:

- zero splits influenciados/reforçados; e
- zero intervenções EFSR aceites,

então o modelo final Reloaded é reconstruído/adoptado exactamente a partir do mesmo
TREPAN Original, com seed, dados, orçamento e capacidade idênticos.

Logo o contrato é:

`sem efeito semântico mensurável => Reloaded == Original`

em regras, previsões e fidelidade.

### 5. Identidade da OWL na GUI

Ao carregar uma ontologia, a GUI passa a registar e mostrar:

- nome do ficheiro;
- `owl:versionInfo` quando disponível;
- SHA-256;
- número de classes e propriedades.

O resumo de treino inclui ainda:

- relações OWL disponíveis;
- features com grupo semântico;
- origem do mapping.

Isso permite distinguir inequivocamente uma OWL antiga de uma versão enriquecida.

### 6. Auditoria EFSR mais completa

Por nó passam a ser guardados, entre outros:

- disagreement rate;
- prioridade da região;
- motivo da elegibilidade (`threshold` ou `top_k_error_region_fallback`);
- fidelidade local data-only/semântica;
- fidelidade em linhas reais de treino data-only/semântica;
- deltas;
- intervenção tentada/aceite/rejeitada;
- motivo da decisão.

O resumo informa também se foi aplicado o mirror exacto para o Original.

## Formato de artefacto

Novo formato:

`biuri-v9.2-production-5-semantic-real-gain`

O loader mantém compatibilidade de leitura com os formatos production-2, production-3
e production-4.

## Testes adicionados

`tests/test_semantic_real_gain_production_v92.py` cobre:

1. promoção dos matches para grafo, relações e grupos;
2. fallback top-k para região com erro abaixo do threshold fixo;
3. rejeição de candidato que melhora queries mas piora dados reais de treino;
4. mirror exacto Original=Reloaded sem efeito semântico mensurável.

## Resultados realmente executados

- `compileall core gui counterfactuals tests scripts`: OK.
- `test_semantic_real_gain_production_v92.py + EFSR + semantic audit + scientific tuning + ontology gate + no-ontology mirror`: **36 passed**, 2 warnings.
- `production no-CART + controlled ablation + evaluation + production semantic + artifacts + scientific lock + historical Original/Reloaded`: **39 passed**.
- `data contract + preprocessing + dataset-agnostic matrix + Reloaded context/oracle + variable-feature dual-oracle`: **33 passed**, warnings de convergência dos MLPs de teste.
- Após mudança do formato do artefacto: `artifacts + production semantic + production no-CART + semantic real-gain`: **16 passed**.
- Teste isolado de mecanismo EFSR com ganho sintético controlado: **1 passed**.

Uma execução agregada maior de 20 ficheiros excedeu o timeout do runtime depois de
mais de 50 testes apresentados sem falha. Não é declarada como suite monolítica verde.

## Invariantes de produção verificados

- CART no runtime: nenhum offender.
- modo Científico obrigatório: activo.
- EFSR: activo.
- gate de ganho em treino real: activo.
- fallback top-k de regiões de erro: activo.
- mirror sem efeito semântico: activo.
- teste externo usado na selecção EFSR: não.
- `results/confirmatory_v7/`: SHA-256 inalterados.

## Limitações deste ambiente

O readiness global continua `false` porque este runtime usa Python 3.13.5, enquanto o
alvo é 3.11/3.12, e não contém toda a stack final (por exemplo `dtreeviz`; GUI/OWL não
foram solicitados no preflight usado nesta validação). A execução real com
PyQt6 + Owlready2/HermiT no Windows continua necessária antes de chamar a release de
produção final.
