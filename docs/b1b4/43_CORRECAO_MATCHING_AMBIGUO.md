# Doc. 43 — Correção isolada do matching ambíguo (política P3)

Autorizada na decisão pós-B3.2. É a **única** alteração em `core/` da fase B3.3; feita em commit próprio, separado das experiências. Diagnóstico prévio: Doc. 40.

## 1. Causa
`OntologyQualityGate.match_features` (`core/ontology_quality.py`) aceitava `best_score >= 0.98` sem exigir margem. Num empate exato entre entidades elegíveis (margem 0) a ordenação estável devolvia a
primeira entidade segundo a ordem de `owlready2` — uma escolha **arbitrária, dependente da ordem de declaração**. A preferência de tipo (classe↔propriedade de dados) só olhava para as **duas** primeiras
posições, pelo que mascarava empates de 3 ou mais entidades (p. ex. classe + duas propriedades de dados).

## 2. Contrato anterior preservado
Matching por nome/rótulos; score mínimo 0,72; margem mínima 0,08; uma entidade não serve duas features (colisão: só a de maior score, sem empate, fica; empate ⇒ todas rejeitadas);
preferência da propriedade de dados sobre a classe em empate classe/propriedade (`tie_break = "datatype_property_over_class"`); aceitação de um casamento único com score ≥ 0,98 mesmo com
uma segunda entidade a < 0,08 (ex.: nomes de píxeis). **Não há exceções novas de score.**

## 3. Implementação
Conjunto de empate `T` = entidades com score ≥ melhor − 1e-9.
* `|T| = 1` ⇒ caminho **inalterado** (mesma ordenação estável, mesma regra, mesmas decisões bit a bit).
* `|T| ≥ 2` e `score ≥ min_match_score`:
  * exatamente **uma propriedade de dados** e as restantes são classes ⇒ **prioridade de tipo**: escolhe-se essa propriedade (`tie_break` registado). Interação com a margem: as classes ficam
    excluídas pela prioridade, mas **qualquer outro concorrente não-classe restante** (propriedade de dados ou de objetos) tem de ficar a ≥ `min_ambiguity_margin` do escolhido, **mesmo que o score seja ≥ 0,98**
    (a exceção dos 0,98 não se aplica a empates). Se não ficar ⇒ `REJECTED_AMBIGUOUS` com diagnóstico `competitor_after_type_priority`;
  * qualquer outra composição (≥ 2 propriedades de dados; ≥ 2 classes sem propriedade; propriedade de objetos empatada; ≥ 3 candidatos) ⇒ `REJECTED_AMBIGUOUS`, diagnóstico `tie_between_entities`
    (n.º e identidade/tipo das entidades empatadas, por ordem determinística tipo→nome). A posição no OWL nunca decide.
* Rejeições por margem insuficiente (sem empate) ganham diagnóstico `insufficient_margin`.
* `MatchDecision` ganha o campo opcional `ambiguity_diagnostic` (`None` nas decisões não ambíguas). O `reason` continua `ambiguous_match`, pelo que as métricas `ambiguous_matches` não mudam de definição.
Independência de datasets: nenhuma referência a nomes/identidades de datasets; só scores, tipos de entidade e margens.

## 4. Testes (`tests/test_ontology_matching_p3.py`, 13; `tests/test_b32_matching.py` atualizado)
Candidato exato único; duas propriedades com score 1,0; classe + propriedade com 1,0 (prioridade de tipo); três candidatos empatados (classe+2 propriedades; 3 propriedades; 2 classes); prioridade de tipo
com concorrente a 0,97 (rejeitado) e com concorrente distante (aceite); diferença < 0,08 (0,86 vs 0,84, rejeitado) e margem suficiente (0,86 vs 0,60); invariância à ordem de declaração (rejeição e
aceitação com prioridade de tipo); colisão entre duas features na mesma entidade; rótulos alternativos (e empate de rótulos); nomes normalizados; ontologia desconhecida e ontologia vazia (rejeição sem erro);
mapeamentos históricos idênticos (ficheiro de evidência) e verificação ao vivo. Os dois testes do Doc. 40 que documentavam o defeito passaram a afirmar o comportamento corrigido.

## 5. Comparabilidade e impacto
* **As 7 ontologias históricas mantêm mapeamentos idênticos**: `docs/evidence/b1b4/b33/matching_p3_historical_mappings.json` compara, decisão a decisão (entidade, tipo, aceitação, razão, estado, `tie_break`, score, segundo score,
  margem e segunda entidade), o gate anterior (`git show HEAD:core/ontology_quality.py`) com o corrigido: iris 4/4, wine 13/13, breast_cancer 30/30, digits 64/64, synthetic_binary 9/9, synthetic_3class 12/12, german_credit 61/61
  idênticas; nenhuma decisão com diagnóstico; o relatório do gate coincide fora da lista `matches`. Não foi preciso parar nem tocar em baselines, âncoras ou resultados históricos.
* 126 ontologias geradas da B3.3 (7 variantes × 6 famílias × 3 sementes) dão as mesmas decisões; as experiências B3.3 (Doc. 42) correram antes da correção e não são afetadas.
* Contrato EQ: sem ontologia ou com ontologia rejeitada o caminho do Original não é tocado; o gate só muda decisões em empates exatos de entidades elegíveis, que antes eram arbitrários.
* Ontologias futuras com empates exatos passam a ser rejeitadas (ou, no caso classe/propriedade, resolvidas pela prioridade de tipo) em vez de escolhidas pela posição no OWL.

## 6. Regressão
Matching → equivalência sem OWL → comportamento do quality gate → regressão científica → guarda de agnosticismo (ver resultados no commit). `scripts/check_dataset_agnosticism.py` OK.
