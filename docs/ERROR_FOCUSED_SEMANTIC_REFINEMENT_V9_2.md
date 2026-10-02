# Error-Focused Semantic Refinement (EFSR) — BIURI/TREPAN Reloaded V9.2

## Objetivo

O EFSR aplica o conhecimento ontológico prioritariamente nas regiões em que a
folha corrente do TREPAN discorda do MLP-oráculo. A semântica deixa de ser um
bónus global aplicado indiscriminadamente e passa a actuar como mecanismo local
de proposta/refinamento.

O EFSR é **agnóstico ao dataset**. Não contém nomes de datasets, features,
classes ou thresholds específicos.

## Contrato científico

1. O conjunto de teste externo nunca entra no EFSR.
2. Original e Reloaded usam o mesmo MLP, seed, limite de nós, `max_n`,
   `min_sample` e orçamento de membership queries.
3. A OWL pode orientar a geração de candidatos e a prioridade das queries, mas
   um split semanticamente diferente só substitui o vencedor data-only quando
   melhora a fidelidade local ao MLP pelo mínimo configurado.
4. O EFSR não aprende regras/thresholds do teste nem exige target na OWL.
5. Probabilidades do MLP, quando disponíveis, são reutilizadas apenas em linhas
   já conhecidas pelo TREPAN para medir incerteza; não criam novas membership
   queries.
6. O runtime de produção continua sem CART.

## Fluxo

```text
MLP-oráculo
   │
   ▼
folha/região TREPAN
   │
   ├─ disagreement rate (TREPAN != MLP)
   ├─ reach
   ├─ uncertainty em linhas já consultadas
   └─ perfil de features associadas ao erro
   │
   ▼
Error Region Profile
   │
   ▼
OntologySemanticGraph
   │
   ├─ grupos semânticos
   ├─ relatedness
   └─ pesos/matching
   │
   ▼
Candidatos m-of-n orientados pela região de erro
   │
   ▼
Local Fidelity Gate
   ├─ melhora fidelidade local -> aceitar
   └─ não melhora -> voltar ao split data-only
```

## Detecção de regiões de erro

Para uma região `R`:

```text
Disagreement(R) = P(TREPAN(X) != MLP(X) | X em R)
```

A prioridade histórica `reach * (1 - fidelity)` é preservada. O Reloaded apenas
reforça a prioridade de regiões que simultaneamente apresentam erro e uma
oportunidade semântica observável.

## Perfil feature↔erro

O EFSR calcula uma diferença robusta (medianas, IQR/escala robusta) entre os
casos locais de discordância e de concordância. Essa pontuação é combinada com
estrutura OWL (grupo, relatedness e peso semântico), sem usar o teste externo.

## Membership queries focadas

As queries continuam a respeitar as restrições raiz→nó e o mesmo orçamento do
Original. Quando a região é elegível, o pool é priorizado por proximidade a
**anchors de erro já conhecidos**, limites candidatos e relevância semântica.

O MLP não é consultado em candidatos descartados apenas para ordená-los.

## Gate local obrigatório

Se a semântica propuser um split diferente, são comparadas as fidelidades locais
que seriam obtidas ao prever a classe majoritária em cada ramo:

```text
fidelity_data_only
fidelity_semantic
Delta = semantic - data_only
```

A alteração só é aceite se:

```text
Delta >= error_focus_min_local_fidelity_gain
```

Caso contrário, o nó usa exactamente o vencedor data-only do TREPAN histórico.

## Auditoria produzida

O modelo guarda por nó:

- `error_region_disagreement_rate`
- `error_region_semantic_opportunity`
- `error_region_mean_uncertainty`
- `top_error_semantic_features`
- `data_only_test`
- `semantic_test`
- `selected_test_after_error_gate`
- `data_only_local_fidelity`
- `semantic_local_fidelity`
- `local_fidelity_gain`
- `error_focused_intervention_attempted`
- `error_focused_intervention_accepted`
- motivo de aceitação/rejeição

O resumo inclui:

- regiões de erro avaliadas/elegíveis;
- intervenções tentadas/aceites/rejeitadas;
- queries focadas em erro;
- reutilização de probabilidades em linhas já conhecidas;
- Ontology Usage Rate e Semantic Decision Impact.

## Validação sintética de mecanismo

`V92_EFSR_MECHANISM_VALIDATION.json` demonstra que o mecanismo consegue gerar
ganho quando existe estrutura semântica realmente informativa. É um teste
sintético de mecanismo, **não é evidência de benchmark nem resultado da tese**.
