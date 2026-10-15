# Documento 5 — Contrato de superioridade (PROPOSTA, revista pela correção arquitetural)

> **Revisão 2 (requisito conjunto obrigatório):** a hierarquia exige agora, **em conjunto**, `A_R > A_O`, `A_O > A_C`, `A_R > A_C`, `F_R > F_O` e `F_R > F_C` (mesmas instâncias, mesmo MLP). U3 passa a ser exigida (e F_R > F_C, antes «suplementar»). Um ganho unilateral (accuracy sem fidelidade, ou o inverso) é `TRADEOFF_*` e **não** é sucesso. A seleção em validação deve ser conjunta (elegível só se ΔA_val ≥ 0 e ΔF_val ≥ 0 face ao Original equivalente). Fonte única: `core/hierarchy_contract.py`; norma: `28_REQUISITO_HIERARQUIA_CONJUNTA.md`. O texto abaixo mantém-se como histórico; onde diverge (U3/U5 e tratamento de F_C como suplementar, U3 «só se o protocolo a impuser»), vale a Revisão 2. Accuracy ≠ precision (precision_macro e macro-F1 são complementares).

> **Revisão:** a hierarquia é `Accuracy_Reloaded_OWL > Accuracy_Original > Accuracy_C4.5` e `Fidelity_Reloaded_OWL > Fidelity_Original`. «Reloaded» abaixo significa **Reloaded com ontologia efetiva (R_OWL)**. Sem ontologia vale o contrato de equivalência (`07_CONTRATO_EQUIVALENCIA.md`): `R∅ ≡ Original` — esse controlo tem o seu próprio veredito `EQ ∈ {EQUIVALENT, VIOLATION}`, **fora** dos quatro estados da hierarquia (uma igualdade exata não é uma «superioridade»). O ganho é sempre medido contra o Original equivalente (mesmo oráculo, dados, parâmetros, seed; mesma representação quando se isola a orientação semântica: Doc. 7 §7).

Define **formalmente**, antes de qualquer execução, o que significa a hierarquia `TREPAN Reloaded > TREPAN Original > C4.5`, como é otimizada (só em validação), medida,
verificada e reportada. A hierarquia é um **requisito de seleção/aceitação e de relatório**, nunca uma garantia de resultado nem uma autorização para mexer nos baselines.

## 1. Símbolos (todos em precisão total, sem arredondamento)
Para uma unidade `u` (dataset × partição), sobre **o mesmo conjunto de teste** e o mesmo oráculo congelado:
* `A_R(u), A_O(u), A_C(u)`: accuracy contra rótulos reais (R_OWL selecionado, Original, C4.5 canónico). Se a ontologia não for efetiva na unidade, `R=R∅≡O`: a unidade recebe `EQ`, não entra nos vereditos de hierarquia e conta como `INSUFFICIENT_EVIDENCE` para H1/H3 (sem tratamento).
* `F_R(u), F_O(u)`: fidelidade ao oráculo congelado (`fidelity_to_oracle`); `F_C` é medida suplementar (não faz parte da cadeia).
* `δ_A ≥ 0`: margem de superioridade em accuracy. **δ_A^estado = 0** (comparações estritas `>`, empates não contam) para o estado por unidade; **δ_A^alegação = 0,01** (pré-registado) para
  declarar superioridade material no Documento 2.
* `δ_F ≥ 0`: margem de superioridade em fidelidade (por omissão 0, estrito). `ν_F = 0,01`: margem de **não-inferioridade** usada como *restrição* na seleção (não confundir com δ_F).

Condições:
```
U1:  A_R(u) − A_O(u) ≥ δ_A      (estrito se δ_A = 0)
U2:  A_O(u) − A_C(u) ≥ δ_A      (estrito se δ_A = 0)
U3:  F_R(u) − F_O(u) ≥ δ_F      (estrito se δ_F = 0)         [exigida só se o protocolo a impuser]
```

## 2. Estados (os quatro pedidos) e nível de evidência
| Estado | Definição |
|---|---|
| `HIERARCHY_SATISFIED` | U1 ∧ U2 (∧ U3 se exigida) |
| `HIERARCHY_NOT_SATISFIED` | U2 verdadeira e (U1 falsa ou U3 falsa) |
| `HIERARCHY_INFEASIBLE` | U2 falsa: o Original não supera o C4.5 ⇒ a cadeia completa é inalcançável com estes baselines (baselines **não** são alterados) |
| `INSUFFICIENT_EVIDENCE` | dados em falta/arm falhado/n abaixo do mínimo, ou intervalo de confiança inconclusivo (níveis dataset/portfólio) |
Cada veredito traz `evidence_level ∈ {observed_only, statistically_sustained}`: **observado** = a desigualdade vale numa execução; **sustentado** = vale com o critério estatístico do nível (§3).

## 3. Três níveis de veredito
1. **Unidade** (observado): função pura de `(A_R,A_O,A_C,F_R,F_O)` não arredondados → um dos 3 estados (nunca `INSUFFICIENT_EVIDENCE` salvo dados em falta). Serve de contagem descritiva, **não** de prova.
2. **Dataset** (10 unidades CV): diferenças pareadas `Δ_RO = A_R − A_O`, `Δ_OC = A_O − A_C`, `Δ^F_RO`. IC95 % unilateral com variância corrigida (Nadeau–Bengio).
   * `INFEASIBLE` (sustentado) se `UCB(Δ_OC) ≤ δ_A` (evidência de que o Original não supera o C4.5);
   * `SATISFIED` (sustentado) se `LCB(Δ_OC) > δ_A` e `LCB(Δ_RO) > δ_A` (e `LCB(Δ^F_RO) > δ_F` se exigida);
   * `NOT_SATISFIED` (sustentado) se `Δ_OC` sustentada e `UCB(Δ_RO) ≤ δ_A`;
   * caso contrário `INSUFFICIENT_EVIDENCE`.
3. **Portfólio** (datasets CONFIRM): taxa de datasets `SATISFIED` com IC (bootstrap sobre datasets), Wilcoxon unilateral sobre diferenças médias por dataset com Holm {H1,H2,H3}.
   Alegação «Reloaded > Original» só se: Holm-p < 0,05 **e** IC bootstrap inferior > δ_A^alegação **e** controlos (`~perm`, `~rand`, `~conv`, `~fe`) não reproduzem o ganho **e** não-inferioridade de complexidade/fidelidade. Caso contrário
   «não sustentada» (com potência declarada). Uma unidade ou seed isolada **nunca** sustenta superioridade geral.

## 4. Como cada condição é otimizada, medida, verificada e reportada
| Condição | Otimizada? | Onde | Medida | Verificada | Reportada |
|---|---|---|---|---|---|
| `A_R > A_O` | **sim**, só em validação: passo 2 da regra lexicográfica de B3 maximiza `A_R` (OOF, MLP por dobra) sujeito às restrições | CV interna do treino | teste externo, 1 vez | `hierarchy_contract` | quadro Δ (§5) + `generalization_gap = Δ_val − Δ_teste` |
| `A_O > A_C` | **não** (baselines congelados; é constante em relação à config do Reloaded) | — | validação (OOF) e teste | `hierarchy_contract` → `INFEASIBLE` | idem |
| `F_R > F_O` | **restrição** `F_R ≥ F_O − ν_F` + desempate; superioridade em fidelidade é H3 (secundária à accuracy) | CV interna | teste externo | `hierarchy_contract` | idem |
**Validação ≠ teste:** a hierarquia em validação (OOF interno) é critério de seleção; o veredito que conta é o do teste externo, avaliado **uma vez** com a configuração congelada
(`config_frozen_hash` registado antes de abrir o teste; sem hash → o teste recusa-se a correr). O optimismo de seleção é medido e publicado.

## 5. Quadro comparativo obrigatório (gerado só de resultados brutos; sem preencher valores em falta — `N/A` explícito)
| Métrica | C4.5 | TREPAN Original | TREPAN Reloaded |
|---|---|---|---|
| Accuracy real | A_C | A_O | A_R |
| Fidelidade ao MLP | F_C (suplementar) | F_O | F_R |
| Macro-F1 | … | … | … |
| Nº nós / profundidade / comp. médio das regras | … | … | … |
| Consultas ao oráculo | 0 | q_O | q_R |
+ linhas `Reloaded − Original (acc)`, `Original − C4.5 (acc)`, `Reloaded − Original (fid)` com IC, `hierarchy_status`, `evidence_level`, n de unidades válidas.

## 6. Invariantes anti-manipulação (verificados por código/testes)
1. Hashes de `trepan_original.py`, `c45_j48_tree.py`, tuning estrutural, contrato do oráculo iguais aos do manifest; falha ⇒ execução abortada.
2. Mesmo `split_hash` e mesmo índice de teste para todos os braços; `assert_disjoint`.
3. Teste externo inacessível antes de `config_frozen_hash`; `EvaluationProtocolGuard` regista 1 leitura.
4. Seeds, grelhas, datasets e margens vêm do manifest; nada escolhido depois dos resultados.
5. Nenhuma unidade removida por «difícil»; falhas ficam em `skipped` e entram na contagem `INSUFFICIENT_EVIDENCE`.
6. Rótulos reais nunca alterados; C4.5 nunca recebe o oráculo; Original nunca recebe bónus ontológico; **o Reloaded nunca usa rótulos reais na construção**.
7. Um fallback do Reloaded para Original nunca conta como vitória do Reloaded (`fallback_original=True`, empate ⇒ `NOT_SATISFIED` com `δ_A=0`).
8. **Equivalência:** em cada unidade o braço R∅ é comparado ao Original (E1–E9, bit-a-bit); `EQ_VIOLATION` bloqueia a execução dessa variante e é reportada (Doc. 7). Nenhuma melhoria independente da ontologia pode explicar um ganho de R_OWL (Δ_conhecimento e Δ_convencional controlam).

## 7. Interface do módulo (para o plano; ainda não implementado)
```python
class HierarchyStatus(str, Enum): SATISFIED, NOT_SATISFIED, INFEASIBLE, INSUFFICIENT_EVIDENCE
@dataclass(frozen=True) class HierarchyMetrics: acc_r, acc_o, acc_c, fid_r, fid_o   # floats não arredondados
def unit_status(m, *, delta_a=0.0, delta_f=0.0, require_fidelity=True) -> UnitVerdict
def dataset_status(units, *, alpha=0.05, delta_a=0.0, delta_f=0.0, require_fidelity=True) -> DatasetVerdict
def portfolio_status(datasets, *, alpha=0.05, delta_a_claim=0.01) -> PortfolioVerdict
```
Testes obrigatórios (Documento 6): empates ⇒ não satisfeito para δ_A=0; ausência de arredondamento (diferença 1e-12 detetada); `A_O ≤ A_C` ⇒ `INFEASIBLE`; `A_R ≤ A_O` ⇒ `NOT_SATISFIED`; cadeia completa ⇒
`SATISFIED`; dados em falta ⇒ `INSUFFICIENT_EVIDENCE`; propriedade metamórfica (permutar unidades não altera o veredito de dataset).

## 8. Expectativa realista (informação, não alvo)
No único contraste exploratório disponível (breast_cancer, 2 ontologias × 5 seeds, **já observado**; não usar para afinar nada): accuracy média `O = 0,9427`, `R0 = 0,9371`, `C = 0,9301` ⇒ `U2` verdadeira, `U1` falsa
(cadeia **não** satisfeita hoje). O Reloaded e o Original só diferem se a semântica intervier; o Original e o Reloaded Core são idênticos nestes dados. O objetivo B1–B4 é testar se alguma variante inverte `U1` de forma
sustentada; a resposta pode ser «não».
