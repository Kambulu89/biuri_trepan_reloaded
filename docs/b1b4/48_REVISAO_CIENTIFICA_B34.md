# Doc. 48 — Revisão científica da B3.4: H1 (conhecimento externo) vs H2 (semântica OWL expressiva) e auditoria dos cálculos de informação

Estado: `B34 = DESIGN_ONLY / NOT_EXECUTED`; manifest B3.4 `frozen = false`. Esta revisão **propõe** alterações ao Doc. 44 (histórico v1 preservado, §5); nada foi congelado nem executado. `M0_M6_CORE_INTEGRATION = NOT_AUTHORIZED`.

## 1. Resumo das conclusões
1. A B3.4 tal como desenhada (Doc. 44 v1) testa **H1** e **só H1**. Não pode, isolada, sustentar nada sobre axiomas, hierarquias ou inferências OWL (H2).
2. Os números de informação do Doc. 44 estão **corretos sob os pressupostos declarados** (§3), mas **medem a coisa errada**: a informação relevante para a regra é a informação mútua I(S;P) = 31,2 bits (p=48) e 42,2 bits (p=96), não os 111,7/302,2 bits da partição; e, mais grave, **o cálculo ignora que a partição é quase totalmente recuperável dos próprios dados de treino** no gerador do Doc. 44.
3. **Falha de desenho encontrada:** com cargas λ ~ U(0,60; 0,85) (Doc. 44), o agrupamento por correlação (braço I) recupera a partição com ARI ≈ 0,99 já com n = 60 e 1,00 com n ≥ 120. Nessas condições o conhecimento da partição é **redundante** com os dados (I(S;P | X) ≈ 0) e a B3.4 v1 **não podia testar H1** nos regimes M1–M6. Corrige-se com cargas fracas e uma porta de recuperabilidade (§5).
4. As previsões secundárias P1 (benefício não crescente com n) e P2 (ontologia incompleta ≤ completa) são **plausíveis mas não monótonas por necessidade**; deixam de ser critérios lógicos de rejeição (§4).

## 2. As duas hipóteses
| | H1 — benefício de conhecimento externo | H2 — benefício de semântica OWL expressiva |
|---|---|---|
| Afirmação | conhecimento **verdadeiro** sobre a organização dos atributos melhora a pesquisa m-of-n quando os dados são insuficientes | axiomas, hierarquias e inferências corretamente modelados e interpretados dão vantagem **adicional** à de uma partição simples |
| O que varia | presença/qualidade da partição | expressividade (equivalências, disjunções, restrições, ordem, limites, inferência) a **informação de agrupamento igual** |
| Mecanismo exigido | pesquisa por unidades (existe, `validation/`) | mecanismos que **usem** cada tipo de axioma (Doc. 47 §3); hoje inexistentes |
| Contraste decisivo | E (OWL) vs A–D, I e vs F, G, H (partição errada) | F (OWL expressiva) vs D (plana, grupos corretos) e E (grupos sem OWL) |
| Conclusão permitida | «conhecimento prévio verdadeiro ajuda (ou não) neste regime» | «a expressividade OWL ajuda (ou não)» |
| Conclusão **proibida** | semântica OWL, hierarquia, inferência | — |
**Pré-condição de H2** (`H2_MECHANISM_PRECONDITION`): para cada tipo de axioma testado deve existir um mecanismo efetivo; se F e D produzirem os mesmos factos em uso (traços idênticos), o contraste F–D é degenerado e H2 fica `UNTESTABLE`, não «refutada». **Não se conclui H2 a partir de resultados que testam apenas H1.**

### 2.1 Contrastes futuros (A–J) para distinguir H1 e H2
| Braço | Descrição | Informação prévia |
|---|---|---|
| A | TREPAN Original canónico | nenhuma |
| B | Original WIDE | nenhuma |
| C | Pesquisa genérica de interações (+ rastreio por pares com controlo de comparações múltiplas) | nenhuma |
| D | **Ontologia plana** com grupos corretos | partição (OWL) |
| E | **Mesmos grupos corretos sem OWL** (lista simples) | partição |
| F | **Ontologia OWL expressiva e documentada** (Doc. 47) | partição + axiomas |
| G | Ontologia expressiva com relações relevantes **neutralizadas** | parcial |
| H | Ontologia **permutada** (ao nível da relação) | informação errada |
| I | Relações **aleatórias estruturalmente comparáveis** (graus/tipos) | informação errada |
| J | Engenharia convencional de *features* | derivada dos dados |
* **D vs E** — representação OWL vs partição simples (esperado: igual, se o mecanismo só lê a partição).
* **F vs D/E** — valor adicional da expressividade (H2).
* **F vs G/H/I** — efeito das relações verdadeiras.
* Fidelidade: mesmo MLP/oráculo e mesmas instâncias; orçamentos (candidatos, literais, consultas, CPU, compilação, complexidade, memória) comparáveis ou diferenças reportadas **por modalidade**; o benefício da informação adicional relata-se à parte da eficiência algorítmica.

## 3. Auditoria matemática da informação prévia (Doc. 44 §4)
Script e evidência: `validation/benchmark/b3q_information_audit.py` → `docs/evidence/b1b4/b3q/information_audit.json`; recuperabilidade: `b3q_partition_recoverability.py` → `partition_recoverability.json`.

**Especificação explícita**
* **Espaço amostral** Ω: pares (P, {A,B}); depois, dados X e rótulos y gerados condicionalmente (fora destes cálculos).
* **Partição**: P = partição dos p = K·g atributos rotulados em K blocos **não rotulados** de tamanho g. |Ω_P| = p! / ((g!)^K · K!).
* **Prior** sobre P: **uniforme** (pressuposto). **Conceitos-alvo**: {A,B} = par não ordenado de blocos, uniforme sobre C(K,2), **independente** de P (pressuposto).
* **Suporte** da regra: S = A ∪ B (2g atributos).
* **Entropia**: Shannon, em bits. **Condicional**: H(S|P) = log₂ C(K,2). **Marginal**: pela simetria do prior sob permutações dos atributos, S é uniforme sobre os C(p,2g) subconjuntos → H(S) = log₂ C(p,2g) (verificado por enumeração exaustiva em p=8, K=4, g=2: 105 partições = fórmula; 70 suportes distintos = C(8,4), cada um obtido 9 vezes).
* **Informação mútua** I(S;P) = H(S) − H(S|P).
* **Independência pressuposta**: P ⊥ {A,B}; X e y não entram nestes cálculos incondicionais.

| | K=8, g=6 (p=48) | K=16, g=6 (p=96) |
|---|---|---|
| H(P) | 111,72 bits | 302,16 bits |
| H(S) | 36,02 | 49,15 |
| H(S \| P) | 4,81 | 6,91 |
| **I(S;P)** | **31,21** | **42,24** |
| fração de H(P) relevante para S | 28 % | 14 % |
| pares de atributos vs pares de conceitos (nível da pesquisa) | 1128 vs 28 (×40) | 4560 vs 120 (×38) |

**Achados**
* **F1 — corretos mas mal interpretados.** 36,0 → 4,8 bits (Doc. 44) coincide com H(S) → H(S|P), logo equivale a I(S;P) = 31,2. Os 111,7 bits da partição **não** são «informação sobre a regra»: só 28 % (p=48) / 14 % (p=96) são relevantes para o suporte.
* **F2 — bits ≠ custo algorítmico.** H(S) assume um pesquisador com prior uniforme sobre todos os subconjuntos de 12 atributos; nenhum algoritmo pesquisa assim. O ganho de pesquisa é indireto (pares candidatos reduzidos por ≈ 40× ao nível de conceitos); não se pode tratar a redução de bits como «eficiência algorítmica gratuita».
* **F3 — é conhecimento externo disponibilizado a E–H e N\_\*** (não a A–D, I, K): o contraste E vs A–D,I,K é um estudo de benefício de conhecimento externo; E vs J (informação igual) mede só o processamento.
* **F4 — redundância com os dados (falha de desenho).** A informação **líquida** é I(S;P | X_treino) ≤ I(S;P); se P for (quase) função de X, é ≈ 0. Medido (agrupamento por correlação, K conhecido, 100 réplicas, só X):

| cargas λ | n=60 | n=120 | n=240 | n=1000 |
|---|---|---|---|---|
| U(0,60; 0,85) (**Doc. 44 v1**) | ARI 0,99 | 1,00 | 1,00 | 1,00 |
| U(0,45; 0,65) | 0,67 | 0,97 | 1,00 | 1,00 |
| U(0,35; 0,55) | 0,31 | 0,66 | 0,96 | 1,00 |
| **U(0,25; 0,45)** (proposta) | **0,09** | **0,21** | **0,49** | 0,99 |
No gerador v1 a partição é recuperável sem ontologia em todos os n planeados ⇒ a B3.4 v1 testaria um conhecimento **redundante**. Com λ ~ U(0,25; 0,45) a partição é quase irrecuperável a n ≤ 120 e passa a recuperável perto de n = 1000 (regime B3.3, onde os genéricos igualaram a ontologia): é a dose-resposta correta. A correlação entre a média dos 6 indicadores de um conceito e o latente é ≈ 0,68 (indicador isolado ≈ 0,35): a regra continua aprendível ao nível do conceito.
* **F5 — fuga lexical.** Os nomes `tema_medida` do Doc. 44 revelam a partição a qualquer método que leia nomes. Proposta: nomes **opacos** (`x017`) nos dados, com os mesmos rótulos opacos nas entidades da ontologia (o matching continua por nome/rótulo, sem pistas conceptuais).
* **F6 — não-atribuição.** Dado I(S;P|X) > 0 só quando a partição não é recuperável, qualquer ganho de E sobre I (agrupamento por dados) e sobre D (rastreio por pares) é, por construção do gerador, **conhecimento externo** e deve ser rotulado assim.

## 4. Previsões secundárias (revisão explícita)
| Previsão (Doc. 44 v1) | Estado |
|---|---|
| P1 — ganho (E−F) não crescente em n (M1 ≥ M2 ≥ M3); «um ganho maior com mais dados contradiz o mecanismo» | **Rebaixada a observação descritiva.** O benefício esperado é maior com poucos exemplos, mas pode não ser monótono (ruído de amostragem, oráculo fraco, porta de recuperabilidade). Não entra no veredicto global nem rejeita H1. Reporta-se a curva. |
| P2 — ontologia incompleta ≤ completa | **Rebaixada a observação descritiva.** Uma ontologia mais completa pode juntar conhecimento verdadeiro mas irrelevante ou custoso; uma incompleta pode, por acaso, expor o suporte relevante. Não é critério de rejeição. |
| P3 — M0 sem ganho (controlo nulo) | **Mantida** como verificação do **protocolo** (erro de tipo I), não da hipótese. |
| P4 — E ≈ J (mesma partição sem OWL) | **Mantida como igualdade esperada**; se confirmada, a conclusão é «valor de conhecimento prévio», não de semântica OWL (H1, não H2). |
**Critérios de rejeição de H1 (únicos):** num regime, H1 é refutada se E não superar F e G, ou não superar nenhum método genérico forte de recursos comparáveis, ou se os controlos negativos mostrarem o mesmo ganho, ou se a porta de recuperabilidade/oráculo falhar (então `INSUFFICIENT_EVIDENCE`, não refutação).

## 5. Revisões propostas ao Doc. 44 (v2; **PROPOSTAS** até aprovação e antes de qualquer congelamento)
Histórico: **v1 = commit `66ad1a9`** (texto original preservado em `44_PROTOCOLO_B34_CONHECIMENTO_PREVIO.md`; uma nota de histórico foi acrescentada no fim do ficheiro sem alterar o corpo). As revisões abaixo só entram em vigor por decisão explícita, **antes** do congelamento experimental:
* **R1** — P1 e P2 passam a observações descritivas; retirar «P1 não contradita» do veredicto global (§4).
* **R2** — Declarar o estudo como **H1 only**; qualquer conclusão sobre OWL expressiva fica reservada a H2 (§2).
* **R3** — Cargas dos indicadores λ ~ U(0,25; 0,45) e **porta de recuperabilidade**: uma célula só é informativa se o ARI do agrupamento por dados (braço I) nas sementes de calibração for ≤ 0,5 (valor a **decidir pelos revisores**; corte inicial proposto, justificado pela tabela acima onde a partição passa de irrecuperável a recuperável entre n=120 e n=240; não validado); células com ARI > corte são `REDUNDANT_KNOWLEDGE` e não contam.
* **R4** — Nomes opacos (F5).
* **R5** — Texto de informação do Doc. 44 §4 substituído pela especificação do §3 deste documento (I(S;P), pressupostos, bits ≠ custo).
* **R6** — Acrescentar ao desenho a decomposição D/E/F de H2 **apenas como estudo futuro** (não executar na B3.4).
* **R7** — Orçamento de CPU: inalterado (≈ 7 CPU-h; limite 9), mais ≈ 0,05 CPU-h da porta de recuperabilidade.
O manifest `B34_EXPERIMENT_MANIFEST.DRAFT.json` mantém `frozen=false`; recebeu apenas o estado `DESIGN_ONLY / NOT_EXECUTED` e a lista de revisões pendentes, sem alterar parâmetros.

## 6. Decisões pendentes e custo
* **Custo da B3-Q já gasto:** auditoria DEV + cálculos + simulação de recuperabilidade < 0,1 CPU-h; sem benchmark.
* **Decisões:** (1) aprovar R1–R7 para um Doc. 44 v2 (ou rejeitar alguma); (2) decidir o corte de recuperabilidade e a política de células `REDUNDANT_KNOWLEDGE`; (3) designar os três revisores (domínio, ontologias, metodologia); (4) decidir se, antes da B3.4, se constrói uma ontologia DEV enriquecida com fontes (Doc. 47) para um estudo exploratório H2 — exige mecanismos novos em `validation/` (custo não estimado, requer autorização); (5) se se mantém a autorização das ≈ 7 CPU-h só para H1.
* **Não se executa** a B3.4 nem se implementam mecanismos; `core/` e baselines intocados.
