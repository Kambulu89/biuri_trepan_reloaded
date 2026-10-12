# B1–B4 — Fase 0: auditoria e protocolo (para a tua aprovação)

**Nenhum código de produção foi alterado.** Esta pasta contém os 6 documentos pedidos. Base: `main` @ `1789c17`.

| Doc | Conteúdo |
|---|---|
| [01](01_AUDITORIA_TECNICA.md) | Fluxo real, implementações, diagnóstico B1–B4 **com correções**, fuga de dados, problemas por prioridade |
| [02](02_PROTOCOLO_EXPERIMENTAL.md) | DEV/SELECT/CONFIRM, datasets, unidades, oráculos, seleção aninhada, hipóteses, estatística, potência |
| [03](03_ARQUITETURA_PROPOSTA.md) | Alterações por módulo, B1/B2/B3/B4 especificados, observabilidade, agnosticismo, **decisões D-1…D-8** |
| [04](04_MATRIZ_ABLACAO.md) | R0–R7 + subvariantes, controlos (`~perm`, `~rand`, `~conv`, `~fe`), nomes opacos, estágios |
| [05](05_CONTRATO_SUPERIORIDADE.md) | `A_R > A_O > A_C`, `F_R > F_O`, estados, níveis de veredito, o que é otimizado/medido/verificado |
| [06](06_PLANO_IMPLEMENTACAO.md) | Fases, dependências, riscos, esforço, critérios |

## O que a auditoria mudou no diagnóstico anterior
1. **Correção (D1):** o «bónus semântico 5–9× o ganho de informação» misturava escalas (score normalizado menos IG cru). Reanálise: o termo semântico aditivo vale **0,83×** a componente estatística
   (mediana; excede-a em **12/30** nós, até 9× onde a estatística é fraca) — mesma ordem de grandeza, não dominância sistemática. Detalhe no Doc. 1 §3.1.
2. **Correção (D3):** o limiar 0,002 compara-se com a granularidade da **amostra de decisão** (1/200–1/426), não com `1/143` do teste.
3. Os ganhos negativos aceites (3/10) são **desenho** (tolerância 1,5 pp comprada com «coerência semântica»), não bug.
4. O braço Reloaded Core ≡ Original; o controlo `~perm` não controla o conteúdo das features nem a engenharia convencional; não existe braço de engenharia convencional.
5. **Validação por accuracy exige MLP reajustado em cada dobra interna**; sem isso a seleção pela hierarquia é enviesada.

## Conflitos que preciso que resolvas (CLAUDE.md) — ver Doc. 3 §10
* **D-1** «Nunca otimizar nem codificar para produzir a ordem Reloaded > Original > C4.5» ↔ novo requisito formal. Proposta: hierarquia = critério **pré-registado** de seleção em validação interna +
  relatório; continua proibido tocar nos baselines, usar o teste, escolher seeds/datasets a posteriori. Só altero o CLAUDE.md com a tua confirmação.
* **D-2** Tuning/oráculo «congelados» ↔ B1–B4 mudam o Reloaded. Proposta: manifest v2; Original/C4.5/tuning/oráculo continuam congelados por hash.
* Restantes (D-3…D-8): fallback para Original, Modo B (rótulos reais), `n_eff`, nomes opacos (sidecar), **lista de datasets CONFIRM e ontologias**, margens/grelhas.

## Honestidade sobre o objetivo
A hierarquia pretendida **não está satisfeita hoje** no único contraste exploratório (O 0,9427 > C 0,9301, mas R0 0,9371 < O). Com ≥ 12 datasets só se detetam efeitos ≳ 1,2–2,4 pp (Doc. 2 §7). O protocolo
foi desenhado para que o resultado, seja qual for, seja **defensável**: se nenhuma variante superar o Original de forma sustentada, o relatório dirá «não sustentada».

## Próximo passo (só com a tua aprovação)
Aprovar/ajustar: protocolo (Doc. 2), decisões D-1…D-8 e a lista de datasets CONFIRM. Depois: Fase 0.5 → Fase 1.
