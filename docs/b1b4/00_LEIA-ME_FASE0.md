# B1–B4 — Fase 0 (revista): auditoria, protocolo e contrato de equivalência

**Nenhum código de produção foi alterado.** Base: `main` @ `1789c17`. Esta revisão incorpora a **correção arquitetural**: o Reloaded é o Original **mais** uma extensão ontológica.

| Doc | Conteúdo |
|---|---|
| [07](07_CONTRATO_EQUIVALENCIA.md) | **Contrato de equivalência** `Reloaded(Ω=∅) ≡ Original` (E1–E9), contrato de enriquecimento, testes, estado atual, atribuição de ganho |
| [01](01_AUDITORIA_TECNICA.md) | Fluxo real, implementações, diagnóstico B1–B4 (com errata) e §9 auditoria da equivalência |
| [02](02_PROTOCOLO_EXPERIMENTAL.md) | DEV/SELECT/CONFIRM, pistas T e M, hipóteses H0–H4, seleção aninhada, estatística, potência |
| [03](03_ARQUITETURA_PROPOSTA.md) | Despacho estrutural para o Original, B1–B4 como mecanismos ontológicos, decisões D-1…D-11 |
| [04](04_MATRIZ_ABLACAO.md) | Os seis grupos A–F, variantes R0–R7, controlos, Original na representação OWL, estágios |
| [05](05_CONTRATO_SUPERIORIDADE.md) | Hierarquia com `R_OWL`, estados, níveis de veredito, invariantes |
| [06](06_PLANO_IMPLEMENTACAO.md) | Fases (nova **1a: conformidade EQ**), riscos, testes |

## O que mudou com a correção arquitetural
1. **Regra central** (também em `CLAUDE.md` §2b): sem ontologia (ausente, rejeitada ou sem estrutura ativa) o Reloaded **é** o Original — em árvore, regras, m-of-n, **consultas ao oráculo**, previsões,
   fidelidade, accuracy e complexidade. Diferenças só se forem não semânticas, justificadas e previamente autorizadas (allowlist **vazia**).
2. **B1–B4 passam a ser só mecanismos ontológicos.** Retirei tudo o que melhorava o Reloaded sem ontologia: o «Modo B» (rótulos reais, γ), a poda genérica, qualquer «ganho do Core».
   A antiga «D-9» (ganho arquitetural independente) — que no meu texto anterior não tinha esse número — fica substituída: o objetivo é o **ganho adicional da ontologia**.
3. **Benchmark principal** reorganizado nos teus seis grupos: A C4.5 · B Original · C **Reloaded sem ontologia (controlo EQ)** · D Reloaded com ontologia · E controlo permutado/aleatório ·
   F engenharia convencional; mais **Original na representação OWL** para separar efeito da representação do efeito da orientação semântica (Δ_total = Δ_repr + Δ_guia). Pistas T (árvore) e M (MLP) separadas.
4. **Hierarquia:** `Accuracy_R_OWL > Accuracy_Original > Accuracy_C4.5` e `Fidelity_R_OWL > Fidelity_Original`, como objetivo pré-registado de seleção/aceitação, nunca imposto.

## Achado da auditoria de conformidade (importante)
Com o código atual, **sem informação ontológica** o Reloaded produz **as mesmas regras, previsões e nós** que o Original (6/6 sondas: iris, wine, breast_cancer × 2 seeds), **mas consulta o oráculo
2,6–3,1× mais** (p.ex. iris/42: 652 → 1712 amostras; 5 → 13 chamadas) por causa de `_known_row_uncertainty` (EFSR) em cada nó. Pelo teu critério isto é uma **violação do contrato de equivalência (V1)**;
não é «diferença justificada». Corrigir na **Fase 1a**, antes de B1. (Sonda e resultados: `docs/evidence/b1b4/`.)

## Estado das decisões (Doc. 3 §10)
Resolvidas: D-1 (hierarquia como objetivo formal), D-2 (congelamento/evolução), D-4 (sem Modo B), D-9 (nova). A confirmar: D-3, D-5, D-6, D-10 (allowlist vazia), D-11 (V1 = a corrigir).
**Pendentes de ti:** D-7 (lista de ≥ 12 datasets CONFIRM e autoria das ontologias) e D-8 (grelhas λ, δ_A, δ_F, ν_F).

## Honestidade sobre o objetivo
A hierarquia **não está satisfeita hoje** no único contraste exploratório (O 0,9427 > C 0,9301, mas R0 0,9371 < O). Com ≥ 12 datasets só se detetam efeitos ≳ 1,2–2,4 pp. Se nenhuma variante
superar o Original de forma sustentada, o relatório dirá «não sustentada».
