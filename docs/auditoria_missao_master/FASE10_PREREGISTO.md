# Fase 10 — pré-registo: orçamento de consultas efetivo ao oráculo

Escrito antes da campanha. As fases 1 a 9 e os seus resultados não são alterados. O núcleo (`core/`) não é alterado. Código: `main` em `4a1221c` (inclui P20/P21) e o executor `validation/benchmark/run_fase10.py`.

## 1. Pergunta

Todas as conclusões sobre ganho semântico (`GANHO_SEMANTICO_OWL.md`) têm a mesma ressalva: com o mesmo orçamento configurado, os braços com OWL **contam** 0,57 a 5,16 vezes as consultas ao oráculo do braço sem OWL (mediana 2,22). Nos registos das fases 7 a 9, porém, as consultas de pertença (pontos sintéticos novos) do Original e do Reloaded O são quase iguais; a diferença está nas chamadas a `predict_proba` que o Reloaded faz sobre a amostra de decisão de cada nó (`_known_row_uncertainty`).

> Os braços com OWL (R, R7) recebem do oráculo mais informação **nova** do que o Original? E as consultas adicionais que contam influenciam alguma decisão da árvore?

## 2. Medição (sem alterar respostas)

O oráculo de cada árvore é envolvido por um registador (`_CountingRec`) que guarda, por chamada, o método e a impressão digital de cada linha. As respostas e as contagens do oráculo congelado ficam intactas. Por árvore:

- `pontos_novos_rotulados`: linhas pedidas a `predict` que não são linhas de treino nem foram pedidas antes;
- `pontos_novos_so_probabilidade`: linhas novas pedidas só a `predict_proba`;
- `reconsultas_probabilidade_de_pontos_conhecidos`: linhas de `predict_proba` que já eram conhecidas (treino ou já rotuladas);
- `reconsultas_rotulo_de_pontos_conhecidos`, `linhas_unicas_total`, `linhas_treino_consultadas`.

## 3. Braços e desenho

Pipeline das fases 7 a 9: MLP selecionado só no treino, amostrador conjunto, 3 réplicas, holdout de 25 %, sementes 42, 7, 123, 2024 e 11. Os checkpoints de tuning das fases 8 e 9 são reutilizados; o tuning é determinístico e um checkpoint inválido é recalculado.

| Braço | O que é |
|---|---|
| B, B7 | TREPAN Original (B7: no máximo 7 nós) |
| R, R7 | Reloaded O (atributos da OWL v2, sem mecanismos), como nas fases 5 a 9 |
| Rnp, R7np | R com um oráculo **sem** `predict_proba`. O Reloaded usa então incerteza «indisponível»; o resto fica igual |

Datasets: os 9 em que R − B foi medido nas fases 6 a 9 (iris, wine, breast_cancer, digits, adult, german_credit, phys_bmi, phys_kinetic, phys_power). São 45 unidades.

## 4. Hipóteses e critérios (decididos agora)

- **H10.1 (informação nova):** em cada unidade e orçamento, R não recebe mais pontos novos do que B: média das réplicas `pontos_novos_rotulados(R) ≤ 1,10 × pontos_novos_rotulados(B)`, e `pontos_novos_so_probabilidade(R) = 0`. Reporto a razão por unidade e dataset.
- **H10.2 (sem influência):** Rnp tem previsões iguais às de R em todas as instâncias de teste, em todas as unidades, réplicas e orçamentos; o mesmo para R7np e R7.
- **Integridade:** MLP, C4.5, B, B7, R e R7 com previsões iguais às das fases 8 (iris, wine, breast_cancer, digits, adult) e 9 (german_credit e físicos) na mesma unidade. Se falhar numa unidade, essa unidade não entra na conclusão.

**Decisão:** em cada dataset e orçamento onde H10.1 e H10.2 se verificam, a ressalva das comparações R − B e R7 − B7 em `GANHO_SEMANTICO_OWL.md` passa a dizer: «o braço com OWL recebeu um número de pontos novos igual ou inferior ao do Original; as consultas adicionais contadas são probabilidades de pontos já conhecidos e não influenciam as decisões». Onde não se verificam, a comparação fica marcada como confundida pela informação. Os números de accuracy e fidelidade não mudam: a fase só mede, não recalcula ganhos.

## 5. Fora do âmbito

Os braços com mecanismos (E, V, D, A) usam as probabilidades nas decisões (EFSR e consultas ativas) e não são avaliados aqui; a ressalva das consultas mantém-se para eles. O braço histórico `reloaded_owl_full` é registado (o executor regista todas as árvores), mas a sua comparação com `reloaded_core` usa outro amostrador e só é reportada descritivamente.

## 6. Teste de funcionamento (antes deste pré-registo)

iris, semente 42, 1 réplica; o iris não serviu para escolher nenhum critério. B, R, B7, R7, MLP e C4.5 iguais à fase 8. Rnp igual a R e R7np igual a R7. Pontos novos: R 4864, B 4791; as 5888 linhas extra do R são probabilidades de pontos conhecidos.
