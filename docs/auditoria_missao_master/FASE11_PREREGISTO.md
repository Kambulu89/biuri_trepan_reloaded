# Fase 11 — pré-registo: complexidade igual (P07) e consultas efetivas dos braços com mecanismos (P02)

Escrito antes da campanha. As fases 1 a 10 e os seus resultados não são alterados. O núcleo (`core/`) não é alterado. Executor: `validation/benchmark/run_fase11.py`.

## 1. Perguntas

**A. Complexidade (P07).** Nas fases 6 a 9, a comparação com o C4.5 não controla o tamanho das árvores:
- com 7 nós, o C4.5 crescia sem limite (até 169 nós no digits);
- com a estrutura do tuning, o TREPAN chega a ter 2,17 vezes mais nós do que o C4.5 (mediana, fase 1).

> Com a mesma complexidade máxima, o TREPAN Original e o Reloaded O superam o C4.5 em accuracy e em fidelidade?

**B. Consultas efetivas (P02).** A fase 10 mostrou que as consultas a mais do R e do R7 são probabilidades de pontos conhecidos, sem efeito nas decisões. Isso foi medido com o MLP selecionado no treino e sem os braços com mecanismos.

> Os braços com mecanismos (E: EFSR original; D: Reloaded completo; A: consultas ativas) recebem mais pontos novos do que o Original? E com o MLP por omissão (fase 6), a conclusão da fase 10 sobre R e R7 mantém-se?

## 2. Braços

Pipeline das fases 7 a 10: MLP selecionado só no treino, amostrador conjunto, 3 réplicas, holdout de 25 %, sementes 42, 7, 123, 2024 e 11. O tuning é reutilizado dos checkpoints das fases 8 e 9, e o registador do oráculo é o da fase 10.

| Braço | O que é |
|---|---|
| B, R | Original e Reloaded O, estrutura do tuning |
| B7, R7 | idem, no máximo 7 nós |
| **Bc, Rc** | idem, no máximo N nós, com N = min(nós do C4.5 canónico da mesma unidade, teto do tuning). O C4.5 é ajustado só no treino, por isso N não usa o teste |
| **c45_d2** | C4.5 canónico (mesma poda e parâmetros) com profundidade máxima 2, ou seja, no máximo 7 nós em divisões binárias |
| E, V, D, A | Reloaded com mecanismos (fases 5 e 9); servem só para medir consultas |

Execução adicional com o **MLP por omissão** (`--mlp omissao`, pipeline da fase 6, checkpoints da fase 6) nos 4 datasets da fase 6 (german_credit, phys_bmi, phys_kinetic, phys_power): braços B, R, B7, R7, Rnp e R7np.

Datasets do pipeline principal: os 9 da fase 10 (45 unidades). Execução com o MLP por omissão: 20 unidades.

## 3. Contrastes e critérios (decididos agora)

Estatística das fases 7 a 10: a unidade é a semente; as réplicas são promediadas dentro da unidade; IC 95 % por bootstrap percentil de 5 diferenças emparelhadas (20 000 reamostragens); sustentado = limite inferior > 0; observado = média > 0. Sem correção para comparações múltiplas.

**A. Complexidade igual**
- Ao tamanho do C4.5: H1c accuracy Rc − Bc; H2c accuracy Rc − C4.5; H3c accuracy Bc − C4.5; H4c fidelidade Rc − Bc; H5c fidelidade Rc − C4.5.
- A 7 nós: H2d accuracy R7 − c45_d2; H3d accuracy B7 − c45_d2; H5d fidelidade R7 − c45_d2. H1 e H4 são os de R7 − B7.
- Hierarquia sustentada, observada e atribuível com as definições da fase 6. A atribuição exige H1 e H4 sustentados; aqui não há controlo de OWL permutada, por isso «atribuível» não se aplica e reporta-se só sustentada e observada.
- **Veredicto principal** sobre os datasets elegíveis das fases 7 e 8 (MLP selecionado − C4.5 > 0,02 só no treino): wine, breast_cancer, digits, german_credit, phys_bmi, phys_power. Todos os 9 são reportados.
- Verificação de construção: o número de nós de Bc e Rc nunca passa N, e o de c45_d2 nunca passa 7.

**B. Consultas efetivas**
- H11.1: por unidade, pontos novos de E, D e A ≤ 1,10 × pontos novos de B (média das réplicas), e `pontos_novos_so_probabilidade` = 0. V é reportado à parte: o lote novo do V é, por desenho, informação nova.
- H11.2 (MLP por omissão): H10.1 e H10.2 da fase 10 para R e R7: pontos novos R/B ≤ 1,10 e Rnp ≡ R instância a instância.

**Integridade:** MLP, C4.5, B, B7, R e R7 com previsões iguais às das fases 8 e 9 (pipeline principal) e da fase 6 (MLP por omissão). Uma unidade que falhe não entra nas conclusões.

**Decisão:**
- **P07** passa a «medido com complexidade igual», com o resultado obtido.
- **Ressalva das consultas em `GANHO_SEMANTICO_OWL.md`:** atualizada para E, D e A e para as linhas da fase 6, nos datasets e orçamentos onde H11.1 e H11.2 se verificam. Onde não se verificam, a comparação fica marcada como confundida pela informação.

## 4. Teste de funcionamento (antes deste pré-registo)

iris, semente 42, 1 réplica:
- B, R, B7, R7, C4.5 e MLP iguais à fase 8.
- c45_d2: 5 nós. Bc = B7 porque o C4.5 desta unidade tem 7 nós.
- Pontos novos: B 4791, E 4877, D 5183, A 4581, V 6895 (o lote novo).
- Nenhum ponto novo só por probabilidade.

Depois do teste corrigi o limite de Bc e Rc para nunca ultrapassar o teto do tuning (mínimo dos dois). No teste o limite não mudava, porque o C4.5 tinha 7 nós e o teto do tuning era 31.
