# Fase 5 — ablação e correção do Reloaded: pré-registo

Escrito e registado em commit **antes** da campanha. Um teste de funcionamento do executor foi feito numa unidade (iris/7, 1 réplica) fora da evidência; serviu para descobrir o ponto da secção 2.2 e não entra na análise. Resultados em `FASE5_RESULTADOS.md`.

## 1. Pergunta

Na fase 4, o Reloaded completo (D) ficou abaixo do mesmo espaço sem mecanismos (O) no iris e no breast_cancer. Esta fase identifica que mecanismo causa a perda e testa duas correções que já existem no código, escolhendo uma por regra fixada aqui. A correção escolhida fica congelada para a fase 6, em datasets novos.

## 2. Braços

Executor `validation/benchmark/run_fase5.py`. Espaço enriquecido pela OWL v2 (exceto B), amostrador conjunto, 3 réplicas.

### 2.1 Ablação

| Braço | λ | Refinamento focado no erro (EFSR) | Consultas ativas | Score |
|---|---|---|---|---|
| B | — | — | — | Original, espaço original (integridade) |
| O | 0 | não | não | ganho estatístico |
| W | 1 | não | não | peso de ganho + Gain_Reloaded (profundidade ontológica, cobertura de erro) + candidatos m-of-n por grupos |
| E | 1 | sim | não | como W |
| A | 1 | não | sim | como W |
| D | 1 | sim | sim | como W (= D2 da fase 3; integridade) |

### 2.2 Porque E e A têm λ = 1

No teste de funcionamento, E e A com λ = 0 deram árvores idênticas a O: o EFSR e as consultas ativas só atuam com a estrutura semântica, que só é passada com λ > 0. Por isso cada um é medido como acréscimo sobre W: W − O (canal de score), E − W (EFSR), A − W (consultas ativas).

### 2.3 Correções candidatas (já existentes em `core/trepan_reloaded_historical.py`, `semantic_calibration`)

| Braço | O que faz |
|---|---|
| T (desempate) | score = s + τ·clip(q, −1, 1), com τ = 1/√n_real; a semântica só reordena candidatos cujo ganho estatístico difere menos de 2τ |
| L (limitada) | score = s + clip(m, −0,2·s, +0,2·s); o contributo semântico nunca passa de 20 % do ganho estatístico |
| Tq, Lq | T e L com a ontologia permutada (controlo) |

T e L têm EFSR e consultas ativas desligados. O parâmetro λcap = 0,2 de L é o valor por omissão do código; não é ajustado.

## 3. Unidades

iris, wine, breast_cancer × sementes 42, 7, 123, 2024, 11 (15 unidades), 3 réplicas. Limite de 5400 s por unidade.

## 4. Análise

Medida por unidade: média das 3 réplicas. Diferença emparelhada e intervalo a 95 % (bootstrap percentil, 20 000 reamostragens das 5 unidades).

**Diagnóstico (descritivo):** W − O, E − W, A − W, D − O, em accuracy e fidelidade. Um mecanismo «prejudica» num dataset se o intervalo ficar abaixo de zero numa métrica e não acima na outra.

**Integridade:** B tem de reproduzir as fases 2 a 4, e D a D2 da fase 3. Se não reproduzirem, a fase é inválida.

## 5. Regra de escolha da correção (congelada)

1. Candidatas: W, T, L.
2. **Elegível:** em nenhum dos três datasets o intervalo de X − O fica abaixo de zero, em accuracy nem em fidelidade.
3. Entre as elegíveis, escolhe-se a de maior média de (Δaccuracy + Δfidelidade)/2 face a O, calculada sobre os três datasets.
4. Se nenhuma elegível tiver essa média > 0, o «Reloaded corrigido» é **O** (a ontologia entra só como atributos, sem mecanismos). Empate (diferença < 0,001): fica a de menos mecanismos, pela ordem O, T, L, W.
5. A escolha é registada em `FASE5_RESULTADOS.md` e congelada para a fase 6, sem ajustes.

**Meta do passo:** Reloaded corrigido ≥ Original (B). Critério de não-inferioridade observada, por dataset: diferença média ≥ 0 e limite inferior do intervalo ≥ −0,02, em accuracy e em fidelidade.

Atribuição (descritiva nesta fase): T − Tq e L − Lq.

## 6. Limites conhecidos

- DEV, datasets já observados; a escolha da correção é feita nestes dados e por isso só a fase 6, em datasets novos, pode confirmá-la.
- Com 5 unidades por dataset, a não-inferioridade só é «observada»; a margem de 0,02 é larga.
