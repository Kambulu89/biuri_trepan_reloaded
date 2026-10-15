# Documento 14 — Plano da Fase 1b: duplicação de consultas no espelhamento ontológico (V2)

Estado: **executado — ver Doc. 15** (relatório da Fase 1b).

## 1. Problema
Com `mirror_when_no_semantic_effect=True` (por omissão) e ontologia com estrutura ativa mas sem efeito mensurável (`ontology_influenced_splits==0` e `error_focused_interventions_accepted==0`), `fit` reexecuta o Original no mesmo oráculo depois da tentativa semântica: o estado final é o do Original, mas o oráculo recebe **tentativa semântica + Original completo** (≈2× consultas, a que se somam as projeções). Teste: `test_v2_mirror_oracle_stream_equals_original_known_violation` (xfail estrito).

## 2. Objetivos (D-16 e instruções de encerramento da 1a)
1. Inventariar todas as situações em que o `mirror` duplica consultas (por combinação: projetor, pesos, grupos, relatedness, domínio, EFSR on/off, `max_queries`, ontologia que passa/ não passa o gate).
2. Distinguir quatro níveis: ontologia **carregada** → **válida** (gate) → **estrutura semântica ativa** → **enriquecimento efetivamente utilizado**; definir para cada um o contrato (≡ Original ou Original+extensão).
3. Equivalência ao Original quando não existe efeito ontológico, **sem** consultas duplicadas.
4. Preservar o comportamento legítimo quando o conhecimento está efetivamente ativo (âncoras `golden_effective*.json` **não se alteram**).
5. Não eliminar consultas necessárias à construção semântica.
6. Qualquer otimização preserva resultados e rastreabilidade (auditoria `pre_mirror_*`, motivo do espelhamento).
7. Testes específicos escritos **antes** de remover o `xfail`.

## 3. Hipóteses de desenho (a avaliar; sem decisão)
* **H-A (decisão antecipada):** decidir "sem efeito" antes de consumir o oráculo, com informação puramente estrutural (pré-voo sem consultas) e delegar. Risco: o efeito mensurável só se conhece depois de construir a árvore.
* **H-B (reutilização exata):** manter a construção semântica e, no caso sem efeito, reutilizar a **mesma sequência de consultas** do Original (cache chaveada por conteúdo exato das amostras e semente) em vez de reexecutar. Exige provar que os pedidos do Original são subconjunto/identidade dos já feitos; caso contrário não é exato.
* **H-C (não duplicar a contabilidade):** se a duplicação é inevitável, torná-la **explícita e auditável** (contadores separados "tentativa semântica" vs "Original"), de modo que o fluxo do Original isolado seja igual ao do Original. Só aceitável se o utilizador/protocolo definir o custo da tentativa semântica como parte do enriquecimento (decisão científica).
* **H-D:** `mirror` desligado por omissão no Reloaded do benchmark (já `False` no braço D) e restrito a produção/GUI — opção de configuração, não solução.

## 4. Testes a escrever primeiro
Spy de oráculo com hash/sequência; casos 1–4 (ausente, rejeitada, válida sem estrutura, válida com enriquecimento efetivo e **com** / **sem** efeito mensurável); orçamentos aleatórios; várias seeds; binário/multiclasse; verificação de que a contagem total de consultas do Reloaded espelhado ≤ Original + consultas semânticas auditadas; âncoras inalteradas; `mirror` com `max_queries` esgotado; propriedade: estado final ≡ Original.

## 5. Critérios de entrada/saída
Entrada: Doc. 13 aceite; CI verde; decisão científica sobre H-A/H-B/H-C. Saída: `xfail` removido por correção real, E4 exato para todos os casos sem efeito ontológico, âncoras inalteradas, agnosticismo e CI verdes, relatório de fecho. Custo estimado: < 5 CPU-h (DEV, sintético + 4 datasets), dentro do teto Stage A (40 CPU-h).
