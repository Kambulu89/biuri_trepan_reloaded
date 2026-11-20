# Recálculo das execuções de produção do Breast Cancer depois de P21

As três execuções `prod_bc` (`scripts/biuri_cli.py train-production`, ARFF `breast_cancer_wisconsin_diagnostic_dados.arff`, alvo `diagnosis {M,B}`, OWL `breast_cancer_wisconsin_diagnostic_enriquecida_biuri.owl`, sementes 7, 42 e 123) foram feitas em 2026-11-11/12, entre 23:40 e 00:17, antes de `16e6be8`. Treinaram com as classes na ordem alfabética `B, M`. A correção P21 (`010619f`) usa a ordem declarada `M, B`. Os números destas execuções estão citados em `AUDITORIA_ALGORITMOS.md` (achado 3), em `PROBLEMAS_IDENTIFICADOS.md` (P01) e em `VALIDACAO_FUNCIONAL.md`; por isso foram recalculados. Nenhuma outra experiência foi recalculada.

## 1. Desenho

Para cada semente, a mesma linha de comando, os mesmos dados, a mesma OWL e a mesma semente, em duas versões do código:

1. **Controlo:** o código antigo (`af02d5d`), corrido hoje. Serve para separar o efeito da correção de qualquer diferença de ambiente.
2. **Recálculo:** o código corrigido (`8375c6c`, cujo código é o de `010619f`).

Sobre o recálculo:
- As execuções das sementes 42 e 123 registam `dirty=1`: no início havia no checkout um ficheiro de documentação ainda por registar (os scripts desta pasta). Nenhum ficheiro de código estava alterado.
- Desde `af02d5d`, as únicas diferenças do núcleo que tocam este percurso são a ordem das classes (P21) e o estado do gate (`16e6be8`, que só muda o rótulo do estado).
- O ARFF não tem atributos nominais e a OWL não tem relações declaradas (`derivedRelation` = 0).

As previsões foram extraídas dos modelos guardados (`production_bundle.joblib`) por `prod_bc_predicoes.py`, sempre convertidas em rótulos `M`/`B`. As probabilidades do MLP foram alinhadas pelo rótulo e não pela posição da coluna. Em todas as execuções, a accuracy e a fidelidade recalculadas a partir dessas previsões coincidem exatamente com as do relatório. A comparação foi feita por `prod_bc_comparar.py`, instância a instância, pelo índice da linha.

**Oráculo das comparações de fidelidade.** Em todas as execuções, a fidelidade do TREPAN Original e do Reloaded é medida contra o MLP de produção da própria execução (`teacher_id = mlp_original`; o Reloaded fica no espaço original). O `oracle_id` de cada um está na tabela. Original e Reloaded da mesma execução usam sempre o mesmo oráculo.

## 2. Resultados

| Semente | Execução | Instâncias de teste comuns com a original | Oráculo (MLP) | Acc. MLP | Acc. C4.5 | Acc. Original | Acc. Reloaded | Fid. Original | Fid. Reloaded | Δ fid. (R − O) | Nós Original | Nós Reloaded | Previsões mudadas nas instâncias comuns (MLP / C4.5 / Original / Reloaded) | Estado do gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 7 | original (2026-11-11/12) | — | `272f545de3fc2053` | 0,979 | 0,937 | 0,923 | 0,930 | 0,944 | 0,951 | +0,007 | 27 | 23 | — | SEMANTIC_CONTRIBUTION_SUPPORTED |
| 7 | controlo: código af02d5d, hoje | 143 de 143 | `272f545de3fc2053` | 0,979 | 0,937 | 0,923 | 0,930 | 0,944 | 0,951 | +0,007 | 27 | 23 | 0 / 0 / 0 / 0 | SEMANTIC_CONTRIBUTION_SUPPORTED |
| 7 | recálculo: código corrigido | 29 de 143 | `cc82ec7bafa88437` | 0,986 | 0,944 | 0,916 | 0,951 | 0,930 | 0,965 | +0,035 | 57 | 47 | 0 / 1 / 2 / 0 | SEMANTIC_CONTRIBUTION_SUPPORTED |
| 42 | original (2026-11-11/12) | — | `11fd425f34d1e47b` | 0,951 | 0,944 | 0,958 | 0,958 | 0,923 | 0,937 | +0,014 | 31 | 23 | — | SEMANTIC_CONTRIBUTION_SUPPORTED |
| 42 | controlo: código af02d5d, hoje | 143 de 143 | `11fd425f34d1e47b` | 0,951 | 0,944 | 0,958 | 0,958 | 0,923 | 0,937 | +0,014 | 31 | 23 | 0 / 0 / 0 / 0 | SEMANTIC_CONTRIBUTION_SUPPORTED |
| 42 | recálculo: código corrigido | 39 de 143 | `65f72d9e392b1a7b` | 0,958 | 0,937 | 0,944 | 0,937 | 0,958 | 0,937 | −0,021 | 27 | 29 | 3 / 1 / 1 / 3 | SEMANTIC_EFFECT_WITHOUT_FIDELITY_GAIN |
| 123 | original (2026-11-11/12) | — | `237e4a5d3441736e` | 0,965 | 0,958 | 0,895 | 0,923 | 0,916 | 0,930 | +0,014 | 29 | 25 | — | SEMANTIC_CONTRIBUTION_SUPPORTED |
| 123 | controlo: código af02d5d, hoje | 143 de 143 | `237e4a5d3441736e` | 0,965 | 0,958 | 0,895 | 0,923 | 0,916 | 0,930 | +0,014 | 29 | 25 | 0 / 0 / 0 / 0 | SEMANTIC_CONTRIBUTION_SUPPORTED |
| 123 | recálculo: código corrigido | 36 de 143 | `ab5d1f9aaae942b1` | 0,958 | 0,944 | 0,937 | 0,951 | 0,909 | 0,923 | +0,014 | 31 | 29 | 1 / 2 / 1 / 2 | SEMANTIC_CONTRIBUTION_SUPPORTED |

Acc. = accuracy face aos rótulos reais; Fid. = fidelidade ao oráculo indicado; Δ fid. = Reloaded − Original na mesma execução. As previsões mudadas contam-se só nas instâncias de teste comuns à execução original e à execução comparada.

Detalhe do gate de contribuição semântica:

| Semente | Original: impacto semântico, EFSR aceites/tentados | Recálculo: impacto semântico, EFSR aceites/tentados |
|---|---|---|
| 7 | 0, 0/15 | 0,129, 4/29 |
| 42 | 0,267, 4/15 | 0,200, 3/14 (estado `SEMANTIC_EFFECT_WITHOUT_FIDELITY_GAIN`: Δ fid. negativo) |
| 123 | 0,200, 3/15 | 0,133, 2/14 |

## 3. Leitura

1. **O ambiente reproduz as execuções originais.** Com o código antigo, as três sementes dão as mesmas instâncias de teste, o mesmo oráculo, previsões iguais em todas as instâncias e em todos os modelos, probabilidades iguais (diferença máxima 0) e o mesmo número de nós. Todas as diferenças abaixo vêm da correção.
2. **A mudança de ordem não alterou só a apresentação das classes.** A partição treino/teste é estratificada pelos rótulos. Com `M=0, B=1`, o `train_test_split` estratificado escolhe outras instâncias de teste. Isto foi verificado diretamente: com os rótulos em texto reproduz os índices antigos e com os códigos reproduz os novos, nas três sementes. Ficam em comum só 29, 39 e 36 das 143 instâncias. Com outro treino, o MLP é outro (outro `oracle_id`, outros pesos), e as árvores e o número de nós também mudam.
3. **Nas instâncias comuns** mudam 0 a 3 previsões de cada modelo. O MLP muda em 0, 3 e 1 instâncias; a maior diferença de probabilidade é 0,003 (semente 7), 0,992 (semente 42) e 0,670 (semente 123).
4. **Os números citados mudam:**
   - semente 7: o Reloaded tem 47 contra 57 nós do Original (antes 23 contra 27), e fidelidade +0,035 (antes +0,007);
   - semente 42: o Reloaded perde fidelidade (−0,021) e o gate dá `SEMANTIC_EFFECT_WITHOUT_FIDELITY_GAIN`;
   - semente 123: o Δ fid. mantém-se +0,014.
5. **P01:** o caso que o reproduzia (semente 7, impacto 0, 0 intervenções aceites, estado «SUPPORTED») é reproduzido exatamente pelo controlo com o código antigo; continua a ser um exemplo válido do defeito do gate antigo. Com o código corrigido, a mesma semente tem impacto 0,129 e 4 intervenções aceites, por isso já não é um exemplo do defeito. A correção do gate é lógica e está coberta por testes unitários (`tests/test_audit_semantic_contribution_gate.py`).

## 4. Observação para decisão

Com uma ordem de classes declarada, a partição estratificada passa a depender dessa ordem. Ambas as partições são estratificadas e válidas, e a anterior dependia da ordem alfabética, igualmente arbitrária. Se se quiser que a partição seja a mesma qualquer que seja a codificação das classes, basta estratificar pelos rótulos originais em vez dos códigos. É uma alteração de uma linha, mas mudaria de novo estes resultados e exige decisão; não a fiz.

## 5. Ficheiros

`evidencia/producao_bc_recalculo/`:
- `prod_bc_predicoes.py`, `prod_bc_comparar.py`, `run_prod_bc_recalc.sh`;
- `previsoes/`: previsões por instância das 9 execuções (original, controlo e recálculo);
- `comparacao.csv` e `comparacao.json` (instâncias com previsão mudada);
- `corrigido/`: relatórios e manifestos do recálculo;
- `controlo_af02d5d/`: manifestos do controlo;
- `logs/`: commit, estado do checkout, início, fim e código de saída de cada execução.

Os relatórios originais continuam em `evidencia/producao_bc/`, sem alteração.
