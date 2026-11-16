# Auditoria completa do repositório

Base auditada: commit `df045ee` do ramo `main` de `Kambulu89/biuri_trepan_reloaded`. Este documento é o índice da auditoria; os detalhes estão nos documentos referidos.

## 1. Inventário

| Área | Conteúdo | Dimensão |
|---|---|---|
| `core/` | algoritmos, pipelines, gates, ontologia, benchmark | 108 ficheiros, ~34 600 linhas de Python no nível superior |
| `gui/` | aplicação PyQt6 | 30 ficheiros, ~12 100 linhas; `biuri_app_complete.py` tem 5773 |
| `counterfactuals/` | explicações contrafactuais | 164 ficheiros |
| `validation/` | benchmarks, estudos e controlos | 176 ficheiros |
| `tests/` | suite pytest | 175 ficheiros na origem; 1772 testes |
| `docs/` | documentação e evidência histórica | 4797 ficheiros (62 MB em `docs/evidence`) |
| `results/`, `experiments/` | resultados e estudos anteriores | 222 + 56 ficheiros |
| Raiz | mais de 60 relatórios e JSON de validação de versões anteriores | — |

Stack: Python 3.11/3.12, scikit-learn (MLP `MLPClassifier`), owlready2 + HermiT, PyQt6. Não usa TensorFlow nem PyTorch. Histórico Git: 277 commits acessíveis, a maioria de sessões de IA anteriores.

## 2. Arquitetura real

Há três caminhos de execução que não partilham configuração:

1. **Benchmark** (`core/benchmark/`, `validation/benchmark/run.py`): braços A–F, oráculo congelado, tuning estrutural sem ontologia. É o mais rigoroso e o que usei para o baseline.
2. **Produção** (`core/production_training.py`, `scripts/biuri_cli.py`): Original e Reloaded lado a lado, sem braço sem ontologia e sem controlos por omissão.
3. **GUI** (`gui/biuri_app_complete.py` + `core/trepan_reloaded_extractor.py`): caminho próprio, com o extrator de 5109 linhas.

## 3. Estado por funcionalidade

| Funcionalidade | O que devia fazer | O que faz | Prioridade | Detalhe |
|---|---|---|---|---|
| Carregamento CSV/ARFF | ler dados e inferir tipos | funciona; um defeito de inferência corrigido | Média | P09 |
| MLP | oráculo treinado só no treino | funciona; não é reproduzido bit a bit fora do ambiente numérico fixado | Média | P13 |
| C4.5 | baseline supervisionado fiel | é um C4.5 genuíno | — | `AUDITORIA_ALGORITMOS.md` §2 |
| TREPAN Original | extração por consultas ao oráculo | estrutura fiel, com desvios; pré-seleção de atributos dependente da escala | Alta | P04 |
| TREPAN Reloaded | Original + mecanismos ontológicos | a ontologia participa no algoritmo; sem ganho medido; orçamento 2,2 vezes maior | Crítica | P02, P06 |
| Gates de ontologia | aceitar só conhecimento útil | o gate de contribuição declarava sucesso sem efeito semântico | Crítica | P01 (corrigido) |
| Controlos negativos | atribuir ganhos à ontologia | existem no benchmark; ausentes por omissão na produção | Crítica | P03 |
| Manifest e congelamento | fixar a referência | 5 de 11 ficheiros derivaram | Alta | P05 |
| Métricas | accuracy, F1, fidelidade | corretas (700 valores recalculados) | — | `VALIDACAO_FUNCIONAL.md` |
| Ontologias | TBox consistentes | 41 consistentes; só uma enriquecida para dados reais | Média | `AUDITORIA_ONTOLOGIAS.md` |
| Testes | prevenir regressões | 1742 aprovados; cobertura 72,1 % | Média | P14 |

## 4. Declarações anteriores verificadas

| Declaração na documentação do projeto | Verificação |
|---|---|
| «1732 passed» | Confirmado: 1732 aprovados, 40 ignorados, antes das minhas correções |
| Hierarquia conjunta não demonstrada | Confirmado de forma independente: 0 de 20 unidades |
| Reloaded sem ontologia ≡ Original | Confirmado nas 20 unidades |
| Ganho com ontologia igual ao da permutada (Breast Cancer) | Coerente: a real não supera a permutada com intervalo acima de zero em nenhum dataset |
| Baselines «congelados por hash» | **Não confirmado:** 5 ficheiros diferem do manifest |
| C4.5 nativo, não CART | Confirmado por leitura e pela suite |
| Rácio de consultas Reloaded/Original 2,14 (Doc. 30) | Coerente: medi 2,2 |

## 5. Documentos desta auditoria

`PROBLEMAS_IDENTIFICADOS.md` · `PLANO_CORRECCAO.md` · `CORRECCOES_IMPLEMENTADAS.md` · `VALIDACAO_FUNCIONAL.md` · `AUDITORIA_ALGORITMOS.md` · `AUDITORIA_ONTOLOGIAS.md` · `PROTOCOLO_EXPERIMENTAL.md` · `RESULTADOS_COMPARATIVOS.csv` · `ANALISE_ESTATISTICA.md` · `RELATORIO_FINAL.md` · `evidencia/`

## 6. O que não foi auditado

Contrafactuais (`counterfactuals/`), a GUI em uso manual, a execução em Windows, o otimizador de MLP (`core/mlp_optimizer.py`), os estudos em `validation/experimental/` e a evidência histórica em `docs/evidence/` (não reverifiquei resultados antigos, só reproduzi o baseline atual).
