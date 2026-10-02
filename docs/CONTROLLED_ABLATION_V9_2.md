# Ablação controlada V9.2 — TREPAN Original vs TREPAN Reloaded

## Objectivo

A V9.2 compara TREPAN Original e TREPAN Reloaded com um protocolo pareado em que a única variável experimental permitida é a extensão semântica/ontológica.

Os dois braços usam obrigatoriamente:

- o mesmo conjunto de treino e o mesmo holdout bloqueado;
- o mesmo MLP Original como oráculo;
- a mesma seed raiz;
- o mesmo `min_sample`;
- o mesmo `max_queries`;
- o mesmo limite de nós e profundidade;
- o mesmo `max_n`, beam width e critérios estatísticos.

No braço OWL, `OriginalOracleProjection` apenas selecciona as colunas originais antes de consultar o MLP. Não existe treino de um MLP residual ou de um segundo professor. Assim, qualquer diferença entre os braços não pode ser atribuída a uma troca de oráculo.

## Ordem experimental

1. Criar split treino/teste estratificado.
2. Treinar o MLP Original somente no treino.
3. Executar o gate de saúde do oráculo somente por CV interna do treino:
   - MLP deve superar `DummyClassifier` por uma margem configurada;
   - MLP não pode ficar abaixo do C4.5-Nativo por mais que a margem configurada.
4. Ajustar a ontologia/`OntologyProcessor` apenas no treino.
5. Ajustar TREPAN Original e TREPAN Reloaded com o mesmo orçamento.
6. Congelar os dois modelos.
7. Consultar o holdout final uma única vez numa avaliação pareada.
8. Calcular diferenças OWL−sem-OWL.

## Estatística agregada

As repetições do mesmo dataset não são tratadas como observações independentes. Primeiro é calculada a diferença média dentro de cada dataset; bootstrap e Wilcoxon trabalham depois com **dataset como unidade de análise**.

Datasets cujo oráculo falha o gate são registados como `oráculo_invalido`, mas não entram no efeito agregado OWL.

## Estado do ambiente desta candidate

`owlready2/HermiT` não está disponível no runtime utilizado para construir esta candidate. Por isso, nesta ronda foi executado apenas o controlo sem OWL. O braço OWL está registado como `not_executed`; nenhum efeito OWL novo foi calculado.
