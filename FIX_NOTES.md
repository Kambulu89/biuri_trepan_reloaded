# Correção do erro de geração de explicações

## Causa

O código criava a variável `clases`, mas tentava ler `classes`. Isso produzia
`NameError: name 'classes' is not defined` ao gerar a análise por categoria.

O mesmo lapso existia no relatório de carregamento da ontologia, embora pudesse
ficar oculto até essa operação ser usada.

## Alterações mínimas

- Uniformização de `clases` para `classes` em dois pontos de
  `gui/biuri_app_complete.py`.
- Inclusão de dois testes de regressão em
  `tests/test_gui_explanation_regressions.py`.
- Nenhum algoritmo de treino, extração TREPAN, interface ou fluxo existente foi
  alterado.

## Validação efetuada

- Geração de explicação com `ClassA` e `ClassB`: OK.
- Verificação estática do relatório da ontologia: OK.
- Compilação de todos os ficheiros Python do projeto: OK.
- Carga dos 12 OWL com o `owlready2` incluído no projeto: OK.
