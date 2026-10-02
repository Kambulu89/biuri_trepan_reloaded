# Relatório de testes V9

Data: 2026-09-02

## Resultado consolidado

- **166 testes** da suíte `tests/` aprovados, além de **5 subtestes**.
- **4 regressões legadas não gráficas** da raiz aprovadas.
- **32 testes `unittest` críticos** aprovados.
- **9 regressões específicas da V9** aprovadas: oito no módulo dedicado e uma
  para propagação de CFs no caminho Original/Residual do Reloaded.
- **144 ficheiros Python** compilados sem erros.
- Verificação estática fatal (`E9`, `F63`, `F7`, `F82`) aprovada nos ficheiros
  alterados.

## Cobertura da melhoria da árvore substituta

- filtro de confiança e validade no professor;
- classificação A/B/C;
- gate real de densidade e consulta ao vizinho mais próximo;
- teto do peso agregado dos CFs;
- separação dos professores e feature spaces Original/Reloaded;
- bloqueio explícito de C4.5 na melhoria de substitutos;
- espelhamento do Reloaded quando não existe OWL aceite;
- controlo e candidata com a mesma semente;
- seleção e aceitação em holdouts internos diferentes;
- rejeição de uma configuração que vence na seleção mas falha na auditoria;
- ausência de acesso ao teste final;
- não reutilização do lote CF interativo genérico;
- desativação real do lote CF automático no controlo Reloaded;
- propagação de `extra_X`, `extra_y` e `extra_weights`;
- ligação estática do diálogo, worker, estados e visualização da GUI.

## Smoke test ponta a ponta

Foi treinado um professor real e extraído um TREPAN Original num conjunto de
desenvolvimento controlado. A opção gerou CFs, selecionou uma configuração,
refez o par controlo/candidata no holdout de aceitação e rejeitou corretamente a
candidata sem ganho.

| Campo | Resultado |
|---|---|
| Ajuste da configuração | 108 amostras |
| Seleção da configuração | 36 amostras |
| Ajuste antes do gate | 144 amostras |
| Holdout de aceitação | 36 amostras |
| Estado | `rejected_by_quality_gate` |
| Blockers | ganho real; contribuição contrafactual |
| Teste usado na seleção | Não |
| Teste final avaliado | Não |
| Árvore existente preservada | Sim |

Este smoke test valida o fluxo e o comportamento fail-safe; não é um resultado
de benchmark nem evidência de superioridade.

## Limitação do ambiente gráfico

Os dois testes que importam widgets Qt dentro de `tests/`, bem como os scripts
gráficos legados da raiz, não puderam ser recolhidos neste contentor porque a
biblioteca de sistema `libEGL.so.1` não está instalada. O problema ocorre ao
importar `PyQt6.QtWidgets`, antes da execução do código do projeto.

Como compensação verificável neste ambiente:

- todos os módulos da GUI compilam;
- a árvore sintática do diálogo, worker e aplicação foi validada;
- as ligações do diálogo, opções, contextos sem teste e rótulos das duas árvores
  melhoradas possuem regressões estáticas aprovadas;
- `requirements-win.txt` mantém PyQt6 e o runtime Qt fixados para Windows.

## Avisos não bloqueantes

Alguns testes de MLP e regressão logística usam poucas iterações para terminar
rapidamente e emitem `ConvergenceWarning`. Quatro testes legados devolvem
booleano em vez de usar apenas `assert`, produzindo `PytestReturnNotNoneWarning`.
Nenhum destes avisos causou falha.

## Benchmark confirmatório

O benchmark confirmatório bloqueado **não foi executado**. A V9 não abre o teste
para escolher ou aprovar a melhoria. Uma futura execução confirmatória deverá
ocorrer uma única vez, depois de congelados datasets, ontologias, seeds,
hiperparâmetros e critérios estatísticos.

## Validação do pacote

- `unzip -t`: nenhum erro no conteúdo comprimido.
- Cópia extraída: 144 ficheiros Python recompilados.
- Cópia extraída: 21 testes críticos de melhoria, serviço e roteamento Reloaded
  aprovados.
- SHA-256 do ZIP: será registado na entrega depois do empacotamento final da
  própria documentação.
