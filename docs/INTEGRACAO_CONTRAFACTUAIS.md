# Integração contrafactual — auditoria técnica

## Âmbito

Esta versão incorpora as capacidades contrafactuais úteis de `trepa_reloaded (2)`
na arquitectura de `trepa_reloaded_contrafactuais_dataset_carregado`. Toda a
análise continua limitada ao dataset carregado na sessão; os datasets de
experiência não são usados pela interface interactiva.

## Matriz de integração

| Capacidade da versão de origem | Situação anterior no destino | Decisão |
|---|---|---|
| DiCE e fallback | Já implementado no motor unificado | Reutilizado; não duplicado |
| CLEAR | Já implementado no motor unificado | Reutilizado; não duplicado |
| CoGS | Já implementado no motor unificado | Reutilizado; não duplicado |
| LORE-Local | Já implementado no motor unificado | Reutilizado; não duplicado |
| LORE-Global/local por árvore | Já implementado no motor unificado | Reutilizado; não duplicado |
| Restrições, plausibilidade, esparsidade e robustez | Já implementado e mais completo | Preservado |
| Protocolo formal P4–P7 / concordância MLP–árvores | Coberto pelo protocolo P1–P8 de transferência | Preservado |
| Regras contrafactuais globais para C4.5/TREPAN/Reloaded | Ausente | Implementação única acrescentada |
| Agregação, filtro de fragilidade e estruturas audit/fairness/OWL | Ausente | Acrescentados sem duplicar os extractores |
| Árvore explicativa construída com contrafactuais | Ausente | Acrescentada com treino e avaliação corrigidos |
| L1, L2, L1 ponderada, L0 normalizada, Mahalanobis e margem | Parcial | Avaliação formal acrescentada ao resultado local |
| Comparação estatística entre métodos | Ausente | Wilcoxon e Cohen d quando existem pares suficientes |
| Interface e exportação de regras/árvore/métricas | Parcial | Acrescentadas à aba responsiva e aos três formatos |

## Correcções em relação à versão de origem

1. Os três módulos quase idênticos de regras globais foram substituídos por um
   único módulo parametrizado para qualquer árvore sklearn do projecto.
2. A comparação de caminhos considera condições adicionadas, removidas,
   invertidas e mudanças de limiar. Regras diferentes nunca têm custo zero por
   omissão de uma feature.
3. O valor `1/(1+n_alterações)` é identificado como `symbolic_stability_proxy`;
   não é apresentado como robustez empírica.
4. A margem probabilística funciona em problemas multiclasse: probabilidade da
   classe prevista menos a alternativa mais forte.
5. A árvore CF não é treinada apenas com uma instância e poucos CFs. Usa uma
   vizinhança real do dataset rotulada pelo oráculo, reserva holdout sempre que
   possível e acrescenta somente CFs válidos com peso explícito.
6. A árvore original nunca é modificada. O resultado guarda o modelo novo apenas
   para visualização em memória e remove-o automaticamente de JSON/CSV/Markdown.
7. Resultados antigos são rejeitados se pertencerem a outro dataset, modelo ou
   espaço de features.

## Pontos de acesso

- **Gerar contrafactual**: explicação local e avaliação formal.
- **Regras contrafactuais globais**: disponível para TREPAN Original, TREPAN
  Reloaded e C4.5-Nativo.
- **Construir árvore CF**: activa após existir pelo menos um candidato local válido.
- **Visualizar árvore CF**: abre a árvore construída na visualização interactiva.
- **Avaliar transferência**: mantém o protocolo P1–P8 no dataset activo.
- **Exportar resultados**: JSON, CSV e Markdown para qualquer um dos resultados.

## Dependências

Não foram acrescentadas dependências. A implementação usa `numpy`, `scipy` e
`scikit-learn`, já declarados no projecto. O `requirements-win.txt` foi preservado
para não introduzir conflitos com Python/Windows/TensorFlow.

## Revisão da interface contrafactual

A aba contrafactual utiliza agora três breakpoints internos, sem depender da
resolução física do monitor:

- **Desktop:** parâmetros em quatro colunas, seis ações e seis indicadores numa
  única linha; explicação e tabelas lado a lado.
- **Intermédio:** parâmetros em duas colunas, ações/indicadores em grelha 3×2.
- **Estreito:** parâmetros numa coluna, ações/indicadores em grelha 2×N e
  resultados empilhados verticalmente.

O conteúdo está protegido por scroll vertical, os divisores continuam
redimensionáveis e nenhum controlo pode ser comprimido abaixo da sua altura
legível. Os nomes técnicos das métricas são apresentados em português, valores
proporcionais usam percentagem e contagens deixam de aparecer com quatro casas
decimais. Os nomes originais permanecem inalterados nos ficheiros exportados.
