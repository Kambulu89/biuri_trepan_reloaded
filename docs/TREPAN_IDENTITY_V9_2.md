# Identidade do TREPAN na V9.2

A classe `core.trepan_extractor.TREPANExtractor` histórica ajusta `DecisionTreeClassifier` a rótulos do MLP. Na V9.2 ela é tratada como **CART destilado (baseline de destilação)** e mantém o alias apenas para compatibilidade.

O nome **TREPAN Original** passa a referir-se a `core.trepan_original.TrepanOriginalClassifier`, que reutiliza o motor `CanonicalTrepanClassifier`: expansão best-first, testes m-of-n, limite de nós/profundidade/consultas e consultas ao oráculo. A amostragem adicional actual é empírica marginal. Portanto, esta implementação aproxima o protocolo de Craven & Shavlik, mas a amostragem condicional específica a cada nó ainda deve ser considerada uma extensão pendente se for exigida reprodução literal do artigo original.

Esta distinção impede que um CART destilado seja apresentado como TREPAN clássico.
