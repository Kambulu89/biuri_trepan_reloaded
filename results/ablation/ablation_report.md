# Estudo de ablação OWL — TREPAN Reloaded

Teste bloqueado; ontologias TBox RDF/XML; reasoner OWL DL obrigatório.

| Modelo | Accuracy | Balanced accuracy | Macro-F1 | Fidelidade ativa | Fidelidade MLP original |
|---|---:|---:|---:|---:|---:|
| C4.5-Nativo | 0.951 | 0.952 | 0.951 | 0.706 | 0.706 |
| TREPAN Original | 0.748 | 0.743 | 0.718 | 0.886 | 0.886 |
| TREPAN Reloaded — com OWL | 0.922 | 0.915 | 0.917 | 0.922 | 0.707 |
| TREPAN Reloaded — sem OWL | 0.748 | 0.743 | 0.718 | 0.886 | 0.886 |

## Efeito pareado da OWL

| Métrica | Diferença média | IC bootstrap 95% | Vitórias/Empates/Derrotas | Wilcoxon p |
|---|---:|---:|---:|---:|
| accuracy | 0.174 | [0.053, 0.315] | 8/0/1 | 0.0391 |
| balanced_accuracy | 0.172 | [0.049, 0.303] | 8/0/1 | 0.0391 |
| macro_f1 | 0.199 | [0.047, 0.364] | 8/0/1 | 0.0391 |
