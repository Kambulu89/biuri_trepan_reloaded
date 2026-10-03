# Contrafactuais — resumo

- pipeline: `cfkit-1.0` · resultados: 3

## LORE-inspired · instância None · CONSTRAINT_INFEASIBLE
- modelo explicado: **MLP** · dataset: synthetic
- Não foi encontrado um contrafactual válido para MLP mudar a previsão de 0 para 1: [CONSTRAINT_INFEASIBLE] existem vectores que mudam a classe mas todos violam constraints hard
- semântica: UNDETERMINED · seed 0 · 0.557s · terminação: completed

## CLEAR-inspired · instância None · SUCCESS
- modelo explicado: **MLP** · dataset: synthetic
- Para o modelo MLP prever 1 em vez de 0, o contrafactual encontrado (CLEAR-inspired) altera 2 característica(s): children: 4 → 1 (-3); income: 25.8397 → 35.9041 (+10.0643). Com estas alterações o modelo passaria a prever 1. Método CLEAR-inspired é uma aproximação inspirada no método original (não a implementação canónica). Isto descreve o comportamento do modelo, não uma relação causal nem uma garantia sobre o mundo real.
- semântica: VALID · seed 0 · 0.092s · terminação: completed

## CoGS-inspired · instância None · SUCCESS
- modelo explicado: **MLP** · dataset: synthetic
- Para o modelo MLP prever 1 em vez de 0, o contrafactual encontrado (CoGS-inspired) altera 2 característica(s): children: 4 → 3 (-1); income: 25.8397 → 65.7423 (+39.9025). Com estas alterações o modelo passaria a prever 1. Método CoGS-inspired é uma aproximação inspirada no método original (não a implementação canónica). Isto descreve o comportamento do modelo, não uma relação causal nem uma garantia sobre o mundo real.
- semântica: VALID · seed 0 · 0.498s · terminação: found_enough
