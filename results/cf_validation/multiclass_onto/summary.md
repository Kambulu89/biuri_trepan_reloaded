# Contrafactuais — resumo

- pipeline: `cfkit-1.0` · resultados: 3

## LORE-inspired · instância None · SUCCESS
- modelo explicado: **MLP** · dataset: synthetic
- Para o modelo MLP prever 0 em vez de 1, o contrafactual encontrado (LORE-inspired) altera 1 característica(s): employment: b → c. Com estas alterações o modelo passaria a prever 0. Método LORE-inspired é uma aproximação inspirada no método original (não a implementação canónica). Isto descreve o comportamento do modelo, não uma relação causal nem uma garantia sobre o mundo real.
- semântica: VALID · seed 0 · 0.576s · terminação: completed

## CLEAR-inspired · instância None · CONSTRAINT_INFEASIBLE
- modelo explicado: **MLP** · dataset: synthetic
- Não foi encontrado um contrafactual válido para MLP mudar a previsão de 1 para 0: [CONSTRAINT_INFEASIBLE] existem vectores que mudam a classe mas todos violam constraints hard
- semântica: UNDETERMINED · seed 0 · 0.104s · terminação: completed

## CoGS-inspired · instância None · SUCCESS
- modelo explicado: **MLP** · dataset: synthetic
- Para o modelo MLP prever 0 em vez de 1, o contrafactual encontrado (CoGS-inspired) altera 2 característica(s): children: 2 → 3 (+1); income: 32.6698 → 13.3202 (-19.3496). Com estas alterações o modelo passaria a prever 0. Método CoGS-inspired é uma aproximação inspirada no método original (não a implementação canónica). Isto descreve o comportamento do modelo, não uma relação causal nem uma garantia sobre o mundo real.
- semântica: VALID · seed 0 · 0.825s · terminação: found_enough
