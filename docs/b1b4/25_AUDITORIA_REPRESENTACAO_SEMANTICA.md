# Doc. 25 — Auditoria da representação semântica (preparação B3; nada implementado)

Âmbito: apenas **auditoria** da representação enriquecida existente (`validation/benchmark/audit_representation.py`), DEV, 18 unidades, só treino. Sem alterar B1/B2/oráculo/Original. Estado: B3 `DESIGN_ONLY`, B4 `NOT_IMPLEMENTED`.

## 1. Perguntas e hipóteses (registadas antes)
- HA1 As derivadas OWL não acrescentam informação sobre Y condicionada às originais (são funções determinísticas das colunas de origem).
- HA2 A estrutura de grupos da ontologia verdadeira alinha-se melhor com a estrutura dos dados do que grupos permutados.
- HA3 As derivadas verdadeiras ganham ao melhor membro/originais na raiz com frequência superior às permutadas/convencionais.
- HA4 A representação enriquecida aumenta o espaço de candidatos m-of-n sem redundância controlada.

## 2. Resultados (médias das unidades; controlo = ontologia permutada `both`; convencional = grupos orientados por dados)
| Medida | OWL verdadeira | permutada | convencional |
|---|---|---|---|
| nº de derivadas (média/unidade) | 16,6 | 16,6 | 2,9 |
| |ρ| médio dentro dos grupos | 0,52 | 0,39 | 0,68 |
| z do alinhamento vs grupos aleatórios | **1,28** | 0,12 | 2,79 |
| fração de grupos acima do p95 nulo | 0,41 | 0,04 | 0,89 |
| derivadas ≈ uma só original (|r|>0,95) | 0,17 | 0,56 | 0,54 |
| derivadas com MI > melhor membro | 0,30 | 0,17 | 0,14 |
| info. condicional I(Y;D\|X) | 0 | 0 | 0 |
| raiz: bootstrap em que a derivada vence | **0,23** | 0,09 | 0,06 |
| ganho médio de IG derivada−original | −0,035 | −0,070 | −0,136 |
| razão candidatos enriquecido/original | 1,32 | 1,39 | 1,19 |
Matching: 18/18 unidades `VALID_DOMAIN_ONTOLOGY`, cobertura de features 1,0, raciocinador consistente. Estabilidade das derivadas sob reajuste com 80 % dos dados: correlação mínima 0,92, variação relativa máx. do desvio-padrão 0,12.
Por dataset: iris — 2 derivadas, alinhamento ≈ 0 (z≈0), 50 % quase duplicam uma original; wine/digits — z 1,6–2,4, nenhuma quase-duplicada; breast_cancer — 53 derivadas, z≈1,5, 9 % quase-duplicadas.

## 3. Interpretação
1. **HA1 confirmada** (por construção): derivadas = média padronizada de colunas de origem ⇒ I(Y;D|X)=0. A representação não cria informação; só reorganiza (agrupa) o que o modelo teria de descobrir. O valor possível é **indutivo** (viés de árvore/regras mais compactas), não informacional.
2. **HA2 parcialmente confirmada**: a ontologia verdadeira alinha-se com a estrutura dos dados acima de grupos permutados (z 1,28 vs 0,12) na maioria dos datasets (não em iris). É evidência de **conhecimento de domínio válido**, mas inferior à agregação guiada por dados (z 2,79) — a ontologia não é o melhor agrupamento numérico, e não é suposto ser.
3. **HA3 parcial**: derivadas verdadeiras vencem a raiz em 23 % dos bootstraps vs 9 %/6 %; porém o ganho médio de IG face às originais é **negativo** em média. Utilidade numérica ocasional, não sistemática.
4. **HA4 confirmada**: +32 % de candidatos m-of-n (até 53 derivadas em bc); a permutada aumenta ainda mais (1,39) — logo o aumento do espaço em si não distingue verdadeira de permutada, o que é consistente com o Doc. 23 (verdadeira ≈ permutada em ΔF/ΔAcc).
5. Distinção explícita: **conhecimento ontológico válido** (alinhamento, consistência, cobertura) ≠ **feature numericamente útil** (ganho de IG, fidelidade). Nenhuma feature usa rótulos (derivadas ajustadas só em X de treino; verificado em `transform_contract` e testes de agnosticismo).
6. Uso real de relações semânticas: pesos não-unitários, grupos nomeados, relatedness e profundidade presentes (ex.: iris 2/2/6/1,0), mas o seu efeito na decisão de splits não supera permutação (Doc. 23 §2.5). Contribuição de axiomas individuais **não isolada** (fora de âmbito; exigiria ablação por axioma).

## 4. Se a representação só duplica as originais — alternativas para B3 (não implementadas)
- B3-a **Restrições estruturais no m-of-n** (limitar combinações a membros de grupos OWL, em vez de derivadas numéricas): usa a relação sem criar features redundantes.
- B3-b **Orientação semântica de splits como prior de desempate** (candidatos equivalentes em ganho resolvidos pela hierarquia OWL): efeito pequeno mas atribuível; mantém Original ≡ Reloaded sem ontologia.
- B3-c **Regularização por profundidade ontológica** (preferir literais mais gerais quando o ganho é equivalente): ligada a regras mais compactas; testar contra complexidade.
- B3-d **Derivadas condicionais** (contrastes/razões entre membros do grupo, em vez da média) para introduzir informação nova **sem rótulos**; exige novo controlo (convencional equivalente).
Cada um: flag, auditável, controlo negativo permutado e convencional, orçamentos preservados, separação por oráculo quando o espaço de entrada do MLP diferir.

## 5. Riscos experimentais
Permutações podem criar artefactos de escala (mitigado: derivadas padronizadas, mesmos tamanhos de grupo); DEV pequeno (4 datasets, ontologias manuais, EP 0,004–0,007); iris com 2 derivadas e duplicação 50 %; baixo poder para efeitos < 0,008; risco de interpretar utilidade numérica como conhecimento. Sem confirmatório.

## 6. Decisões pendentes
Autorizar (ou não) B3-a…d e dataset(s)/ontologias adicionais com categóricas reais. Nada será implementado antes de autorização.
