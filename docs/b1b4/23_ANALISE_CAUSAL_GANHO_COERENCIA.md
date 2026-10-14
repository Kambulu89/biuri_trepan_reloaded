# Doc. 23 — Análise causal do ganho D − C (amostragem coerente vs histórica)

Estado: **PRÉ-REGISTO ESCRITO ANTES DAS EXPERIÓNCIAS** (commit anterior a qualquer execução desta fase). Os resultados são acrescentados abaixo, sem alterar este pré-registo; qualquer desvio é listado em «Alterações de protocolo». Investigação **exploratória**: DEV (4 datasets, 18 unidades) já observado; não declara superioridade generalizada nem causalidade definitiva.

## 0. Pré-registo

### 0.1 Ponto de partida (evidência já existente, Doc. 22)
D − C ≈ +0,011 (EP 0,005; 10 melhores/2 piores/18). Violação algébrica do Reloaded histórico (C) já é 0. Mudar só o modelo de amostragem em regime só-representação (λ=0) dá ≈ 0 (+0,0004 ± 0,009). Com ativas desligadas: E − F ≈ 0. Logo o efeito aparece com a camada semântica ligada e consultas ativas ligadas.

### 0.2 Mecanismos candidatos (hipóteses, formuladas antes de correr)
| Id | Hipótese | Previsão testável | Intervenção isoladora (NÃO é «coerência») |
|---|---|---|---|
| H-L | O teste de adoção do modelo local (`differs_from`: KS/χ² por coluna) no histórico corre sobre TODAS as colunas (originais+derivadas); no coerente só sobre as originais ⇒ taxas de adoção diferentes ⇒ distribuições de consulta diferentes | taxa de adoção difere entre C e D; trocar a regra de colunas de teste (fator L) desloca ou anula o efeito | fator **L** ∈ {nativo, trocado} |
| H-S | O espaço/modelo de sorteio (enriquecido+projetor vs original+transformador) altera, por si, o fluxo de sorteios (colunas derivadas sorteadas e descartadas, nº de sorteios do RNG) e a distribuição | com L fixo, o efeito S ≠ 0 | fator **S** ∈ {histórico, coerente} |
| H-A | A seleção ativa (pool ×4) interage com S (a pool no coerente tem outra geometria) | interação S×A ≠ 0 | fator **A** ∈ {ativas, sem ativas} |
| H-N | Parte substancial de D−C é ruído de fluxo RNG (as duas variantes não são emparelháveis a nível de sorteio) | desvio-padrão entre réplicas (sementes de amostragem) comparável ao efeito; o efeito encolhe com réplicas | réplicas r = 0,1,2 (semente da árvore `random_state + 1000·r`; mesma divisão, oráculo, orçamento, ontologia) |
| H-M | O efeito passa por escolhas m-of-n/seleção de splits (n.º de literais, uso de derivadas, aceitação semântica) e por complexidade | diferenças nesses indicadores acompanham ΔF | indicadores observacionais |
| H-O | O comportamento do oráculo nas consultas (entropia/partilha de rótulos) difere entre variantes | diferenças na distribuição de rótulos das consultas | indicador observacional |

### 0.3 Desenho fatorial (3 fatores binários, 8 células)
S (sorteio: histórico/coerente) × A (consultas ativas: sim/não) × L (colunas do teste de adoção local: nativo/trocado). «Nativo» = o comportamento atual de cada variante (histórico: todas as colunas; coerente: originais). «Trocado» = intervenção experimental feita **fora do núcleo** (parcheamento transitório no executor `run_causal`), identificada como fator separado. Mesmos dados, divisões, sementes mestras, oráculo congelado, orçamento estrutural (tuning em *checkpoint*), parâmetros do algoritmo e ontologia. Nenhuma normalização nem mudança de representação é introduzida.
- Réplicas: 3 (r = 0 já existente para as 4 células nativas; r = 1,2 novas) em iris/wine/breast_cancer; 2 (r = 0,1) em digits (custo). Emparelhamento exato entre S=hist e S=coerente **é impossível** (fluxos RNG distintos): usa-se média por célula das réplicas por unidade e diferenças emparelhadas por unidade, com EP entre as unidades e a variância entre réplicas reportada.
- Efeitos: 3 principais (S, A, L), 3 interações de 2.ª ordem, 1 de 3.ª ordem = **7 contrastes** sobre ΔF (primário) e 7 sobre ΔAcc (secundário). Família de 7 para ajuste de Holm (reportado ao lado do EP; interpretação exploratória).
- Estimando D − C = (S=coerente,A=sim,L=nativo) − (S=histórico,A=sim,L=nativo); decomposição em efeitos: parte devida a L (via trocar L), a S (L fixo), à interação com A.

### 0.4 Controlos ontológicos (P2) — hipóteses
| Id | Hipótese | Braços (todos coerentes+ativas e sem ativas; mesmo orçamento) |
|---|---|---|
| P2-a | A estrutura semântica verdadeira (pesos/grupos/relatedness/profundidade) orienta melhor do que a permutada, **mantendo as mesmas features** | `coh_structshuf` (features verdadeiras; estrutura permutada) vs `coh_full` |
| P2-b | O conteúdo das features derivadas verdadeiras (grupos de colunas da ontologia) é melhor do que grupos aleatórios, **mantendo a estrutura verdadeira** | `coh_featshuf` (features de colunas permutadas; estrutura verdadeira) vs `coh_full` |
| P2-c | Efeito conjunto (já medido): permutada total `coh_full_shuf`, convencional `conv_coh`, só-representação `coh_features` | evidência existente (Doc. 22) reutilizada |
Decomposição pretendida: engenharia convencional (H) ; representação OWL (Rc−H) ; estrutura verdadeira (D − P2-a) ; conteúdo das features (D − P2-b) ; orientação das divisões (D − Rc) ; geração de consultas (D − C) ; interações. Permutação: `ShuffledSemanticProvider` permuta colunas de origem e/ou estrutura com a mesma dimensionalidade, mesmos tamanhos de grupo e mesmos pesos (apenas reatribuídos) — sem artefactos de escala (derivadas ainda padronizadas).

### 0.5 Critérios (declarados antes)
- Mecanismo «identificado» se: (i) a trocar um fator isolado reproduz ou anula ≥ 70 % de D − C (média das réplicas) **e** (ii) o contraste tem |ΔF| > 2 EP; caso contrário o relatório afirma «não identificado» e apresenta os limites.
- Ontologia verdadeira «útil» só se superar **ambas** as permutações parciais **e** a convencional com |ΔF| > 2 EP em ΔF e em ΔAcc no mesmo sentido.
- Nenhuma seleção de configuração/semente/dataset com base nestes resultados; nenhum critério alterado depois de ver resultados sem registo abaixo.

### 0.6 Orçamento e controlo computacional (teto 10 CPU-h)
| Bloco | Custo estimado (CPU) |
|---|---|
| Fatorial S×A×L, 20 células-réplica novas (iris/wine/bc) + 12 (digits) | ≈ 3,1 h |
| Controlos P2 (4 braços × r = 0,1 em pequenos; r = 0 em digits) | ≈ 1,7 h |
| Plausibilidade de consultas (amostradores, AUC, fronteira) e testes categóricos | ≈ 0,5 h |
| Auditoria da representação | ≈ 0,3 h |
| **Total previsto** | **≈ 5,6 h < 10 h** |
Verificações antes da campanha: reutilização de *checkpoints* (fingerprint v2) em **todas** as 18 unidades (não só Iris/42): contagem de ajustes retomados e igualdade dos braços repetidos; piloto pequeno (iris/42) e verificação de integridade (braços de referência idênticos aos já publicados); uma única campanha em simultâneo; retoma por unidade (< 120 min por unidade).

### 0.7 Registo de comparações (preenchido no fim)
Ver «Resultados» abaixo: nº de contrastes realmente calculados e ajustados.

## Alterações de protocolo
(nenhuma até ao momento)
