# Doc. 23 — Análise causal do ganho D − C (amostragem coerente vs histórica)

Estado: **PRÉ-REGISTO ESCRITO ANTES DAS EXPERIÊNCIAS** (commit anterior a qualquer execução desta fase). Os resultados são acrescentados abaixo, sem alterar este pré-registo; qualquer desvio é listado em «Alterações de protocolo». Investigação **exploratória**: DEV (4 datasets, 18 unidades) já observado; não declara superioridade generalizada nem causalidade definitiva.

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


## 1. Execução e controlo computacional (factos)
- 18 unidades (iris, wine, breast_cancer ×5 sementes mestras; digits ×3), 8 células S×A×L × réplicas (r=0 nativas reaproveitadas do benchmark anterior, bit-a-bit) + 4 braços P2 × 2 réplicas (1 em digits) + braços de referência (MLP, C4.5, Original, Reloaded-core).
- **Reutilização de checkpoints verificada em TODAS as 18 unidades**: `tuning_fits_resumed` = nº total de ajustes (iris 120, wine/bc 150, digits 240), `tuning_execution_count` = 1; fingerprint v2 (Doc. 22/tests) coincidiu em todas; nenhum ajuste de tuning refeito.
- **Integridade**: 54/54 comparações dos braços de referência (Original, Reloaded-core, C4.5) idênticas às do benchmark anterior (fidelidade, accuracy, nº de nós).
- Custo: soma de tempos de parede das unidades ≈ 3,7 h (execução em paralelo limitado; unidades < 120 min cada, a maior ≈ 40 min). CPU total estimado ≈ 5 h (soma dos tempos de parede das unidades + plausibilidade, auditoria e regressão; **estimativa, não medido por contador de CPU**), abaixo do teto de 10 CPU-h e do previsto (≈ 5,6 h). Nenhuma unidade repetida; nenhuma duplicação de benchmarks.
- Regressão (testes de benchmark/protocolo/equivalência): 176 passed.

## 2. Resultados (exploratórios, DEV, `observed_only`; ΔF = fidelidade ao oráculo)
### 2.1 Matriz S×A×L (média sobre unidades; cada unidade = média das réplicas)
| S | A | L nativo | L trocado |
|---|---|---|---|
| histórico | ativas | 0,9121 | 0,9119 |
| histórico | sem ativas | 0,9182 | 0,9200 |
| coerente | ativas | 0,9183 | 0,9187 |
| coerente | sem ativas | 0,9189 | 0,9195 |
Original (referência comum): média ≈ 0,9203 (D−Original −0,002 ± 0,004; histórico-ativas−Original −0,008 ± 0,005; 5 melhores/11 piores).

### 2.2 Efeitos (todas as réplicas; EP entre unidades; Holm sobre 7 contrastes)
| Contraste | ΔF | EP | p (Holm) |
|---|---|---|---|
| S (coerente−histórico) | +0,0033 | 0,0027 | 1,0 |
| A (ativas−sem) | −0,0039 | 0,0033 | 1,0 |
| L (trocado−nativo) | +0,0006 | 0,0007 | 1,0 |
| S×A | +0,0032 | 0,0024 | 1,0 |
| S×L | −0,0002 | 0,0005 | 1,0 |
| A×L | −0,0006 | 0,0006 | 1,0 |
| S×A×L | +0,0004 | 0,0004 | 1,0 |
Accuracy (real): nenhum contraste significativo (A −0,0066 ± 0,0037, p_Holm 0,65). Nº de nós: S −1,17 ± 0,45 (p 0,018; Holm 0,13): árvores coerentes ≈ 1 nó menores (exploratório).

### 2.3 Decomposição de D − C
- **Com a réplica original (r=0), D − C = +0,0121 ± 0,0051 (10 melhores/2 piores)** — reproduz o valor do Doc. 22 (+0,011).
- Réplicas novas: r=1 **+0,0040 ± 0,0082**; r=2 **+0,0023 ± 0,0057**. Média das réplicas por unidade: **+0,0062 ± 0,0040 (8 melhores/7 piores, 3 empates)**. Com ativas desligadas: +0,0007 ± 0,0031.
- Desvio-padrão entre réplicas dentro da célula (mesma unidade, só muda o fluxo de amostragem): **0,0166** — mais de 3× o efeito pontual D−C e ~1,4× o EP entre unidades.
- Fator L (regra de colunas do teste de adoção): contribuição para D−C = −0,0003 ± 0,0014 (−5 % [IC95 −164 %, +123 %]) no coerente e −0,0002 ± 0,0007 (−3 %) no histórico. **H-L rejeitada** (critério §0.5: ≥ 70 % e > 2 EP — não cumprido). Taxas de adoção do modelo local praticamente iguais (0,50–0,51 em todas as células).
- Por dataset (D−C, todas as réplicas): wine +0,012, breast_cancer +0,005, digits +0,004, iris +0,004; accuracy ≈ 0.

### 2.4 Indicadores de mecanismo (observacionais, média das unidades)
| Célula | adoção local | partilha m-of-n | partilha de literais derivados | KS orig. | fora de intervalo | TV rótulos vs. reais | literais/split |
|---|---|---|---|---|---|---|---|
| h/on | 0,50 | 0,88 | 0,26 | 0,210 | 0,36 | 0,171 | 2,45 |
| c/on | 0,50 | 0,87 | 0,27 | 0,221 | 0,36 | 0,180 | 2,44 |
| h/off | 0,51 | 0,89 | 0,24 | 0,191 | 0,41 | 0,161 | 2,50 |
| c/off | 0,51 | 0,89 | 0,25 | 0,213 | 0,42 | 0,167 | 2,48 |
Nenhum indicador (adoção, escolha m-of-n, uso de derivadas, entropia/partilha de rótulos das consultas) distingue de forma sistemática histórico de coerente. Interpretação: **H-M e H-O sem suporte**; **H-S** (efeito de S com L fixo) pequeno e não distinguível de zero (+0,0033 ± 0,0027); **H-A/interação** de sinal consistente (S×A +0,003) mas não significativa; **H-N confirmada**: o efeito encolhe de +0,012 (r=0) para +0,003 (r=1,2) — compatível com regressão à média sobre um fluxo RNG favorável (*winner's curse*); não se escolheram sementes (r=0 foi o único fluxo disponível antes, não selecionado por resultado).

### 2.5 Controlos ontológicos P2 (fidelidade; verdadeira − permutação parcial; mesmo orçamento e dimensionalidade)
| Contraste | ΔF | EP | n |
|---|---|---|---|
| verdadeira − estrutura permutada (ativas) | −0,0006 | 0,0046 | 18 |
| verdadeira − estrutura permutada (sem ativas) | −0,0042 | 0,0037 | 18 |
| verdadeira − features permutadas (ativas) | −0,0001 | 0,0072 | 18 |
| verdadeira − features permutadas (sem ativas) | −0,0013 | 0,0055 | 18 |
Accuracy: −0,002 ± 0,006; −0,001 ± 0,003; −0,004 ± 0,006; +0,004 ± 0,005. **Nenhum contraste cumpre o critério §0.5 (|Δ| > 2 EP, mesmo sentido em F e Acc).** Em somatório com a evidência do Doc. 22 (permutada total e convencional) a conclusão é: **neste ambiente (DEV, 18 unidades, 4 datasets) a ontologia verdadeira não supera as permutações parciais, nem a engenharia convencional, nem o Original** — o ganho ontológico **não está demonstrado** (nem refutado: o poder estatístico é limitado, EP ≈ 0,004–0,007 face a efeitos plausíveis ≲ 0,01).

## 3. Resposta às perguntas do P0
1. *Mecanismo do ganho D−C ≈ +0,011?* Não identificado como propriedade da coerência: (i) a regra de adoção (L) não o explica; (ii) o efeito principal de S com 3 réplicas é +0,003 (1,2 EP); (iii) o valor +0,011 corresponde sobretudo ao fluxo RNG da réplica r=0 (réplicas 1–2: +0,004/+0,002); (iv) sem consultas ativas D−C ≈ 0,0007; a evidência é de que a perda do histórico com ativas (−0,008 vs Original) tem uma componente de ruído de fluxo e uma interação com ativas (A: −0,006 no histórico vs −0,0006 no coerente), não de coerência algébrica (violação já 0 no histórico, Doc. 21).
2. *Efeito da coerência isolado*: S = +0,003 ± 0,003 (n.s.). *Efeito das ativas*: A = −0,004 ± 0,003 (n.s.; −0,0066 em accuracy, p_Holm 0,65). *Interação*: S×A = +0,003 ± 0,002 (n.s.).
3. *Não emparelhabilidade RNG*: documentada; mitigada com réplicas e médias por unidade. O desvio-padrão entre réplicas (0,0166) é a incerteza dominante.
4. Complexidade: coerente produz árvores ~1,2 nós menores (exploratório).

## 4. Incerteza, comparações e limitações causais
- Contrastes calculados: 7 (fidelidade) + 7 (accuracy) + 7 (nós) + 4 P2 ×2 métricas + 1 decomposição L; Holm aplicado por família de 7; nada sobrevive a Holm. Resultados exploratórios (DEV já observado).
- Limitações: 4 datasets/18 unidades; 2–3 réplicas; ontologias manualmente definidas por dataset (DEV); fator L implementado por *patch* no executor (não é flag do núcleo); S e fluxo RNG confundidos por construção (não emparelháveis); efeito de A medido só com a política ativa atual; ausência de poder para efeitos < 0,008.
- Reforço de política: não se atribui o resultado a «coerência» pelo nome da variante; não há promoção; confirmatório intocado.

## Alterações de protocolo
Nenhuma. (Critérios §0.5 e desenho §0.3 aplicados tal como registados. Execução: r=0 nativas reaproveitadas do benchmark anterior, previsto em §0.3.)
