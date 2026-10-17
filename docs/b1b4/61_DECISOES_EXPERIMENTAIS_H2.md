# Doc. 61 — Decisões experimentais H2: limiares, comparabilidade, revisores, custos e auditoria de CI/ramo

Estado: **proposta para decisão**; nada congelado, nada executado além de verificações técnicas; campanhas P, R e H1-B3.4 **não executadas e não autorizadas**; `core/` intocado; sem merge.

## 1. Auditoria de ramo e CI
* `main` = `f1b2955` (merge B3.0–B3.3, P3, B3-Q, preparação H2/B3.4 v2; contém `21aedf2`). Ramo `claude/repository-architecture-analysis-s1vl5a` = `557c381`, **um commit à frente** de `main` (Docs. 53–57 e adaptador/auditoria/fichas); esse commit **não está em `main`**. Este trabalho (Docs. 58–61 e arnês) fica **só no ramo**.
* CI: `557c381` e `f1b2955` verificados com sucesso na consulta anterior (Windows 3.11/3.12, `agnosticism-guard`, regressões científicas, identidade dos baselines); esta sessão acrescenta commits que ainda não têm CI (§7).
* `core/`: a única alteração desde o início do programa B3 é `core/ontology_quality.py` (commit `6540bd7`, P3, isolada e autorizada); nenhuma desde então. Âncoras/evidência congelada intactas (manifests v3 e freezes B32/B33 não alterados). Sem novos merges.

## 2. Limiares das portas: proposta metodológica (nada escolhido nem congelado)
Valores candidatos (Doc. 57): Δ_o = 0,05 (aprendibilidade do oráculo), κ = 0,375 (adequação estrutural; acaso 0,125), τ_s = 0,8 (identificação supervisionada genérica), τ_u = 0,5 (recuperabilidade não supervisionada, ARI).
* **Interpretação matemática**: Δ_o = q10 de (acc_MLP − acc_maioritária) — margem mínima de aprendizagem; κ = quota de importância por permutação nos indicadores verdadeiros ≥ κ (múltiplo de 3 do acaso g/p); τ_s/τ_u = taxa de acerto (≤ τ) de métodos genéricos, para que a estrutura **não** seja trivialmente recuperável pelo método genérico.
* **Dados de calibração**: sementes independentes (9500, 9700, 9800), nunca dados de avaliação ou desempenho de qualquer ontologia.
* **Dependência do método genérico** (sensibilidade medida, acerto de identificação, 20 repetições, média±EP):
| célula | corr | L1_C0.3 | L1_CV | mutual_info | random_forest | best_of_all |
|---|---|---|---|---|---|---|
| s2.0_n60 | 0.57±0.03 | 0.58±0.04 | 0.47±0.05 | 0.25±0.04 | 0.50±0.03 | 0.64±0.03 |
| s2.0_n120 | 0.78±0.03 | 0.88±0.03 | 0.84±0.04 | 0.30±0.04 | 0.73±0.04 | 0.89±0.03 |
| s2.0_n240 | 0.93±0.02 | 0.99±0.01 | 0.99±0.01 | 0.37±0.04 | 0.89±0.02 | 0.99±0.01 |
| s3.0_n60 | 0.43±0.04 | 0.47±0.05 | 0.39±0.05 | 0.22±0.03 | 0.43±0.03 | 0.56±0.03 |
| s3.0_n120 | 0.60±0.03 | 0.64±0.03 | 0.62±0.03 | 0.17±0.04 | 0.59±0.04 | 0.69±0.03 |
| s3.0_n240 | 0.86±0.03 | 0.91±0.02 | 0.91±0.02 | 0.23±0.04 | 0.82±0.04 | 0.93±0.02 |
Leitura: a taxa depende fortemente do estimador (informação mútua muito mais fraca; L1 e correlação mais fortes; `best_of_all` o mais forte). **A escolha de «o método genérico» altera quais células passam a porta τ_s=0,8**: com `best_of_all`, só s=2 n≥120 e s=3 n=240 atingem ≥ 0,8–0,9 (a porta *falha* a tarefa, isto é, o genérico resolve-a); com MI nenhuma célula chega perto. Recomendação: o limiar deve ser definido **contra o melhor estimador genérico razoável**, declarado antes da avaliação, e não contra o mais fraco.
* **Falsos positivos** (porta passa e o problema é trivial para um genérico): subestimar o método genérico. **Falsos negativos** (porta recusa tarefas em que a ontologia ajudaria): τ demasiado baixo ou estimador demasiado forte. **Alternativas**: τ_s em função do número de repetições (intervalo de confiança do acerto, não o ponto); exigir superioridade sobre o genérico em vez de limiar absoluto; dispensar τ_u e usar só τ_s com ablação.
* **Limitações**: 20 repetições (EP ≈ 0,03–0,05); família sintética R8 única; limiares dependem do gerador; não generalizam a dados reais.
* **Decisão**: pertence ao responsável pela metodologia; **nenhum valor foi escolhido pelo desempenho de OWL**; congelar apenas antes de qualquer avaliação confirmatória.

## 3. Comparabilidade
Contrato do Doc. 56 §2 inalterado (mesmas instâncias, `oracle_id`, representação, recursos, conhecimento contabilizado). Achado do arnês: C ≡ F nas condições do defeito (mesmo traço) ⇒ F vs C seria `UNTESTABLE`; para testar M-P ou M-G é preciso decidir se se aumenta o pool/`max_features_per_node` **igualado em todos os braços** (A, B, C…), o que altera o orçamento de candidatos e deve ser pré-registado — não é ajuste pós-resultado.

## 4. Revisores (pendentes)
Três papéis — domínio, ontologias, metodologia — **sem nomes designados**; as fichas e os formulários (`…/v0_draft/review_forms/`) têm nome, data, assinatura e decisão vazios. Nada foi inventado; sem aprovações, as fichas ficam hipóteses e qualquer execução é rotulada «rascunho de IA não validado» (verificação de mecanismo, nunca benefício validado).

## 5. Custo atualizado (CPU-h; nada autorizado)
| Item | Estimativa | Estado |
|---|---|---|
| Fase R (German, A–J) | ≈ 0,7 (limite 1,0) | não autorizada; bloqueada por S1–S3 e revisões |
| Fase P (sintético) | ≈ 1,5 | não autorizada; depende de portas aprovadas |
| H1-B3.4 | 7,2 | não autorizada |
| Verificações técnicas desta fase | < 0,2 | feitas |
| Sensibilidade a estimadores | ≈ 0,1 | feita |
| Revisão humana | horas por revisor | por designar |
Total a decidir (R+P) ≈ 2,2 CPU-h; H1 à parte.

## 6. Propostas de execução (para decisão)
1. Desbloquear S1–S3 (domínios ou ficheiros) e recomeçar Doc. 58/59. 2. Designar os três revisores. 3. Decidir os limiares (§2), incluindo qual o estimador genérico de referência. 4. Decidir a política de pool igualado (§3). 5. Só então: pré-registo em commit separado e, se aprovado, fase R rotulada.
Bloqueios atuais: S1–S3 inacessíveis; revisores; limiares; evidência de utilização efetiva (apenas 2/10 literais entram no pool).

## 7. Verificações
Guard de agnosticismo: OK. Testes H2 (harness, fontes/inventário, auditoria, rastreio, contrato) e contrato de agnosticismo: todos passam. CI dos novos commits: por verificar após o push.
