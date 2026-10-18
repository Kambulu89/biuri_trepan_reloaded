# Doc. 53 — Plano H2: enriquecimento ontológico real com German Credit (DEV exploratório)

Estado: **plano e preparação; nenhuma campanha H1 ou H2 foi executada ou está autorizada.** Só `validation/` e documentação; nada em `core/`; TREPAN Original, C4.5, MLP, baselines, ontologias congeladas e resultados históricos intocados; nenhum dataset confirmatório; sem merge para `main`. `M0_M6_CORE_INTEGRATION = NOT_AUTHORIZED`.

## 0. Decisões do responsável (registo)
R1–R7 aprovadas para preparar a B3.4 v2 (histórico preservado); R8 aprovada **só** como família adicional e exploratória (Doc. 49, registo de decisão); campanha H1 (7,2 CPU-h) e campanha H2 (2,7 CPU-h) **não** autorizadas; prioridade científica = **H2**; German Credit aprovado como candidato **DEV exploratório sujeito a validação documental**; `core/` e M0–M6 não autorizados. Estado da CI: o commit `21aedf2` tem Windows 3.11, Windows 3.12 e `agnosticism-guard` em sucesso; o merge `f1b2955` na `main` tinha o guard em sucesso e o Windows em curso na última consulta (**não declarado verde**).

## 1. Objetivo e pergunta
Objetivo inalterado: `Reloaded(None) ≡ Original`; com enriquecimento efetivo, `Acc(Reloaded_OWL) > Acc(Original) > Acc(C4.5)` e `Fid(Reloaded_OWL) > Fid(Original)`, `Fid(Reloaded_OWL) > Fid(C4.5)`, em simultâneo, genuinamente atribuível à ontologia — como **objetivo experimental**, não garantido.
Pergunta H2: *como usar conhecimento ontológico verdadeiro, documentado e logicamente válido para melhorar simultaneamente fidelity e accuracy das árvores extraídas do MLP?* O foco **não** é aumentar axiomas, features ou candidatos, mas mecanismos que aproveitem relações OWL realmente relevantes. Sete camadas a manter **separadas** em toda a análise: (1) qualidade intrínseca; (2) informação de domínio adicional; (3) representação das features; (4) inferências do reasoner; (5) orientação dos candidatos m-of-n; (6) alterações reais de split; (7) benefício observado em accuracy e fidelity.

## 2. Seleção do dataset e limitações
Critérios a priori, independentes de desempenho (Doc. 50). German Credit: atributos categóricos one-hot cuja ordem é informação ausente da representação; inferência real em OWL 2 DL (classes definidas); cobertura 61/61; controlos limpos (permutação de ordens). Historial desfavorável ao Reloaded numa implementação anterior (Doc. 12), logo não é uma escolha favorável.
**Limitações:** exposto (EC1) ⇒ nunca confirmatório; **um** dataset ⇒ unidade de inferência = 1, evidência `observed_only`; **sem documentação local** (só `german.data`; rede indisponível) ⇒ fontes não verificadas; axiomas de rascunho de IA; ordem de `job` e agrupamentos de `purpose` frágeis; atributos sensíveis não modelados; os 10 literais estão **todos no espaço de hipóteses do Original** (ganho só por orientação).

## 3. Fases, critérios de aceitação e riscos
| Fase | Conteúdo | Estado | Critério de aceitação | Riscos |
|---|---|---|---|---|
| **H2-P0** Fontes e conceitos | obter e verificar o dicionário de dados; confirmar definição dos atributos, significado das categorias, nominal vs ordinal | **bloqueada** (sem documento; DNS) — feito: verificação estrutural da codificação (X de treino), 14 fichas com decisão de auditoria | cada atributo e categoria usados citados de fonte verificada com localizador; classificação nominal/ordinal confirmada; axiomas sem fundamento excluídos | citar de memória (proibido); ordem de `job` e `employment_since` A71 |
| **H2-P1** Validação ontológica | consistência, satisfatibilidade, inferências, matching, revisão | **parcial**: HermiT consistente e sem classes insatisfazíveis em O1–O3, H, I; 61/61 mapeados; auditoria das 55 inferências e 10 literais (Doc. 54) | + revisões de domínio e de ontologia aprovadas por pessoas designadas; verificador OWL 2 DL formal | revisão inexistente; proxy DL |
| **H2-P2** Compilação experimental | axiomas → literais candidatos em `validation/` | **feita no protótipo**: `h2_axiom_mechanisms`, `h2_trace`, 8+7 testes (Doc. 55) | mapeamento correto; semântica preservada; mesmo oráculo; sem fuga; categorias desconhecidas; complexidade; **equivalência com a semântica desligada** | sem ligação ao arnês de pesquisa; M-G só desenho |
| **H2-P3** Controlo causal | ontologia plana, neutralizada, permutada/aleatória, Original equivalente (matriz A–J) | **desenhada** (Doc. 56) | contrato de identidade (instâncias, MLP, representação, recursos) satisfeito por contraste; controlos construídos antes de qualquer resultado | coincidência de H/I com O2; comparações não interpretáveis |
| **H2-P4** Avaliação exploratória | benchmark | **preparado, NÃO executado** | pré-registo congelado; orçamento aprovado; pré-requisitos do Doc. 56 §7 cumpridos | exposição; um dataset; ajuste implícito |

## 4. H1 e R8: só testes técnicos pequenos
Não se executa a campanha H1. A auditoria mostrou a tensão entre recuperabilidade da partição e aprendizagem do MLP (XOR: não aprendido; AND: nunca acima da maioritária). Autorizados e feitos: testes técnicos de viabilidade da família aditiva (R8), com calibração independente (Doc. 57) — o MLP supera a maioritária em todas as 16 células (D_o médio 0,055–0,329), a quota de importância nos indicadores verdadeiros tem mediana 0,56–0,89 (q10 ≥ 0,44; acaso 0,125) nas células medidas (s ≤ 3), e a identificação supervisionada genérica é incompleta apenas para n pequeno e ruído elevado. **Acima da maioritária não é aprendizagem adequada por si só**: o gate G1b (adequação estrutural) é obrigatório. R8 continua exploratória: gerador a congelar antes de avaliar; a ontologia recebe a partição, não o suporte da regra-alvo; estrutura recuperável dos dados documentada; fidelity e accuracy separadas; sem substituir retroativamente a hipótese XOR.

## 5. Limiares das portas
Proposta técnica em `57_PROPOSTA_LIMIARES_PORTAS.md`: definições matemáticas, finalidade, interpretação, falsos positivos/negativos, sensibilidade, calibração independente e valores candidatos (Δ_o: 0,05; κ: 0,375; τ_s: 0,8 com estimadores mais fortes a acrescentar; τ_u: 0,5), **a decidir pelo responsável pela metodologia e congelar antes de avaliar variantes**. Nenhum foi escolhido pelo desempenho de qualquer ontologia.

## 6. Revisão humana
Fichas separadas (domínio, engenharia de ontologias, metodologia) em `…/v0_draft/review_forms/`, com nome, data, assinatura e decisão **vazios**. Cada revisor valida só a sua competência. Até haver validações, todos os axiomas são `HYPOTHESIS`/pendentes. Nenhum nome, assinatura ou aprovação foi inventado; axiomas produzidos por IA não são apresentados como conhecimento de domínio independentemente validado.

## 7. Orçamento (estimativa; nada autorizado)
* Computacional: fase R (German) ≈ 0,7 CPU-h (limite 1,0); fase P sintética opcional ≈ 1,5; calibração/portas já executadas (< 0,3 CPU-h nesta fase). Detalhe: Doc. 56 §4.
* Humano: dominante; revisão de ≈ 10 fichas substantivas + fontes; obtenção do dicionário de dados; decisão sobre limiares. Engenharia: ligação do adaptador ao arnês D/E/F e rastreio por literal (não estimado em CPU).

## 8. Bloqueios e decisões necessárias
1. **Bloqueio P0:** fornecer o dicionário de dados `german.doc` (ficheiro) ou permitir o acesso a `archive.ics.uci.edu`; sem isso o conjunto avaliável permanece vazio.
2. Designar os três revisores.
3. Rever/fixar Δ_o, κ, τ_s, τ_u (Doc. 57).
4. Aceitar, ou não, executar a fase R com o rótulo «rascunho de IA não validado» (explorar apenas mecanismos) caso a revisão tarde.
5. Autorizar (ou não) o trabalho de engenharia de ligar o adaptador ao arnês e a fase P sintética.
6. Confirmar a CI do merge `f1b2955` (Windows em curso) e do commit atual.
