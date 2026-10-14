# Doc. 19 — Estudo isolado do desempate m-of-n (limitação de reprodutibilidade do Original)

Estado: estudo **experimental isolado**. O TREPAN Original canónico, o C4.5, o MLP e o manifest congelado **não foram alterados**. A variante vive
em `validation/experimental/` (nunca em `core/`; teste estático impõe-no), nunca entra na avaliação B2 como baseline e nunca é misturada com o Original canónico.
Reprodução: `python -m validation.experimental.tiebreak_study --out DIR --dev --seeds 10`; testes: `tests/test_tiebreak_variant.py`;
evidência bruta: `docs/evidence/b1b4/tiebreak_study/tiebreak_study.json`.

## 1. Mecanismo da limitação

1. `IG(máscara)` e `IG(~máscara)` são matematicamente iguais, mas **não** bit a bit em ~56 % dos casos (ordem de somas em `n_left/n·H_l + n_right/n·H_r`).
2. A pesquisa m-of-n ordena candidatos por `(-score, …)`; quando dois candidatos têm score matematicamente igual, o desempate é decidido pelo **ruído de
   ulp**, não por uma regra. Casos típicos: (a) complementos lógicos `m-of-n(L) ≡ (n−m+1)-of-n(¬L)` (mesma partição, ramos trocados); (b) testes distintos com
   o mesmo particionamento nos dados do nó.
3. O critério de extensão do feixe (`score_novo > score_atual`) também é sensível a diferenças de 1 ulp.
4. Consequência: a árvore (e, por mudar a ordem/ramo, o fluxo de amostragem ao oráculo) pode depender da plataforma (libm/BLAS/SIMD), do compilador de NumPy e da
   ordem de reduções. É uma propriedade do Original, **registada como limitação de reprodutibilidade** (contrato EQ v2.0), não um erro do Reloaded.

## 2. Variante experimental (`DeterministicTieBreakTrepan`)

- Quantiza o score a `score_digits` casas (default 12) antes de ordenar/comparar.
- Chave canónica de desempate: `(nº de literais, m, assinatura canónica = min(teste, complemento), assinatura)`; a assinatura canónica é invariante ao complemento.
- Só estende o feixe se o score **quantizado** melhorar (empate quantizado ≠ melhoria).
- Não consome RNG, não faz consultas próprias; só muda a escolha entre candidatos. `score_digits=None` desliga e reproduz o Original **bit a bit** (teste).

## 3. Resultados (Linux, Python 3.11, NumPy 1.26.4, ambiente numérico fixado)

Amostra: 60 problemas sintéticos (10 sementes × {contínuo, misto} × 3 formas p/k) e 9 ajustes DEV (iris/wine/breast_cancer × 3 sementes, MLP sklearn).

| Medida | Sintéticos (60) | DEV (9) |
|---|---|---|
| variante(None) ≡ Original (regras, previsões, fluxo ao oráculo) | 60/60 | 9/9 |
| variante(12) com **mesmas regras** que o Original | 42/60 | 3/9 |
| variante(12) com **mesmas previsões** que o Original | 49/60 | 5/9 |
| variante(12) com **mesmo fluxo ao oráculo** | 49/60 | 5/9 |
| Δ fidelidade média (variante − Original, mesmo treino) | −0,0006 | −0,0038 |
| Δ nº consultas médio | +2,4 | +8,7 |
| Δ nº nós médio | −0,07 | 0,0 |
| **Ruído 1–3 ulp: ajustes em que o Original muda de regras** | **52/60** | **9/9** |
| Ruído 1–3 ulp: Original muda de previsões | 23/60 | 6/9 |
| Ruído 1–3 ulp: Original muda o fluxo ao oráculo | 48/60 | 6/9 |
| Ruído 1–3 ulp: **variante(12)** muda regras/previsões/fluxo | **0/60** | **0/9** |
| Máx. |Δ fidelidade| induzido por ruído no Original | 0,0625 | 0,0476 |
| Quantização 6, 9, 12, 14 dígitos: regras iguais entre si | 60/60 | 9/9 |

Leitura:
- **A limitação é real e grande**: um ruído de 1–3 ulp na pontuação (modelo de uma diferença de plataforma) altera as regras do Original em 52/60 ajustes sintéticos e em 9/9 DEV, e o fluxo de consultas ao oráculo em 48/60 e 6/9. Variações de fidelidade até 6 pontos percentuais por mero ruído de arredondamento são **ruído de medição** a descontar em qualquer comparação Reloaded × Original de diferenças pequenas (como as de B1/B2).
- A variante é **invariante** a esse ruído (0 mudanças) e ao complemento lógico (teste), e preserva a partição quando muda só a etiqueta.
- A variante **não é o Original**: em 18/60 (sint.) e 6/9 (DEV) escolhe outras regras; o efeito médio na fidelidade é pequeno e negativo na amostra (−0,0006; −0,0038) mas **não se interpreta como ganho/perda** (amostra pequena, sem teste externo; é medida de estabilidade, não de desempenho).
- Insensibilidade à quantização entre 6 e 14 dígitos: não existem, nesta amostra, scores a distância entre 1e-14 e 1e-6 que não sejam empates exatos; a escolha `12` é conservadora (acima do ruído de ulp ≈1e-16, abaixo de qualquer diferença de score com relevância científica). A tolerância não é ajustada a resultados.
- O ruído foi **injetado** (determinístico por máscara) — modela, não substitui, uma execução em outra plataforma (§5).

## 4. Complementos lógicos, ordem canónica e impacto nas regras

- Verificado: `m-of-n(L)` e o seu complemento definem partições complementares; `canonical_signature` é igual para ambos.
- Impacto nas regras: as regras exportadas podem diferir **só por troca de ramos** (partição igual). Isto explica por que há mais ajustes com regras diferentes (18/60) do que com previsões diferentes (11/60).
- Impacto no fluxo ao oráculo: um complemento troca o ramo de cada filho; os filhos recebem amostras diferentes → consultas diferentes. Por isso a invariância de fluxo só vale para a variante sob ruído, não entre variante e Original.

## 5. Estabilidade Linux × Windows (plano e estado)

- O CI (Linux; Windows 3.11/3.12) corre `tests/test_tiebreak_variant.py::test_cross_platform_digest_informational`, que **emite (sem falhar)** as impressões digitais das regras do Original e da variante em 4 problemas fixos como aviso `TIEBREAK_XPLAT_DIGEST …` no resumo do pytest.
- Critério de leitura (informativo): se os digests `O=` diferirem entre Linux e Windows ⇒ evidência direta de dependência da plataforma no Original; se os `V=` coincidirem entre plataformas ⇒ evidência de que a variante elimina essa dependência. Resultado: ver §7 (preenchido após o CI).

## 6. Política para B2 e para os baselines

1. Baseline canónico B2 = Original **como está** (com a limitação registada). A variante **não** é usada como baseline nem nos braços Reloaded da matriz principal.
2. Opcional (pré-registar antes de correr): análise de sensibilidade separada, "Original-det vs Reloaded-det" (o Reloaded herda a mesma chave canónica), apresentada como **variante experimental** com o seu próprio manifest; nunca combinada com a tabela canónica.
3. Qualquer comparação Reloaded × Original com |ΔF| < 0,06 por unidade deve ser lida à luz do ruído de arredondamento medido acima (por isso a avaliação por semente/unidade e as comparações emparelhadas multi-semente).
4. Decisão a aprovar: autorizar (ou não) a análise de sensibilidade "det" na fase de avaliação.

## 7. Resultado da verificação de plataformas

Windows 3.11.9 (CI, commit 50e6c1e; `TIEBREAK_XPLAT_DIGEST`) vs Linux 3.11.15 (local):

| Problema | Original (Linux = Windows?) | Variante (Linux = Windows?) |
|---|---|---|
| s0 | `95057818b5` = `95057818b5` ✔ | `95057818b5` = `95057818b5` ✔ |
| s3 | `bbe492cb64` = `bbe492cb64` ✔ | `26f52557b1` = `26f52557b1` ✔ |
| s5 | `42273fba86` = `42273fba86` ✔ | `42273fba86` = `42273fba86` ✔ |
| s7 | `1e9cdb597e` = `1e9cdb597e` ✔ | `024dd76d2d` = `024dd76d2d` ✔ |

Conclusão (prudente): **nestes 4 problemas sintéticos não se observou divergência Linux × Windows nem no Original nem na variante**. A dependência da plataforma do Original **não ficou demonstrada** em ambiente real; ficou demonstrada a **sensibilidade** ao ruído de 1–3 ulp (§3), que é o mecanismo pelo qual uma diferença de libm/BLAS/CPU se manifestaria. A amostra é pequena (4 problemas, oráculo de forma fechada sem BLAS pesado); ausência de divergência não prova invariância. Python 3.12 (CI): ver o resumo de avisos do job (não afeta a conclusão acima). Aconselhado: manter o digest no CI como vigia informativo (já ativo) e, se aparecer divergência, tratá-la como evidência direta da limitação.
