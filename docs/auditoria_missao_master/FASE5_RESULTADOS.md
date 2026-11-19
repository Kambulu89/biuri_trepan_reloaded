# Fase 5 — resultados

Análise feita segundo `FASE5_PREREGISTO.md` (commit `8750958`, anterior à campanha). Dados em `evidencia/fase5/`. Natureza: **DEV, exploratória**.

## 1. Execução

- 15 de 15 unidades com código de saída 0, 10 braços × 3 réplicas, nenhum braço saltado. A unidade mais longa demorou 1183 s.
- Verificação independente das métricas: 3135 valores, diferença máxima 0.
- **Integridade: cumprida.** B reproduz a fase 3, O reproduz a fase 4 e D reproduz o D2 da fase 3 (diferença máxima 1,1 × 10⁻¹⁶ nos três).

## 2. Ablação

| Contraste | iris | wine | breast_cancer |
|---|---|---|---|
| accuracy: W − O | +0,000 [+0,000; +0,000] | +0,000 [+0,000; +0,000] | +0,000 [+0,000; +0,000] |
| fidelidade: W − O | +0,000 [+0,000; +0,000] | +0,000 [+0,000; +0,000] | +0,000 [+0,000; +0,000] |
| accuracy: E − W | +0,000 [-0,005; +0,005] | -0,013 [-0,022; +0,002] | -0,003 [-0,005; -0,001] |
| fidelidade: E − W | -0,000 [-0,005; +0,005] | -0,009 [-0,018; +0,003] | -0,001 [-0,004; +0,002] |
| accuracy: A − W | -0,007 [-0,023; +0,007] | +0,013 [+0,000; +0,025] | -0,005 [-0,015; +0,005] |
| fidelidade: A − W | -0,000 [-0,011; +0,011] | +0,019 [+0,002; +0,037] | -0,002 [-0,011; +0,009] |
| accuracy: D − O | -0,012 [-0,019; -0,009] | -0,019 [-0,040; +0,000] | -0,009 [-0,019; +0,001] |
| fidelidade: D − O | +0,009 [-0,005; +0,023] | -0,007 [-0,024; +0,007] | -0,008 [-0,013; -0,001] |

- **W − O é exatamente zero em todas as unidades.** O canal de score (peso de ganho, Gain_Reloaded, candidatos m-of-n por grupos) produziu árvores idênticas às de O, apesar de influenciar a pontuação de 16 a 18 divisões por árvore. Nenhuma decisão foi alterada.
- **E − W (EFSR):** accuracy −0,3 pontos no breast_cancer com intervalo abaixo de zero; −1,3 no wine (4 de 5 unidades negativas); zero no iris. O EFSR alterou 2,7 a 5,1 decisões por árvore e as alterações pioraram ou não mudaram o resultado.
- **A − W (consultas ativas):** no wine a fidelidade sobe 1,9 pontos com intervalo acima de zero, e a accuracy 1,3 pontos (limite inferior em zero); no iris e no breast_cancer as diferenças são negativas mas os intervalos incluem zero. As consultas ativas não alteram decisões diretamente (0 alteradas); mudam a amostra de decisão.
- **D − O** repete a fase 4: accuracy −1,2 pontos no iris e fidelidade −0,8 no breast_cancer, com intervalo abaixo de zero.

### Porque é que o canal de score é inerte

No `_best_mofn` do Reloaded (`core/trepan_reloaded_historical.py`), quando a divisão proposta pela semântica difere da melhor divisão estatística, só é aceite se o EFSR a aprovar, com base na fidelidade local ao MLP; caso contrário, o Reloaded volta exatamente à divisão estatística. Com o EFSR desligado nenhuma proposta é aprovada. Assim:

1. o peso fixo e o Gain_Reloaded **não causam** a perda; a minha hipótese da fase 4 de que o peso fixo desviava a procura estava **errada**;
2. a perda vem das decisões que o **EFSR aprova**; o gate mede a fidelidade na própria amostra de decisão do nó, o que favorece alterações que não se generalizam.

## 3. Correções candidatas

| Contraste | iris | wine | breast_cancer |
|---|---|---|---|
| accuracy: T − O | +0,000 [+0,000; +0,000] | +0,000 [+0,000; +0,000] | +0,000 [+0,000; +0,000] |
| fidelidade: T − O | +0,000 [+0,000; +0,000] | +0,000 [+0,000; +0,000] | +0,000 [+0,000; +0,000] |
| accuracy: L − O | +0,000 [+0,000; +0,000] | +0,000 [+0,000; +0,000] | +0,000 [+0,000; +0,000] |
| fidelidade: L − O | +0,000 [+0,000; +0,000] | +0,000 [+0,000; +0,000] | +0,000 [+0,000; +0,000] |
| accuracy: T − Tq | +0,011 [+0,002; +0,019] | -0,019 [-0,038; +0,000] | +0,001 [-0,015; +0,016] |
| fidelidade: T − Tq | +0,007 [+0,000; +0,018] | -0,025 [-0,036; -0,012] | +0,001 [-0,009; +0,010] |
| accuracy: L − Lq | +0,011 [+0,002; +0,019] | -0,019 [-0,038; +0,000] | +0,001 [-0,015; +0,017] |
| fidelidade: L − Lq | +0,007 [+0,000; +0,018] | -0,025 [-0,036; -0,012] | +0,001 [-0,009; +0,010] |

T e L são idênticos a O pelo mesmo motivo: ambas atuam só na pontuação, e com o EFSR desligado nenhuma proposta semântica passa. As diferenças face às versões permutadas (Tq, Lq) são as mesmas que O − Q na fase 4, porque a ontologia permutada muda os atributos do espaço, não as decisões.

**Falha do meu desenho:** as correções pré-registadas eram inertes por construção. Só o descobri com os resultados. Não as substituo agora: seria escolher a correção depois de ver os dados.

## 4. Escolha da correção (regra pré-registada)

| Candidata | Elegível | Média de (Δacc + Δfid)/2 face a O | Intervalos abaixo de zero |
|---|---|---|---|
| W | sim | +0,0000 | — |
| T | sim | +0,0000 | — |
| L | sim | +0,0000 | — |

Nenhuma candidata elegível tem média > 0, por isso **a regra escolhe O**: o Reloaded corrigido é o Original no espaço enriquecido pela ontologia, sem mecanismos. Fica congelado para a fase 6 (que já estava pré-registada antes destes resultados, commit `521ce2d`).

## 5. Meta: Reloaded corrigido ≥ Original

| Contraste | iris | wine | breast_cancer |
|---|---|---|---|
| accuracy: O − B | -0,000 [-0,018; +0,012] | -0,006 [-0,033; +0,012] | +0,005 [-0,005; +0,014] |
| fidelidade: O − B | +0,000 [-0,014; +0,011] | -0,013 [-0,038; +0,006] | +0,002 [-0,005; +0,009] |
| accuracy: T − B | -0,000 [-0,018; +0,012] | -0,006 [-0,033; +0,012] | +0,005 [-0,005; +0,014] |
| fidelidade: T − B | +0,000 [-0,014; +0,012] | -0,013 [-0,038; +0,006] | +0,002 [-0,005; +0,009] |
| accuracy: L − B | -0,000 [-0,018; +0,012] | -0,006 [-0,033; +0,012] | +0,005 [-0,005; +0,014] |
| fidelidade: L − B | +0,000 [-0,014; +0,011] | -0,013 [-0,037; +0,006] | +0,002 [-0,005; +0,009] |
| accuracy: W − B | -0,000 [-0,018; +0,012] | -0,006 [-0,033; +0,012] | +0,005 [-0,005; +0,014] |
| fidelidade: W − B | +0,000 [-0,014; +0,011] | -0,013 [-0,038; +0,006] | +0,002 [-0,005; +0,008] |

Critério de não-inferioridade observada (média ≥ 0 e limite inferior ≥ −0,02, nas duas métricas): **cumprido no iris e no breast_cancer; não cumprido no wine**, onde a média é ligeiramente negativa e o limite inferior chega a −0,033 em accuracy e −0,039 em fidelidade.

## 6. Médias por braço

| Dataset | Braço | Accuracy | Fidelidade | Nós | Consultas da árvore final | Decisões alteradas pela semântica | Tempo (s) |
|---|---|---|---|---|---|---|---|
| iris | C4.5 | 0,958 | 0,963 | N/A | N/A | N/A | N/A |
| iris | B | 0,956 | 0,965 | 35,0 | 10337 | 0,0 | 0,3 |
| iris | O | 0,956 | 0,965 | 32,6 | 10856 | 0,0 | 1,9 |
| iris | W | 0,956 | 0,965 | 32,6 | 10856 | 0,0 | 2,8 |
| iris | E | 0,956 | 0,965 | 33,3 | 10793 | 2,7 | 2,6 |
| iris | A | 0,949 | 0,965 | 32,2 | 10058 | 0,0 | 3,1 |
| iris | D | 0,944 | 0,974 | 34,7 | 10410 | 2,4 | 2,7 |
| iris | T | 0,956 | 0,965 | 32,6 | 10856 | 0,0 | 2,9 |
| iris | L | 0,956 | 0,965 | 32,6 | 10856 | 0,0 | 2,9 |
| iris | Tq | 0,946 | 0,958 | 34,2 | 10636 | 0,0 | 2,6 |
| iris | Lq | 0,946 | 0,958 | 34,2 | 10636 | 0,0 | 2,8 |
| wine | C4.5 | 0,929 | 0,916 | N/A | N/A | N/A | N/A |
| wine | B | 0,948 | 0,939 | 29,0 | 10615 | 0,0 | 2,2 |
| wine | O | 0,942 | 0,926 | 37,1 | 11332 | 0,0 | 10,9 |
| wine | W | 0,942 | 0,926 | 37,1 | 11332 | 0,0 | 17,2 |
| wine | E | 0,929 | 0,917 | 31,4 | 10841 | 5,1 | 15,0 |
| wine | A | 0,956 | 0,945 | 31,9 | 11529 | 0,0 | 20,6 |
| wine | D | 0,923 | 0,919 | 34,2 | 11414 | 4,9 | 16,8 |
| wine | T | 0,942 | 0,926 | 37,1 | 11332 | 0,0 | 18,5 |
| wine | L | 0,942 | 0,926 | 37,1 | 11332 | 0,0 | 18,5 |
| wine | Tq | 0,961 | 0,951 | 33,7 | 11826 | 0,0 | 19,3 |
| wine | Lq | 0,961 | 0,951 | 33,7 | 11826 | 0,0 | 19,0 |
| breast_cancer | C4.5 | 0,930 | 0,923 | N/A | N/A | N/A | N/A |
| breast_cancer | B | 0,937 | 0,935 | 35,9 | 16725 | 0,0 | 2,9 |
| breast_cancer | O | 0,942 | 0,937 | 34,5 | 17357 | 0,0 | 11,1 |
| breast_cancer | W | 0,942 | 0,937 | 34,5 | 17357 | 0,0 | 20,2 |
| breast_cancer | E | 0,939 | 0,936 | 35,0 | 17019 | 4,6 | 19,1 |
| breast_cancer | A | 0,937 | 0,934 | 35,7 | 16965 | 0,0 | 23,7 |
| breast_cancer | D | 0,933 | 0,929 | 35,3 | 17458 | 5,3 | 21,9 |
| breast_cancer | T | 0,942 | 0,937 | 34,5 | 17357 | 0,0 | 22,3 |
| breast_cancer | L | 0,942 | 0,937 | 34,5 | 17357 | 0,0 | 23,1 |
| breast_cancer | Tq | 0,940 | 0,936 | 34,7 | 17609 | 0,0 | 21,6 |
| breast_cancer | Lq | 0,940 | 0,936 | 34,7 | 17609 | 0,0 | 21,1 |

## 7. Conclusão da fase

1. **O que degrada o Reloaded é o EFSR**, não o peso nem o ganho modificado.
2. **O desenho atual já é conservador:** a semântica só muda uma decisão se um gate local a aprovar. O problema não é falta de conservadorismo; é o gate aprovar alterações com base na fidelidade medida na própria amostra do nó.
3. **Sem o EFSR, a ontologia não altera nenhuma decisão** nestes datasets e o Reloaded fica igual ao Original com mais colunas.
4. **A meta «Reloaded ≥ Original» cumpre-se em dois dos três datasets**, e só no sentido fraco de não-inferioridade.

## 8. O que ficaria por testar (não pré-registado, não executado)

- Um gate do EFSR validado fora da amostra de decisão do nó (por exemplo, em linhas reais retidas), que só aprove uma proposta semântica se ela melhorar a fidelidade nessas linhas.
- O desempate (T) com esse gate ligado.

Qualquer destas exigiria um pré-registo novo, avaliado em datasets que não estes.
