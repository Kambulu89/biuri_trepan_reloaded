# Scientific changelog

## 2026-08 — substituição do proxy CART pelo C4.5 nativo

- removido o CART-Entropy do treino, da interface e dos relatórios atuais;
- implementado Gain Ratio com filtro de ganho médio de Quinlan;
- adicionados ramos nominais multivalor e valores ausentes fracionários;
- adicionada poda pessimista por `confidence_factor`;
- mantido o C4.5 estritamente no espaço ARFF original e com rótulos reais;
- painel comparativo limitado a TREPAN Original, C4.5-Nativo e TREPAN Reloaded;
- MLP preservada apenas como oráculo interno para a fidelidade;
- Weka/Java não são dependências do projeto.

## 2026-08 — protocolo sem leakage e prova estatística

- seleção MLP por CV/holdout interno do treino;
- configuração de Trepan Original/Reloaded sem acesso ao teste final;
- seleção de features ontológicas ajustada apenas no treino;
- probabilidades OOF no MLP Residual Ontológico;
- aceitação ontológica por não-inferioridade conjunta;
- schemas estritos nas métricas;
- destilação com soft targets e temperatura;
- fidelidade Reloaded reportada contra oráculo ativo e MLP Original comum;
- bootstrap pareado, McNemar exato e guard de alegação forte;
- CV inválida de árvores substitutas pré-extraídas removida;
- baseline histórico CART-Entropy identificado e posteriormente substituído por C4.5-Nativo;
- gráfico identifica precisão ponderada, evitando confusão com accuracy;
- documentação de avaliação clínica, ablação e validação externa.

## 2026-08 — implementação integral dos dez controlos

1. `EvaluationProtocolGuard` bloqueia teste/external-test em seleção e uma
   segunda avaliação final na mesma execução.
2. Registo explícito de identidade do algoritmo e espaço de features; o
   residual é consultado por um wrapper no espaço enriquecido, preservando OWL.
3. OOF estratificado ou por paciente/grupo, com auditoria de cobertura por linha.
4. Schemas nominais e fingerprint; truncamento, alinhamento posicional e
   zero-padding silencioso foram removidos.
5. UI mostra accuracy, balanced accuracy, macro-F1, fidelidade, oráculo, espaço
   e guardião de alegações.
6. Destilação por massa probabilística, temperatura e ênfase na entropia.
7. Consultas ativas iterativas por discordância, incerteza, diversidade e
   proximidade contrafactual, usando apenas validação interna.
8. Poda cost-complexity e seleção Pareto com pisos de não-inferioridade para
   fidelidade e accuracy.
9. Validação biomédica: sensibilidade, especificidade, PPV/NPV, AUROC/AUPRC,
   Brier, ECE, calibração, decision curve, paciente/site/tempo e subgrupos.
10. Runner de ablação para o dataset carregado e benchmark reprodutível em seis
    datasets, com JSON/CSV/Markdown.

O carregamento de CLEAR tornou-se opcional: COGS e o motor interno não falham
quando TensorFlow/CLEAR está ausente; CLEAR falha de forma localizada e clara.

## 2026-08-29 — pipeline ontológico v6

- enriquecimento substituído por `OntologyProcessor.fit/transform`, com `fit`
  exclusivamente no treino externo;
- TBox e ABox auditadas separadamente; instâncias semelhantes a registos ou
  identificadores do teste bloqueiam a ontologia;
- schema estrito e eliminação de features constantes/duplicadas, sem
  truncamento ou zero-padding;
- quality gate com cobertura, score, ambiguidade e rejeição de ontologias
  genéricas;
- indivíduos de vocabulário controlado passam a mapear valores categóricos;
- reasoner HermiT/Pellet executado de verdade e falhas não são aceites como
  consistência;
- ordem nominal da classe preservada a partir do cabeçalho ARFF;
- benchmark refeito com TBoxes de Iris, Wine e Breast Cancer, 9 pares
  `sem OWL`/`com OWL` e teste externo bloqueado.

## V9.2 — Dataset-agnostic training hardening

- Introduzido contrato de dados explícito e serializável, evitando inferência silenciosa do alvo por posição.
- Pré-processamento passa a ser ajustado apenas no treino no caminho V9.2, com suporte seguro a categorias desconhecidas e valores em falta.
- A selecção de candidatos MLP em `run_full_mlp_optimization` deixa de consultar o holdout externo; o modelo escolhido é avaliado apenas depois da escolha.
- Calibração é aceite/rejeitada a partir de validação interna do treino; métricas reportadas depois descrevem o modelo final entregue.
- A ablação V9.2 usa a fábrica adaptativa de MLP e regista gate de capacidade do oráculo.
- O extractor CART historicamente denominado TREPAN é agora explicitamente distinguido de uma implementação canónica best-first/m-of-n.
- Nenhum resultado de `results/confirmatory_v7/` foi regenerado nesta implementação.

## 2026-09-29 — TREPAN Reloaded alinhado ao núcleo histórico

- `TrepanReloadedClassifier` passa a herdar directamente de `TrepanOriginalClassifier`.
- Semântica OWL actua no score dos splits e na geração/projectação das membership queries, não através de CART intermediário.
- Caminho principal OWL deixa de usar active-query CART, soft-tree, redistillation e pruning sklearn.
- Artefacto TREPAN deixa de manter referência ao MLP/oráculo após o treino.
- Validação dirigida: 30 testes aprovados; confirmatório V7 preservado byte-a-byte por SHA-256.

### V9.2 — ablação pareada sem troca de oráculo
A comparação Original vs Reloaded foi endurecida para impedir o confundidor de professor. Ambos consultam o mesmo MLP Original; no espaço OWL, `OriginalOracleProjection` apenas projecta as features antes da consulta. As estatísticas agregadas usam dataset como unidade e excluem oráculos que falham o gate de treino contra Dummy/C4.5.

## V9.2 — Semantic Real-Gain Contract

- A contribuição OWL passa a ser validada localmente também em dados reais internos do treino.
- O EFSR usa regiões prioritárias de erro com fallback top-k, sem consultar teste externo.
- Quando não existe efeito semântico mensurável, o Reloaded final é exactamente o TREPAN Original.
- A identidade/versionamento/hash da OWL ficam auditáveis na GUI e nos relatórios.
