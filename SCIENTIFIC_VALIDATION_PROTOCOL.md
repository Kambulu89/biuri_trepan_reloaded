# Protocolo científico — Trepan Reloaded

## O que esta versão corrige

Esta versão não força o Trepan Reloaded a “ganhar”. Ela remove mecanismos que
podiam produzir uma vitória aparente sem validade científica e acrescenta uma
forma explícita de decidir quando a superioridade é, ou não, suportada.

1. **Teste final reservado.** Hiperparâmetros do MLP e configurações das árvores
   são escolhidos por validação interna do treino, nunca pelo resultado do teste.
2. **Seleção ontológica treino-only.** Mutual information e remoção de colunas
   quase constantes são ajustadas no treino e aplicadas ao teste pelos mesmos
   índices.
3. **Residual sem previsões in-sample.** O MLP Residual Ontológico usa
   probabilidades out-of-fold no seu treino e probabilidades do modelo externo
   apenas na validação/teste.
4. **Aceitação conservadora da ontologia.** Accuracy, F1 e balanced accuracy
   precisam respeitar a margem de não-inferioridade; um ganho isolado já não
   mascara degradação nas outras métricas.
5. **Schemas estritos.** Uma matriz com número errado de features produz erro
   explicativo. Não há truncagem nem zero-padding silencioso nas métricas.
6. **Destilação probabilística.** A árvore Reloaded recebe alvos hard e soft do
   oráculo, com temperatura configurável, para aprender melhor regiões de
   fronteira sem consultar os rótulos do teste.
7. **Sem pseudo-bias por escala.** Multiplicar monotonamente uma feature não
   aumenta o seu ganho de informação numa árvore univariada. A prioridade
   semântica é implementada por amostragem, pesos e coerência, mantendo os
   limiares no espaço original.
8. **Comparação pareada.** O relatório usa bootstrap pareado, McNemar exato e a
   mesma referência MLP Original para comparar fidelidades.
9. **Consultas ativas auditáveis.** O orçamento, discordância, entropia,
   contrafactuais selecionados e decisões de aceitação ficam no audit; o teste
   final nunca entra no pool.
10. **Seleção multiobjetivo.** A árvore final pertence à frente de Pareto entre
    fidelidade, desempenho, robustez, coerência semântica e simplicidade, com
    pisos de não-inferioridade face à árvore anterior.
11. **Guardião clínico.** Ausência de split por paciente, validação de site,
    calibração ou validação externa bloqueia automaticamente alegações clínicas.

## Regra para alegar superioridade

Uma execução só suporta a alegação forte quando, simultaneamente:

- o limite inferior do IC 95% de `accuracy(Reloaded) - accuracy(baseline)` é
  superior à margem predefinida;
- a fidelidade ao **mesmo MLP Original** é não-inferior (margem padrão: 1 ponto
  percentual);
- McNemar é significativo (`p < 0,05`);
- os critérios são satisfeitos contra Trepan Original e contra o baseline de
  árvore.

Se os critérios falharem, a aplicação devolve `non_inferior` ou
`inconclusive_or_inferior` e bloqueia a conclusão forte. Um gráfico com barras
maiores não constitui prova.

## Baseline C4.5 nativo

O baseline anterior, baseado em CART com entropia, foi removido. O projeto usa
agora `C45Classifier`, uma implementação Python nativa com:

- seleção por Gain Ratio, incluindo o filtro de ganho médio de C4.5;
- limiares contínuos em fronteiras de classe;
- ramos multivalor para atributos nominais;
- distribuição fracionária de instâncias com valores ausentes;
- poda pessimista controlada por `confidence_factor`;
- probabilidades de folha com correção de Laplace.

O modelo é identificado como **C4.5-Nativo**. Não deve ser chamado J48, porque
J48 é a implementação específica do Weka e o projeto não integra Weka nem Java.
Para uma alegação científica, descreva os componentes acima, publique os
hiperparâmetros e compare todos os modelos nas mesmas partições externas.

## Protocolo recomendado para resultados publicáveis

### Dados tabulares gerais

- outer repeated stratified CV (por exemplo 5 folds × 10 repetições);
- nested CV para MLP, ontologia, poda e profundidade;
- mesma partição e mesmas linhas para todos os modelos;
- média, desvio, IC 95%, McNemar por repetição e teste agregado corrigido;
- accuracy, balanced accuracy, macro-F1, weighted precision, fidelidade ao mesmo
  oráculo, nós, folhas, profundidade, tempo e estabilidade de regras;
- relatório de ablação: Original; +ontologia; +OOF residual; +soft labels;
  +contrafactuais; configuração completa.

### HealthTech/MedTech

- split por paciente, episódio, hospital ou aquisição; nunca por imagem quando
  várias imagens pertencem ao mesmo paciente;
- validação temporal e validação externa por centro/dispositivo;
- sensibilidade, especificidade, PPV, NPV, AUROC, AUPRC, calibração e Brier;
- intervalos por bootstrap ao nível do paciente;
- subgrupos clínicos/demográficos, missingness e análise de drift;
- decision-curve analysis e limiares definidos pelo uso clínico;
- ontologia versionada, proveniência e revisão das regras por especialista;
- nenhuma alegação clínica baseada apenas nos datasets de demonstração.

## Como interpretar os campos novos

- `fidelity_reference`: oráculo ativo usado para treinar o Reloaded.
- `active_oracle_fidelity`: fidelidade ao oráculo ativo.
- `fidelity_to_mlp_original`: fidelidade controlada, comparável com Trepan
  Original e C4.5-Nativo.
- `scientific_validation.comparisons`: diferenças pareadas e ICs.
- `strong_superiority_claim_supported`: única flag autorizada para uma alegação
  forte dentro daquela partição.
- `claim_guard=do_not_claim_superiority_from_point_estimates`: resultado ainda
  não demonstrado.

## Limites honestos

Nenhum algoritmo pode garantir superioridade em todos os datasets (teorema “no
free lunch”). Ontologias incorretas ou desalinhadas podem prejudicar o modelo; a
aceitação conservadora existe para recuar ao MLP Original. A destilação melhora
frequentemente fidelidade, mas pode aumentar a árvore; por isso a decisão final
deve considerar o Pareto entre desempenho, fidelidade, coerência semântica e
complexidade.

## Benchmark de engenharia incluído

Execute:

```powershell
python run_ablation_study.py
```

O runner usa seis datasets, duas repetições, holdout estratificado bloqueado e
as variantes hard residual OOF, destilação, consultas ativas e configuração
completa. Os artefactos são escritos em `results/ablation/`. Este benchmark
serve para regressão e planeamento experimental; não substitui nested CV nem
validação externa por centro.
