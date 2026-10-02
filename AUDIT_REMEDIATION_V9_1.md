# BIURI / TREPAN Reloaded V9.1 — execução das correções da auditoria

**Base:** `RELATORIO_AUDITORIA_BIURI_TREPAN_RELOADED_V9_1.pdf`  
**Estado desta entrega:** correções de código/contrato implementadas; validação que depende do ambiente-alvo Windows/Python 3.11/PyQt6/owlready2/Java/HermiT foi automatizada, mas não é falsamente marcada como concluída neste runtime Linux/Python 3.13.

## 1. Contrato científico de aceitação

Implementado no fluxo principal:

- `core/trepan.py` aplica `evaluate_selected_teacher_candidate()` depois da seleção por utility e antes do reajuste final.
- O gate é **conjuntivo**: `accuracy`, `macro_f1` e `balanced_accuracy` têm de ser não-inferiores ao MLP Original dentro da tolerância.
- Ganho de utility deixa de poder compensar uma queda numa métrica primária.
- Massa ontológica zero produz `accepted=False`.
- Quando o gate falha, o professor final passa explicitamente para `FeatureProjectedOracle`/MLP Original; o híbrido candidato fica apenas na auditoria.
- A GUI, quando recebe `accepted=False`, usa a versão base e não reabre a decisão observando o teste final.
- O diagnóstico de degenerescência que pode alterar o professor passou a usar **apenas desenvolvimento** (`test_used_for_selection=False`).

Os estados estão separados em:

- `ontology_feature_gate_accepted`
- `oracle_gate_accepted`
- `surrogate_gate_accepted`

## 2. Gate da árvore substituta e fallback real

Novo módulo `core/surrogate_acceptance.py`:

- exige não-inferioridade da árvore Reloaded em `accuracy`, `balanced_accuracy` e `macro_f1`;
- exige piso de fidelidade ao oráculo ativo;
- exige limite de complexidade;
- suporta, no benchmark confirmatório, não-inferioridade adicional contra o C4.5 como **baseline supervisionado**, não como oráculo TREPAN;
- a decisão de fallback só pode ocorrer em `development_acceptance_holdout`;
- no teste final o mesmo cálculo tem papel `locked_test_audit_only` e nunca seleciona modelo;
- em rejeição, o benchmark usa a árvore base real e recalcula as percentagens — não copia métricas.

## 3. Dois oráculos / duas fidelidades

`core/metrics_comparator.py` e a GUI passaram a distinguir:

- `fidelity_to_active_oracle`: Reloaded vs professor efetivamente usado;
- `fidelity_to_mlp_original`: referência de controlo comum com o TREPAN Original.

O gráfico comparativo de fidelidade usa a referência comum ao MLP Original. O C4.5 está identificado como baseline supervisionado e a sua concordância tem papel auxiliar.

## 4. Protocolo imutável e diagnóstico linha a linha

Novo `core/protocol_audit.py` cria SHA-256 determinísticos para:

- linhas lógicas do teste e `y_test`;
- matriz original e matriz Reloaded;
- schema original/enriquecido;
- ordem das classes;
- seed, número de repetições e identificador de pré-processamento;
- contagem de missing values.

A comparação também guarda, por linha bloqueada:

- ID da linha;
- rótulo real;
- previsão MLP Original;
- previsão do professor ativo;
- previsão TREPAN Original;
- previsão TREPAN Reloaded;
- valores `onto_*` usados;
- metadados/regra/conceito OWL quando disponíveis.

São calculadas ainda accuracy, balanced accuracy, macro-F1 e matriz de confusão do professor ativo no mesmo teste, apenas para relatório.

## 5. Proveniência e validação das features semânticas

`core/ontology_processor.py` e `core/semantic_utility_gate.py` agora registam, por `onto_*`:

- entidade/propriedade OWL de origem e colunas-fonte;
- regra de derivação;
- proveniência;
- razão de não duplicação/rejeição;
- taxa de valores constantes;
- estabilidade entre folds;
- mutual information média;
- efeito OOF individual em accuracy, balanced accuracy, macro-F1 e minority recall.

A evidência semântica é calculada apenas no desenvolvimento/OOF.

## 6. `ONTO_FEATURE_BIAS_WEIGHT`

O viés está explicitamente classificado como **heurística**, não como weighted information gain formal. Se o valor não for fornecido pelo utilizador/ambiente, é calibrado numa validação interna do desenvolvimento, com auditoria que contém:

- candidatos avaliados;
- peso selecionado;
- métricas internas;
- `test_used=False`;
- `formal_weighted_information_gain=False`.

## 7. Convergência dos MLPs

Novo `core/mlp_convergence.py` regista:

- `n_iter`;
- `max_iter`;
- `loss_curve` e perda final;
- solver, tolerância, early stopping/validation fraction;
- seed/random state;
- estado explícito de convergência.

`core/mlp_optimizer.py` prioriza candidatos convergidos antes do score quando há candidatos convergidos elegíveis. O treino residual pequeno foi alterado para `lbfgs`, `max_iter=2000`, `tol=1e-5` e passa a registar a convergência.

**Nota:** os 46 `ConvergenceWarning` vistos no subconjunto de regressão desta máquina vêm de fixtures de testes que instanciam deliberadamente `MLPClassifier` com `max_iter` 50/100/200/300. Eles continuam visíveis — não foram escondidos. O caminho de treino auditado passou a classificar e guardar convergência.

## 8. Métricas e linguagem científica

- Gráficos que representam proporção total de acertos passaram para **Accuracy / Exatidão**; `Precision / Precisão` fica reservada a `precision_score`.
- A GUI não cria um ranking global por média de fidelidades com referências diferentes.
- `claim_guard` é a autoridade para linguagem de superioridade. Só `superiority_supported` permite essa conclusão.
- Estimativa pontual maior é descrita como resultado daquela execução, não como prova de superioridade.

## 9. Empacotamento e execução limpa

Adicionados/ajustados:

- `pyproject.toml` — pacote instalável, Python alvo `>=3.11,<3.12`;
- `pytest.ini` — `testpaths` e `pythonpath=.`;
- `requirements-runtime.txt` — inclui `seaborn`, `owlready2`, `dtreeviz`, Graphviz etc.;
- `requirements-gui.txt` — runtime + PyQt6;
- `requirements-test.txt` — GUI + pytest;
- `requirements-win.txt` — wrapper compatível do ambiente Windows;
- `.github/workflows/windows-ci.yml` — Windows + Python 3.11 + Java + preflight + `python -m pytest -q`;
- `scripts/test_clean_windows.ps1` — reprodução local do CI;
- `scripts/environment_preflight.py` — valida Python, imports, Java e Graphviz;
- `scripts/run_confirmatory_locked.py` — launcher que exige preflight e confirmação `--execute` antes de consumir o teste externo único.

Imports opcionais de `dtreeviz`/`owlready2` foram endurecidos para que módulos não-OWL possam ser importados sem eles e para que o fluxo OWL falhe com mensagem acionável quando a dependência não está instalada.

## 10. Refatoração incremental

Em vez de reescrever os módulos gigantes de uma vez, responsabilidades críticas foram extraídas para módulos pequenos e testáveis:

- `protocol_audit.py` — identidade/reprodutibilidade;
- `surrogate_acceptance.py` — contrato de qualidade da árvore;
- `mlp_convergence.py` — convergência;
- `ontology_acceptance.py` — contrato do professor;
- `semantic_utility_gate.py` — evidência OOF semântica.

Isto reduz acoplamento preservando o comportamento coberto pela suíte existente. A decomposição física restante da GUI/extrator pode continuar incrementalmente sem misturar essa refatoração com a alteração científica.

## 11. Benchmark confirmatório pré-registado

`core/confirmatory_benchmark.py` agora congela, antes dos resultados:

- datasets e ontologias;
- seeds planeadas;
- split externo e holdout de aceitação;
- margens de não-inferioridade;
- piso de fidelidade e limite de complexidade;
- métricas primárias;
- Wilcoxon unilateral + bootstrap pareado 95%;
- política de seleção.

O teste externo é usado uma vez depois de congelar professor e árvore. Um lock impede repetir e escolher a execução favorável.

**Não foi consumida a execução confirmatória nesta máquina.** O preflight falhou corretamente antes de criar o diretório de saída porque o runtime disponível é Python 3.13.5 (não 3.11) e não tem `owlready2`/`dtreeviz`; a instalação não pôde ser feita sem acesso de rede. A execução única deve ser feita no ambiente alvo após a suíte Windows/GUI/OWL ficar verde.

## 12. Validação executada nesta entrega

| Verificação | Resultado |
|---|---|
| `python3 -m compileall -q core gui counterfactuals tests` | **OK**, sem SyntaxWarning após correção de duas strings legadas `N\\A` |
| Novos testes de remediação | **13 passed** |
| Subconjunto de regressão científico/sem GUI/sem OWL direto | **65 passed, 46 warnings** |
| 3 testes GUI exigidos pela auditoria | **Bloqueado no ambiente atual:** `PyQt6` ausente |
| Teste OWL direto | **Bloqueado no ambiente atual:** `owlready2` ausente |
| Preflight do ambiente alvo | **Não pronto:** Python 3.13.5, faltam `dtreeviz`, `owlready2`, `PyQt6`; Java e `dot` disponíveis |
| Benchmark confirmatório bloqueado | **Não consumido**; launcher aborta antes da execução enquanto preflight falha |

Evidências de execução estão em `results/audit_remediation_v9_1/`.

## 13. Sequência final no Windows alvo

```powershell
# Ambiente limpo, Python 3.11 x64 + Java instalados
powershell -ExecutionPolicy Bypass -File scripts/test_clean_windows.ps1

# Só depois de a suíte completa/GUI/OWL passar:
python scripts/run_confirmatory_locked.py
# rever o pré-registo/ambiente; para consumir a execução externa única:
python scripts/run_confirmatory_locked.py --execute
```

Não ajustar pesos, margens, seeds, ontologias ou hiperparâmetros depois de observar o resultado confirmatório. Se o `claim_guard` não for `superiority_supported`, o sistema deve manter a limitação científica e não alegar superioridade.
