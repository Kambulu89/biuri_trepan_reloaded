# BIURI / TREPAN Reloaded — regras permanentes do projeto

## 1. O sistema é AGNÓSTICO a datasets (regra inviolável)
- Em produção serão usados **outros datasets**, desconhecidos. Iris, Wine, Breast Cancer, Digits e os sintéticos são **apenas** o conjunto de
  validação/regressão offline; nunca definem o comportamento do núcleo.
- Proibido no núcleo (`core/`, `gui/`): `if dataset_name == ...`, hiperparâmetros por dataset, nomes de datasets conhecidos, nº fixo de
  features/classes, limites de recursos ou de capacidade ligados à identidade do dataset, atalhos "só para o Digits/Wine/...".
- Os datasets conhecidos vivem só em `validation/` (benchmarks em `validation/benchmark/`). Acrescentar um dataset = acrescentar **dados +
  metadados** (`DatasetSpec`), nunca alterar o algoritmo.
- Qualquer correção motivada por um dataset só entra no núcleo se for uma propriedade **geral** do problema/dados. Um problema que aparece
  só num dataset investiga-se como violação de uma regra genérica, não com um ramo específico.
- Otimizações de desempenho: genéricas (vectorização, cache com chave matemática exata, retoma/checkpoint), nunca por dataset; políticas de
  recursos são globais e configuráveis (`--max-wall-time-per-unit`), nunca trocam silenciosamente a configuração científica.
- **Contrato executável (não depende só desta instrução):** `scripts/check_dataset_agnosticism.py` (estático, só stdlib: nomes de datasets —
  incluindo todos os `dataset_id` do registo —, ramificações/tabelas/políticas por identidade, formas fixas, `core → validation`) e
  `tests/test_agnosticism_contract.py` (mutation tests do verificador, metamórficos, dataset desconhecido, dimensionalidade, classes, tipos de
  features, ontologia, política de recursos global, CI). A CI `agnosticism-guard` (`.github/workflows/agnosticism.yml`) falha qualquer PR que
  viole o contrato; deve estar marcada como *required status check* na proteção do ramo. Testes anteriores que também o impõem:
  `tests/test_dataset_agnosticism.py`, `tests/test_v92_static_guards.py`.
- Dependência permitida: `validation → core`. Nunca `core → validation`.

## 2. Desenho experimental congelado
- Não alterar (nem em resposta a resultados): tuning científico (CV 5×3, purity_epsilon, max_nodes, política de expansão, critérios
  lexicográficos e de estabilidade, bootstrap diagnóstico) nem o contrato do oráculo (FrozenOracle/oracle_id).
- Braços A–F, protocolo estrutural comum e ontology-blind, mirror desligado, C4.5 canónico: ver `validation/benchmark/BENCHMARK_MANIFEST_v*.json`.
- **REQUISITO CIENTÍFICO OBRIGATÓRIO (hierarquia conjunta).** O TREPAN Reloaded enriquecido por ontologia OWL deve ser otimizado e avaliado para alcançar **simultaneamente**:
  - *Accuracy (rótulos reais):* `Reloaded+OWL > Original`, `Reloaded+OWL > C4.5` e `Original > C4.5`.
  - *Fidelidade ao MLP:* `Reloaded+OWL > Original` e `Reloaded+OWL > C4.5`.
  Todos medidos **nas mesmas instâncias** e, para a fidelidade, contra o **mesmo MLP/oráculo de referência** (FrozenOracle/`oracle_id`). Uma intervenção (B1–B4, amostragem, mecanismos
  semânticos) **só é cientificamente bem-sucedida se as cinco condições se verificarem em conjunto**: melhorar a accuracy à custa da fidelidade (ou o inverso) é um **compromisso**
  (`TRADEOFF_*`), nunca um sucesso. Contrato executável: `core/hierarchy_contract.py` (`evaluate_joint`, estados por unidade `JOINT_HIERARCHY_SATISFIED|TRADEOFF|NOT_SATISFIED|HIERARCHY_INFEASIBLE|INSUFFICIENT_EVIDENCE`). Se `Original ≤ C4.5` em accuracy numa unidade, a cadeia completa é `HIERARCHY_INFEASIBLE` com os resultados fixados dos baselines: reporta-se, **não se corrige**, **nenhuma unidade/dataset/seed é excluída** (percentagens sempre sobre todas as unidades; não é impossibilidade universal).
  **Dois níveis, sempre separados:** Nível 1 = ganho ontológico direto (Reloaded+OWL vs Original *equivalente*: mesmo amostrador, oráculo, partições, orçamento; atribuição à ontologia só com ganho conjunto sobre controlos permutado, grupos aleatórios e engenharia convencional); Nível 2 = cadeia global U1–U5. O Nível 1 não substitui o 2. Não-inferioridade (`≥`, filtro de elegibilidade em validação) ≠ superioridade estrita (`>`). Estado lógico ≠ evidência (`OBSERVED_JOINT_GAIN|SUPPORTED_JOINT_GAIN`); unidade de inferência = dataset; DEV já observado nunca é confirmatório.
  É o **objetivo** de otimização/seleção (só treino e validação interna) e o critério de aceitação — **nunca** um resultado imposto, e nenhuma superioridade se declara sem evidência experimental
  (estado `observed_only` vs `statistically_sustained`; réplicas/unidades como no protocolo).
- **Métricas:** a métrica preditiva principal é a **accuracy** em relação aos rótulos reais (`accuracy_real_labels`); **accuracy ≠ precision**. `precision_macro_real_labels` e `macro_f1_real_labels`
  (e balanced accuracy, recall) são **complementares**: reportam-se sempre ao lado, nunca substituem a accuracy nem servem para declarar a hierarquia. A fidelidade é `fidelity_to_oracle`.
- Continua proibido: alterar ou degradar os baselines (Original, C4.5, MLP, oráculo congelado), alterar resultados históricos, usar o teste externo na seleção/otimização, escolher seeds/datasets/ontologias
  depois de ver resultados, repetir unidades até obter a ordem, codificar a ordem, introduzir melhorias independentes da ontologia só no Reloaded. Com ontologia, o ganho tem de vir de mecanismos
  ontológicos reais e verificáveis (controlos permutado/aleatório/convencional) e o sistema permanece dataset-agnóstico. Resultados inesperados são evidência a investigar e reportam-se
  (estados acima). Documento de referência: `docs/b1b4/28_REQUISITO_HIERARQUIA_CONJUNTA.md`.

## 2b. Arquitetura científica: Reloaded = Original + extensão ontológica (contrato de equivalência)
- O TREPAN Reloaded **não** é um algoritmo diferente do Original: é o Original **acrescido de mecanismos de enriquecimento semântico derivados de uma ontologia OWL válida**.
- **Contrato de equivalência (EQ):** `Reloaded(dataset, ontology=None) ≡ Original(dataset)` com o mesmo oráculo congelado, dados, parâmetros, seed e condições — igualdade em
  construção da árvore, seleção de divisões, regras, m-of-n, **fluxo de consultas ao oráculo**, previsões, fidelidade, accuracy e complexidade. O mesmo vale se a ontologia for
  rejeitada pelo quality gate ou não produzir estrutura semântica ativa. Diferenças só são aceites se forem não semânticas, justificadas e **previamente autorizadas** (allowlist
  versionada, vazia por omissão). Formalização e testes: `docs/b1b4/07_CONTRATO_EQUIVALENCIA.md`.
- **Contrato EQ v2.0 (estados S0–S3):** S0 ontologia ausente, S1 rejeitada, S2 válida sem mecanismo semântico executado → equivalência **integral** E1–E9. S3 válida com tentativa semântica
  executada → porção Original ≡ Original isolado **+ tentativa semântica auditada** (todas as consultas adicionais atribuídas e contabilizadas em `oracle_accounting_`, nunca ocultas). Tentativa
  executada ≠ ganho: sem melhoria observada o estado é `SEMANTIC_ATTEMPT_NO_OBSERVED_GAIN`. O TREPAN Original canónico não se modifica (inclui o desempate m-of-n: limitação de
  reprodutibilidade registada; política determinística só como variante experimental isolada).
- Com ontologia efetiva só são permitidos mecanismos **ontológicos** (matching, hierarquia/domínio, enriquecimento de representação, restrições, orientação semântica de splits,
  calibração do contributo semântico, aceitação/controlo de splits influenciados pela semântica, seleção de configurações do Reloaded enriquecido, m-of-n guiado por relações
  OWL), cada um controlável por flag, auditável e com controlo negativo (ontologia permutada/aleatória) e controlo de engenharia convencional de features.
- Proibido no Reloaded: melhorias independentes da ontologia (arquitetura, hiperparâmetros, pré-processamento, oráculo, procura, poda, uso de rótulos reais) que o tornem diferente
  do Original sem ontologia. Ganhos só se atribuem à ontologia se medidos contra o **Original equivalente** nas mesmas condições (mesma representação, oráculo, orçamento, seed).
- Baselines (TREPAN Original, C4.5 canónico, tuning estrutural ontology-blind, FrozenOracle) permanecem congelados por hash; o Reloaded só evolui por mecanismos ontológicos sob
  um novo manifest versionado.

## 3. Processo
- Commits pequenos e lógicos; `git status` antes de commitar; não versionar caches/temporários (`.pyc`, `results/` pesados).
- Merge para `main` só a pedido explícito. Não criar PR sem pedido.
- Resultados brutos são imutáveis e as tabelas reconstroem-se só deles; equivalência de otimizações provada com
  `validation/benchmark/equivalence_report.py` (comparação exata).
