# Execução local no Windows (VS Code) — BIURI / TREPAN Reloaded V9.2

Todos os comandos abaixo foram retirados da estrutura real do repositório (`pyproject.toml`, `requirements-*.txt`, `run_biuri.py`,
`scripts/environment_preflight.py`, `.github/workflows/windows-ci.yml`) e a instalação foi verificada num ambiente virtual limpo
(`pip install -r requirements-win.txt` → `environment_preflight.py --gui --owl` com `ready: true` e `BiuriApp()` instanciada).

## 1. Pré-requisitos
| Item | Versão | Notas |
|---|---|---|
| Python 64-bit | **3.11.x (recomendado)**; 3.12 também testado na CI | `pyproject.toml`: `>=3.11,<3.13`. Os *pins* de PyQt6 (6.4.2) estão validados para 3.11; em 3.12 instala-se PyQt6 ≥ 6.6, < 6.9 automaticamente |
| Java (JRE/JDK 64-bit) | 17 (Temurin recomendado) | só para o raciocinador OWL (HermiT). Defina `JAVA_HOME` ou `OWLREADY2_JAVA_EXE` (ver `docs/REASONER_WINDOWS_V9_1.md`) |
| Git | qualquer recente | |
| Graphviz (`dot`) | opcional | só para exportar a árvore como imagem; a ausência nunca impede treino, MLP, ontologia ou TREPAN |

## 2. Dependências (de `requirements-win.txt` → `-r requirements-gui.txt` → `-r requirements-core.txt`, e `-r requirements-owl.txt`)
numpy 1.26.4 · scipy 1.15.2 · pandas 2.2.3 · scikit-learn 1.6.1 · statsmodels 0.14.6 · patsy 1.0.2 · matplotlib 3.10.1 · seaborn 0.13.2 ·
graphviz 0.20.3 · dtreeviz 2.2.2 · liac-arff 2.5.0 · optuna 4.2.1 · joblib 1.4.2 · jinja2 3.1.6 · sympy 1.13.3 ·
PyQt6 6.4.2 + PyQt6-Qt6 6.4.3 (Python 3.11) ou PyQt6 ≥ 6.6,<6.9 (Python 3.12) · owlready2 0.47.
Para correr os testes: `requirements-test.txt` (acrescenta pytest 8.3.5, pytest-cov, pyflakes). TensorFlow/CLEAR é opcional (`requirements-clear.txt`).

## 3. Clonar / atualizar (PowerShell, no VS Code: Terminal → New Terminal)
```powershell
# primeira vez
git clone https://github.com/Kambulu89/biuri_trepan_reloaded.git
cd biuri_trepan_reloaded
git checkout main

# atualizar uma cópia existente
cd biuri_trepan_reloaded
git status                 # deve estar limpo; guarde trabalho local antes de continuar
git fetch origin
git checkout main
git pull --ff-only origin main
git log -1 --oneline       # confirme o SHA indicado no relatório da fase 5
```

## 4. Ambiente virtual
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
# se a execução de scripts estiver bloqueada:  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python --version           # Python 3.11.x
```
No VS Code: *Ctrl+Shift+P → Python: Select Interpreter →* `.venv\Scripts\python.exe`.

## 5. Instalar
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements-win.txt     # GUI + núcleo + owlready2
python -m pip install -e . --no-deps              # opcional (a CI usa-o); run_biuri.py funciona sem isto
python -m pip install -r requirements-test.txt    # opcional: só se for correr os testes
```

## 6. Verificar o ambiente (problemas de dependências/configuração)
```powershell
python scripts/environment_preflight.py --gui --owl
```
Imprime JSON; `"ready": true` e todos os `"checks"` a `true` significam ambiente completo. Leitura dos campos:
`runtime_dependencies=false` → falta/errada alguma biblioteca do núcleo (reinstale `requirements-win.txt`); `gui_dependencies=false` → PyQt6;
`owl_dependencies=false` → owlready2 ou **Java** não encontrados (instale Java 17 e/ou defina `JAVA_HOME`; reabra o terminal/VS Code);
`graphviz_dot_binary=false` é só um aviso (exportação de imagem). Com `--json caminho.json` guarda o relatório.
Se a GUI não abrir, `python -c "import PyQt6; print('ok')"` isola o problema; mensagens de importação aparecem no terminal.
Sem Java, a ontologia **não** é ativada e os dados carregam em modo sem OWL (TREPAN Reloaded espelha o Original).

## 7. Iniciar a interface gráfica
```powershell
python run_biuri.py
```
(Ponto de entrada: `run_biuri.py` → `gui.biuri_app_complete.main`.) No VS Code pode usar *Run Python File* sobre `run_biuri.py` com o interpretador `.venv`.

## 8. Carregar ARFF e OWL
1. Botão **📁 Carregar dados** → diálogo «Cargar Datos ARFF y Ontología OWL».
2. **Archivo de Datos (ARFF)** (obrigatório) → *Buscar*. A **última coluna é a classe**. Codificação: `utf-8` (ou `latin-1`/`cp1252`).
3. **Archivo de Ontología (OWL)** (opcional) → *Buscar*. Sem OWL, o sistema funciona normalmente (modo compatibilidade).
4. Ficheiros no repositório: `data/dataset.arff` (Adult, ~48 mil linhas — pesado; use para teste de fumo) e ontologias em
   `data/benchmark_ontologies_v7/` (`iris`, `wine`, `breast_cancer`, `digits`, `diabetes_progression`: estado `VALID_DOMAIN_ONTOLOGY` em
   `data/ontology_catalog_v8.json`). **Não use** `data/adult_census_full_ontology.owl` para avaliar: está catalogada como `CONTAMINATED_ONTOLOGY`.
5. Os ARFF dos datasets de demonstração não estão no repositório. Gere-os (nomes de colunas normalizados, classe na última coluna):
```powershell
python -m validation.export_sklearn_arff --name iris --out data\local\iris.arff
python -m validation.export_sklearn_arff --name wine --out data\local\wine.arff
python -m validation.export_sklearn_arff --name breast_cancer --out data\local\breast_cancer.arff
```
   e emparelhe `iris.arff` com `data\benchmark_ontologies_v7\iris.owl`, etc.
6. Se carregar **outra ontologia** (ou removê-la) depois de treinar, os modelos/árvores/métricas/contrafactuais existentes são **descartados** e as
   ações dependentes mostram «Modelos desatualizados»: é preciso **🤖 Treinar modelo** de novo (comportamento introduzido no PR #9).

## 9. Testes (opcional)
```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python -m pytest -q                                   # suite completa (~30–50 min)
python -m pytest tests/test_owl_world_isolation.py tests/test_gui_stale_model_guard.py tests/test_historical_results_registry.py -q
python scripts/check_dataset_agnosticism.py
```

## 10. Roteiro de testes manuais
_Escrito a partir dos rótulos e do comportamento do código; os passos de GUI não foram executados por um humano nesta sessão — diferenças de redação/ordem são esperadas e devem ser reportadas._

Janela principal: barra lateral (**📁 Carregar dados**, **🤖 Treinar modelo**, **💡 Gerar explicação**, **🌳 Visualizar árvore**,
**📊 Comparar métricas**, **🧠 Explicações naturais**, **🔀 Gerar contrafactuais**, **📈 Melhorar árvore substituta**, exportações) e separadores
**📋 Resultados · 🌳 Visualização · 📊 Métricas · 🔍 Auditoria · 🔀 Contrafactuais**. O combo de modo tem
«INTERACTIVE / EXPLORATORY» (por omissão) e «SCIENTIFIC / BENCHMARK».

| # | Cenário | Passos | Resultado esperado |
|---|---|---|---|
| 1 | Sem ontologia (MLP Original, TREPAN Original, C4.5, Reloaded) | Carregar `iris.arff` **sem** OWL → Treinar → Gerar explicação → Visualizar árvore → Comparar métricas | Treino conclui; árvores TREPAN Original e Reloaded e C4.5 aparecem; métricas (accuracy/fidelidade) por modelo; com 0 ontologia o Reloaded espelha o Original |
| 2 | Com ontologia (MLP Ontológico) | Carregar `iris.arff` + `iris.owl` (Java instalado) → Treinar | Resultados mostram carga da ontologia e o critério de aceitação ontológica (aceite/rejeitada, com razão); se aceite, aparece o MLP Ontológico; separador Auditoria preenchido |
| 3 | TREPAN Reloaded vs Original | Após 2, Gerar explicação e Visualizar árvore; escolher cada árvore no seletor | Árvores distintas só se a semântica interveio; auditoria de divisões semânticas disponível. **Não esperar** que o Reloaded vença: é para observar |
| 4 | C4.5 | Em Visualizar/Comparar, escolher C4.5 | Árvore C4.5 (baseline independente) e as suas métricas |
| 5 | Trocar de ontologia | Após treinar, carregar outra OWL (ou dados com OWL diferente) | Aviso/estado «Modelos desatualizados»; Visualizar/Comparar/Exportar/Contrafactuais recusam até treinar de novo; novo treino desbloqueia |
| 6 | Isolamento A→B→A | Carregar `iris.owl`, treinar, anotar nº de features derivadas na Auditoria; carregar `wine.owl`+`wine.arff`, treinar; voltar a `iris.owl`+`iris.arff` | Mesmos resultados da primeira vez (determinístico para a mesma seed) |
| 7 | Contrafactuais | Treinar (cenário 1 ou 2) → separador Contrafactuais → escolher instância/classe → **🔀 Gerar contrafactuais** | Lista de contrafactuais válidos com métricas; construir árvore CF local («Árvore CF local — Oráculo: X») e abri-la em Visualização. Alterar modelo/dataset/ontologia invalida os resultados |
| 8 | Modo científico | Combo de modo → SCIENTIFIC / BENCHMARK → Treinar | Executa o pipeline científico único; as árvores mostradas são as avaliadas (não são reconstruídas) |
| 9 | Exportações | Exportar resultados / árvore | Ficheiros gerados; exportar imagem pode exigir Graphviz (`dot`) — opcional |
| 10 | Sem Java | Renomear temporariamente `JAVA_HOME`/PATH e carregar OWL | Aviso claro; dados carregam sem OWL |

## 11. Problemas conhecidos
Ver o relatório final da fase 5 e `docs/ONTOLOGY_ISOLATION_AUDIT.md` §9 e §11 (resultados históricos `NOT_VERIFIED_OWL_ISOLATION`).
