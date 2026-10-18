# Instalação e teste no Windows — BIURI / TREPAN Reloaded

Esta é uma **versão de software para instalação e testes**, não uma conclusão científica de superioridade. A superioridade da variante com ontologia **não está demonstrada** (ver §12). Todos os comandos abaixo foram retirados do repositório (`run_biuri.py`, `requirements-*.txt`, `pyproject.toml`, `scripts/environment_preflight.py`, `.github/workflows/windows-ci.yml`, `validation/export_sklearn_arff.py`) e confirmados numa instalação Linux; **os passos de interface gráfica ainda não foram executados por uma pessoa num Windows real** (a CI Windows corre os testes automatizados com `QT_QPA_PLATFORM=offscreen`). Complementa `docs/LOCAL_WINDOWS_SETUP.md` e `docs/REASONER_WINDOWS_V9_1.md`.

## 1. Pré-requisitos
| Item | Versão | Notas |
|---|---|---|
| Windows | 10 ou 11, 64-bit | testado automaticamente em `windows-latest` (GitHub Actions) |
| Python 64-bit | **3.11.x recomendado**; 3.12 também testado na CI | `pyproject.toml`: `>=3.11,<3.13` |
| Git | recente | |
| Java 17 (JRE/JDK 64-bit) | Temurin 17 recomendado | **só** para o raciocinador OWL (HermiT); sem Java a aplicação funciona **sem ontologia** |
| Graphviz (`dot`) | opcional | só para exportar imagens da árvore; a ausência nunca impede treino, MLP, ontologia nem TREPAN |

## 2. Java (apenas para ontologias OWL)
Instale Java 17 e abra um **novo** terminal. Alternativas: definir `JAVA_HOME` (pasta do JDK/JRE) ou `OWLREADY2_JAVA_EXE` (caminho completo de `java.exe`). Verificação: `java -version`. A aplicação também procura Java no `PATH` e em instalações comuns (ver `docs/REASONER_WINDOWS_V9_1.md`).

## 3. Clonar a `main`
```powershell
git clone https://github.com/Kambulu89/biuri_trepan_reloaded.git
cd biuri_trepan_reloaded
git checkout main
git log -1 --oneline
```

## 4. Ambiente virtual (PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```
Se o PowerShell bloquear scripts: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` (só para esta sessão).

## 5. Instalar dependências
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements-win.txt      # GUI (PyQt6) + núcleo + owlready2
python -m pip install -r requirements-test.txt     # opcional: só para correr os testes (pytest, pytest-cov, pyflakes)
```
Dependências obrigatórias do núcleo (`requirements-core.txt`): numpy, scipy, pandas, scikit-learn, statsmodels, patsy, matplotlib, seaborn, graphviz, dtreeviz, liac-arff, optuna, joblib, jinja2, sympy. GUI: PyQt6 (6.4.2 em Python 3.11; ≥ 6.6 e < 6.9 em 3.12). OWL: owlready2 0.47. **Opcionais:** TensorFlow/CLEAR (`requirements-clear.txt`), Excel/Parquet (`requirements-excel-parquet.txt`).

## 6. Verificar o ambiente
```powershell
python scripts/environment_preflight.py --gui --owl
```
Imprime JSON; `"ready": true` com todos os `"checks"` a `true` indica ambiente completo (`java=false` → instale Java; `owl_dependencies=false` → owlready2 ou Java; `gui_dependencies=false` → PyQt6; `graphviz_dot_binary=false` é só um aviso).

## 7. Iniciar a aplicação
```powershell
python run_biuri.py
```
Ponto de entrada real: `run_biuri.py` → `gui.biuri_app_complete.main` (abre a janela PyQt6). A aplicação **não exige** nenhum documento nem ontologia de German Credit para iniciar, e **nenhum mecanismo experimental H2/B3.x está ligado à interface** (vivem em `validation/`, sem importações a partir de `core/`, `gui/` ou `run_biuri.py`).

## 8. Carregar ARFF e OWL
1. **📁 Carregar dados** → diálogo «Cargar Datos ARFF y Ontología OWL».
2. **ARFF** (obrigatório): extensão **`.arff` em minúsculas**; **a última coluna é a classe**.
3. **OWL** (opcional): sem OWL o sistema funciona normalmente (modo sem ontologia: TREPAN Reloaded ≡ Original).
4. Se escolher OWL e a ontologia não puder ser ativada (Java ausente, reasoner falhou, ontologia inconsistente, ABox rejeitada), a interface mostra «ONTOLOGÍA NO ACTIVADA» com a razão e **continua sem OWL**.
5. Ontologias de demonstração no repositório: `data\benchmark_ontologies_v7\{iris,wine,breast_cancer,digits,diabetes_progression}.owl`. **Não use** `data\adult_census_full_ontology.owl` para avaliar (catalogada como contaminada).
6. Os ARFF de demonstração não estão no repositório; gere-os:
```powershell
python -m validation.export_sklearn_arff --name iris --out data\local\iris.arff
python -m validation.export_sklearn_arff --name wine --out data\local\wine.arff
python -m validation.export_sklearn_arff --name breast_cancer --out data\local\breast_cancer.arff
```
   (cria as pastas em falta; a classe fica na última coluna). Emparelhe `iris.arff` com `data\benchmark_ontologies_v7\iris.owl`, etc.
7. Mudar/remover a ontologia depois de treinar **invalida** os modelos, árvores, métricas e contrafactuais («Modelos desatualizados»): treine de novo.

## 9. Treinar, construir e visualizar árvores
Barra lateral: **🤖 Treinar modelo** (MLP + TREPAN Original + TREPAN Reloaded + C4.5, com uma barra de progresso cancelável), **💡 Gerar explicação**, **🌳 Visualizar árvore**, **📊 Comparar métricas**, **🧠 Explicações naturais**, **🔀 Gerar contrafactuais**, **📈 Melhorar árvore substituta**, exportações. Separadores: Resultados, Visualização, Métricas, Auditoria, Contrafactuais. O modo por omissão é «INTERACTIVE / EXPLORATORY»; «SCIENTIFIC / BENCHMARK» executa o serviço científico único. Na Visualização: nós elípticos com as condições (incluindo m-of-n), setas rotuladas, zoom (🔍- / 🔍+), 🔄 Ajustar, ↺ Reset, seleção de nós e seletor de árvore (Original, Reloaded, C4.5). O treino com o Optuna pode demorar **vários minutos** mesmo em datasets pequenos.

## 10. Contrafactuais
Separador **Contrafactuais**: escolher instância e classe-alvo → **✦ Gerar contrafactual** (mudanças propostas e métricas: validade, proximidade…), **▦ Regras globais**, **🌳 Construir árvore CF**, **⇄ Avaliar transferência**, **◉ Visualizar árvore CF**, **⇩ Exportar**. Se não houver contrafactuais válidos a interface indica-o; alterar modelo, dataset ou ontologia invalida os resultados.

## 11. Testes
```powershell
$env:QT_QPA_PLATFORM = "offscreen"
$env:PYTHONUTF8 = "1"
python -m pytest -q                                   # suite completa; na CI Windows demora ≈ 30–60 min
python scripts/check_dataset_agnosticism.py           # guard estático de agnosticismo ao dataset (rápido)
python -m pytest -q tests/test_eq_reloaded_original.py tests/test_gui_stale_model_guard.py tests/test_ontology_matching_p3.py   # amostra rápida
```
Para o conjunto H2 (isolado em `validation/`): `python -m pytest -q tests/test_h2_pool_study.py tests/test_h2_metamorphic.py tests/test_h2_pilot_synthetic.py`.

## 12. Roteiro de testes manuais (a executar no seu computador)
_Escrito a partir dos rótulos e do comportamento do código. Registe qualquer diferença como defeito ou limitação; não foi executado por uma pessoa num Windows real._

**Fluxo A — sem ontologia**
- [ ] Abrir a aplicação (`python run_biuri.py`). [ ] Carregar `iris.arff` sem OWL; confirmar atributos e classes. [ ] **Treinar modelo** (MLP, TREPAN Original, TREPAN Reloaded sem OWL, C4.5).
- [ ] Confirmar que Original e Reloaded sem OWL têm as **mesmas regras** (`Reloaded(None) ≡ Original`). [ ] Visualizar as três árvores. [ ] Consultar accuracy, fidelidade e complexidade em **Comparar métricas**.

**Fluxo B — com ontologia**
- [ ] Carregar `iris.arff` + `data\benchmark_ontologies_v7\iris.owl` (com Java). [ ] Ler o relatório de carga (SHA-256, classes, propriedades) e o separador **Auditoria** (matching, quality gate, reasoner). [ ] Treinar.
- [ ] Confirmar que o relatório distingue ontologia **aceite/ativa**, **rejeitada** e **sem estrutura semântica efetiva**. [ ] Visualizar regras e árvore Reloaded; consultar métricas.
- [ ] **A utilização da ontologia não é prova de ganho:** compare sempre com o Original nas mesmas instâncias.

**Fluxo C — visualização**
- [ ] Árvores legíveis, nós elípticos com texto completo, ligações e rótulos corretos. [ ] Condições m-of-n legíveis. [ ] Zoom, ajuste, reset e deslocação com o rato. [ ] Seleção de nó mostra os detalhes. [ ] Coerência entre árvore e regras (Resultados / Explicações naturais).

**Fluxo D — contrafactuais**
- [ ] Gerar contrafactual para uma instância e classe-alvo; ler as mudanças. [ ] Construir e visualizar a árvore CF. [ ] Escolher uma classe-alvo/instância sem solução e confirmar a mensagem. [ ] Mudar a ontologia e confirmar a invalidação.

**Fluxo E — erros e robustez**
- [ ] ARFF inválido (ver §13: mensagem pouco informativa). [ ] OWL incompatível ou ontologia inconsistente → aviso e continuação sem OWL. [ ] Matching ambíguo → atributos sem entidade (ver Auditoria). [ ] **Sem Java** → «Reasoner OWL no disponible», dados carregam sem OWL. [ ] Dependência em falta → `run_biuri.py` mostra a lista e o comando de instalação. [ ] Categorias desconhecidas em dados novos. [ ] Cancelar o treino. [ ] Caminhos com espaços e acentos (testados em Linux para ARFF e OWL; **por testar em Windows**).

## 13. Erros frequentes
| Sintoma | Causa provável | Resolução |
|---|---|---|
| `ImportError` ao iniciar | dependências em falta | `python -m pip install -r requirements-win.txt` no `.venv` ativo |
| «ONTOLOGÍA NO ACTIVADA / Reasoner OWL no disponible» | Java ausente ou fora do `PATH` | instalar Java 17; definir `JAVA_HOME` ou `OWLREADY2_JAVA_EXE`; reabrir o terminal |
| `owlready2 no utilizable` | pacote corrompido ou pasta homónima `owlready2` | reinstalar `owlready2==0.47`; verificar com o preflight |
| Janela não abre / falha ao importar PyQt6 | versão de PyQt6 incompatível com o Python | usar Python 3.11 (ou 3.12 com PyQt6 ≥ 6.6) |
| «Error al cargar archivos:» **sem texto** | ARFF inválido (a exceção do leitor não traz mensagem) | **limitação conhecida**: confirmar cabeçalho `@RELATION/@ATTRIBUTE/@DATA` e extensão `.arff` |
| Exportar imagem da árvore falha | Graphviz (`dot`) ausente | instalar Graphviz e pô-lo no `PATH` (opcional) |
| Treino lento | tuning Optuna | esperar; usar dataset pequeno para teste de fumo |
| `Modelos desatualizados` | ontologia/dados mudaram | **🤖 Treinar modelo** de novo |

## 14. Atualizar a instalação
```powershell
git status                       # deve estar limpo
git fetch origin
git checkout main
git pull --ff-only origin main
python -m pip install -r requirements-win.txt
python scripts/environment_preflight.py --gui --owl
```

## 15. Limitações conhecidas e estado científico
* **Sem ganho ontológico demonstrado:** B3.2 e B3.3 — `NO_ONTOLOGICAL_ADVANTAGE` nos regimes avaliados; B1 e B2-C experimentais, não promovidas; H1/B3.4 só desenho (campanha não executada); H2 — preparação técnica concluída, avaliação científica pendente; M0–M6 sem integração produtiva. `Acc(Reloaded_OWL) > Acc(Original) > Acc(C4.5)` e `Fid(Reloaded_OWL) > Fid(Original)`, `> Fid(C4.5)` são **objetivos**, não resultados.
* German Credit: fontes oficiais (`german.doc`, South German Credit, Grömping 2019) **não consultadas** e revisão humana **pendente**; as ontologias/fichas de rascunho em `validation/` são hipóteses de IA e **não** são usadas pela aplicação.
* ARFF: só extensão `.arff` minúscula; mensagem de erro vazia para ARFF inválido; texto da interface misto (espanhol/português).
* Treino interativo pode demorar vários minutos (tuning).
* Testes de interface em Windows real e em caminhos Windows com espaços/Unicode **dependem do seu computador**.
