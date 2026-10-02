# Auditoria e implementação de contrafactuais

Data da revisão: 24 de agosto de 2026

## Conclusão executiva

O código recebido **não continha toda a funcionalidade descrita em
`contrafactuais_artigo.docx`**. Existia um fluxo legado útil — geração CLEAR e
COGS, indicadores A/B/C e melhoria posterior das árvores — mas faltavam a API
interactiva completa, DiCE/LORE, o protocolo formal P1-P8, restrições gerais,
robustez, validação ontológica conservadora, filtro RST, exportação unificada e
uma aba dedicada na GUI.

Essas lacunas foram implementadas como uma camada pós-treino isolada. O treino
da MLP, a extracção Trepan/C4.5, a aceitação ontológica, as métricas preditivas e
o fluxo legado de melhoria não foram substituídos. A acção avançada antiga
continua a produzir o formato A/B/C esperado por `improve_surrogates`.

Na aplicação, toda a análise contrafactual é limitada ao dataset ARFF carregado
e treinado na sessão actual. Os seis datasets de referência não são uma fonte
de dados da interface: existem exclusivamente para a validação experimental P9,
num utilitário alojado sob `tests/support/`.

Esta separação também é coerente com
`Counterfactual_Enhanced_Surrogate_Models.pdf`: os datasets ali apresentados
pertencem ao protocolo de avaliação/benchmark, enquanto a geração de uma
explicação contrafactual opera sobre uma instância e o respectivo modelo do
dataset em análise. Por isso, a interface não expõe um selector multi-dataset.

## Correcção de âmbito aplicada

- A sessão contrafactual é construída directamente a partir das matrizes,
  modelos, metadados e ontologia do ARFF que o utilizador carregou.
- A aba mostra o nome do dataset activo e declara que apenas ele será analisado.
- O botão de transferência só é activado quando a MLP e as duas árvores
  necessárias estão disponíveis para esse dataset.
- `evaluate_transfer_from_session` rejeita chaves de âmbito externo ou múltiplo
  (`dataset`, `dataset_name`, `datasets`, `sessions`).
- O resultado operacional identifica `protocol=P1-P8` e
  `scope=loaded_dataset_only`; não contém resumo global nem coleção de datasets.
- O código P9 foi removido dos módulos operacionais e colocado sob
  `tests/support/`, com um runner cujo nome e saída indicam explicitamente que se
  trata de teste.

## Melhoria de visibilidade da interface

A aba de contrafactuais foi reorganizada para evitar os campos sobrepostos ou
cortados observados em janelas com altura limitada:

- o formulário passou de nove linhas para uma grelha responsiva de quatro;
- a área superior preserva a sua altura mínima em vez de ser comprimida pelas
  tabelas;
- os botões usam títulos curtos e tooltips que conservam a descrição completa;
- explicação, métricas e candidatos estão em splitters horizontais e verticais,
  permitindo ao utilizador ajustar manualmente as proporções;
- as tabelas ocultam a numeração redundante, redimensionam colunas pelo conteúdo
  e reservam a coluna de alterações para expansão;
- cada célula mantém o valor integral num tooltip, mesmo quando a janela é
  estreita;
- os cabeçalhos mudam conforme o resultado: `Classe` na geração e `Categoria` /
  `Robustez conjunta` na avaliação de transferência;
- o botão de transferência não é reactivado indevidamente após uma operação se
  os três modelos necessários não estiverem disponíveis.

## Matriz artigo × implementação

| Requisito do artigo | Estado inicial | Estado após revisão | Implementação principal |
|---|---|---|---|
| Camada opcional pós-treino | Parcial | Implementado | `counterfactuals/service.py` |
| MLP Original e MLP Ontológica como oráculos | Parcial | Implementado com portão de aceitação | `service._resolve_interactive_context`, `transfer._space` |
| DiCE | Ausente | Implementado; biblioteca externa opcional e fallback interno | `engine._generate_dice` |
| CLEAR local | Legado, não unificado | Implementado no motor e legado preservado | `engine._generate_clear` |
| CoGS multiobjectivo | Legado, não unificado | Implementado no motor e legado preservado | `engine._generate_cogs` |
| LORE-Local | Ausente | Implementado | `engine._generate_lore_local` |
| LORE-Global | Ausente | Implementado com árvore global e cobertura | `engine._tree_counterfactuals` |
| Imutabilidade, acção, protecção e domínios | Ausente | Implementado e configurável por metadados | `CounterfactualConstraints` |
| Categóricas/binárias | Parcial | Implementado por valores admissíveis e projecção discreta | `CounterfactualConstraints.project` |
| Regras causais explícitas | Ausente | Implementado | `CounterfactualConstraints._validate_causal_rules` |
| Validação OWL opcional | Ausente no fluxo CF | Implementado de forma conservadora | `OntologyCFValidator` |
| Filtro RST opcional | Ausente | Implementado com frequência de alterações × dependência aproximada | `apply_rst_filter` |
| Validade, proximidade e esparsidade | Parcial | Implementado | `engine._evaluate_candidate` |
| Diversidade e estabilidade | Ausente | Implementado | `engine._aggregate_metrics` |
| Cobertura/fidelidade local e global | Ausente | Implementado quando aplicável | metadados LORE e `global_fidelity` |
| Acionabilidade, plausibilidade e consistências | Ausente | Implementado; N/A quando não há regra/mapeamento | motor e validador |
| Robustez por perturbações | Ausente | Implementado com semente e epsilon configuráveis | `evaluate_local_robustness` |
| Protocolo P1-P8 | Ausente | Implementado | `evaluate_transfer_protocol` |
| P9 em pelo menos seis datasets | Ausente | Implementado apenas como teste isolado | `tests/support/counterfactual_multidataset_runner.py`, `run_cf_transfer_tests.py` |
| Cinco categorias formais | Ausente | Implementado com os identificadores exactos | `TransferCategory` |
| Agregação no dataset carregado e por método | Parcial (A/B/C) | Implementado | `summary`, `summary_by_method` |
| Compatibilidade de espaços original/enriquecido | Frágil | Implementado com validação e falha controlada | `FeatureSpaceAdapter` |
| Aba dedicada na GUI | Ausente | Implementado | `gui/counterfactual_panel.py` |
| JSON, CSV e Markdown | Ausente | Implementado | `counterfactuals/export.py` |
| JSON e CSV experimentais P9 | Ausente | Implementado apenas no runner de testes | `tests.support.counterfactual_multidataset_runner` |

## Semântica dos modelos e dos oráculos

Na geração interactiva, a escolha Trepan/C4.5 determina a família de métodos e,
no LORE-Global, a árvore cujas regras globais são examinadas. A validade do
contrafactual continua a ser definida pela MLP de referência, como exige o
artigo. Para Trepan Reloaded, a MLP Ontológica só é seleccionada quando a
ontologia foi aceite e existem modelo, matriz e nomes do espaço enriquecido.

Na avaliação de transferência, o contrafactual é sempre gerado para a MLP. O
mesmo conjunto de valores comuns é avaliado nas duas árvores. Uma projecção do
espaço enriquecido para o original é segura por nomes; a operação inversa exige
uma função que recalcule as features derivadas. Sem essa função, o sistema lança
`FeatureSpaceMismatchError`, em vez de preencher ou truncar silenciosamente.

## Robustez não é consistência

Foram mantidos conceitos separados:

- **consistência/transferência**: acordo entre MLP, Trepan Original e Reloaded
  sobre o factual e o contrafactual;
- **robustez local**: proporção de pequenas perturbações do contrafactual que
  preservam a classe desejada no mesmo modelo;
- **robustez conjunta**: proporção das mesmas perturbações em que os três modelos
  preservam simultaneamente a nova classe.

Assim, um caso pode transferir fortemente no ponto contrafactual e ainda ser
localmente frágil. Os resultados não misturam estes dois argumentos.

## Fluxo operacional P1-P8 no dataset carregado

1. Carrega ou recebe uma sessão de dataset e modelos.
2. Escolhe a MLP original ou ontológica aceite.
3. Valida modelos, schemas e nomes de features.
4. Faz amostragem estratificada, por padrão 33%, com semente 42.
5. Aplica o portão de acordo inicial MLP/Trepan/Reloaded.
6. Gera na MLP com LORE-Local, CLEAR ou CoGS.
7. Aplica o vector compatível aos três modelos.
8. Classifica cada par com uma das cinco categorias formais e calcula robustez.
O serviço devolve explicitamente `scope: loaded_dataset_only` e o nome do
dataset da sessão. Opções que tentem fornecer `dataset`, `datasets` ou sessões
externas são rejeitadas, prevenindo a inclusão acidental dos datasets de teste.

## Validação experimental P9, exclusiva para testes

A repetição em pelo menos seis datasets e a agregação global foram mantidas
fora do código operacional. Este teste pode ser executado voluntariamente com:

```bash
python run_cf_transfer_tests.py
```

Os artefactos são gravados, por padrão, em
`counterfactuals/resultados_consolidados/testes_p9_multidataset/`.

## Testes executados

| Verificação | Resultado |
|---|---|
| Compilação de todos os ficheiros Python (`compileall`) | Passou |
| Testes unitários independentes de GUI | 9/9 passaram |
| Cinco geradores num problema controlado | Todos encontraram CF válido |
| Restrições, OWL conservador, RST e robustez | Passaram |
| Exportação JSON/CSV/Markdown | Passou |
| Falha controlada original → enriquecido sem derivação | Passou |
| Serviço rejeita entrada multi-dataset | Passou |
| P1-P8 chamado pelo serviço com apenas o Iris carregado | Passou (`scope=loaded_dataset_only`) |
| P9 isolado com Iris, Wine, German Credit, WDBC, Sonar e Hepatitis | Passou; fora da aplicação |
| Amostragem de 33% no Iris, três métodos, 108 pares | Passou |
| Compilação da integração PyQt6 e testes GUI adicionados | Passou na análise estática |
| Testes headless do novo layout responsivo | Adicionados; requerem PyQt6 no ambiente Windows |

No teste de 33% do Iris foram seleccionadas 36 instâncias, produzindo 108 pares
instância×método. O protocolo terminou sem excepções; 87,0% dos pares elegíveis
obtiveram contrafactual válido na MLP. Estas taxas são apenas um teste dos
artefactos fornecidos, não uma alegação científica final.

O ambiente de validação não inclui o binário PyQt6, por isso não foi possível
abrir fisicamente a janela. A GUI foi compilada, as ligações de sinais foram
revistas e foram adicionados testes headless que serão executados quando PyQt6
e pytest estiverem disponíveis no ambiente Windows do projecto.

## Limitações explícitas

- A consistência causal só pode ser avaliada quando o dataset fornece regras
  causais; sem regras, a métrica representa apenas ausência de violação
  conhecida.
- A validação OWL verifica mapeamentos, ranges de datatype e regras semânticas
  fornecidas. Não inventa axiomas nem executa automaticamente um reasoner Java.
- O filtro RST usa uma aproximação de região positiva por atributo, combinada
  com a frequência das alterações. Não afirma calcular todos os redutos de
  Pawlak.
- `dice-ml` permanece opcional. O fallback garante funcionamento sem aumentar
  as dependências obrigatórias, mas não é numericamente idêntico ao DiCE.
- Modelos persistidos com scikit-learn são sensíveis à versão. O runner corrige
  a ausência do atributo `monotonic_cst` em árvores 1.3 carregadas em versões
  novas e recupera nomes posicionais se metadados pandas antigos não abrirem.

## Referências científicas usadas na revisão

- Guidotti et al., LORE: https://arxiv.org/abs/1805.10820
- White e Garcez, CLEAR: https://arxiv.org/abs/1908.03020
- Mothilal, Sharma e Tan, DiCE: https://arxiv.org/abs/1905.07697
- Maragno et al., contrafactuais esparsos e robustos:
  https://arxiv.org/abs/2201.09051
- Pawelczyk et al., robustez com garantias probabilísticas:
  https://arxiv.org/abs/2305.11997
