# Protocolo obrigatório de engenharia, execução e validação (permanente)

**Regra absoluta:** é proibido declarar uma funcionalidade como verificada sem ter executado o processo real que comprova o seu funcionamento. Tem prioridade sobre rapidez, economia de tokens e conveniência.

## 1. Níveis de evidência (não são equivalentes)
1. Código escrito · 2. compilado/análise estática · 3. testes unitários executados · 4. testes de integração executados · 5. processo funcional executado de ponta a ponta · 6. resultado científico validado experimentalmente.
Carregar um dataset ≠ validar o treino · carregar uma ontologia ≠ consistência · instanciar o MLP ≠ treinar · um teste unitário ≠ validar a aplicação · dados sintéticos ≠ dados reais. Nunca substituir evidência experimental por leitura do código.

## 2. Afirmações proibidas sem execução real
«Testado com sucesso», «funcionamento verificado», «problema resolvido», «treino validado», «implementação concluída e funcional», «integração confirmada», «pipeline validado», «sem erros».

## 3. Execução obrigatória por etapas
1. **Compreensão** (comportamento esperado, código, dependências, configuração, dados, critérios de aceitação).
2. **Implementação** (completa, respeitando a arquitetura; sem alterações desnecessárias nem remoção de funcionalidades sem justificação).
3. **Execução** da operação real correspondente (CSV → carregamento real; pré-processamento → transformação real; MLP → treino completo; TREPAN → extração real; ontologia → pipeline com ontologia real; OWL → reasoner; API → pedidos reais; interface → fluxo funcional com backend).
4. **Evidências**: comando, ambiente, ficheiros, parâmetros, data, resultado, código de saída, logs, exceções/warnings, duração. Nunca inventar logs ou métricas; o que não foi executado declara-se «não validado».

## 4. Machine Learning (ciclo completo)
Dataset → pré-processamento → treino MLP → predição → avaliação → extração TREPAN → avaliação de regras → integração ontológica → avaliação semântica.
- **Dataset:** existência, formato, coluna-alvo (nunca assumir que é a última quando há mapeamento ontológico), ordem das classes, tipos, ausentes, nº de instâncias/atributos, distribuição das classes, compatibilidade com modelo e metadados ontológicos.
- **Treino MLP:** dataset real → pré-processamento → divisão → ajuste só no treino → treino efetivo → termina? → predições → métricas (accuracy, precision, recall, F1, matriz de confusão, iterações/convergência, tempo, seed) → correspondência índices↔classes. Nunca usar o teste para escolher hiperparâmetros.
- **TREPAN:** MLP efetivamente treinado; extração real; fidelidade ao MLP **separada** da accuracy face aos rótulos reais; estrutura das regras; correspondência das classes; falhas; parâmetros.
- **Ontologia:** ficheiro real; sintaxe; namespaces; mapeamentos ARFF/CSV; alvo; tipos; ordem das classes; reasoner (HermiT) — sem reasoner: «Validação lógica não executada»; classes insatisfazíveis; axiomas; cobertura semântica. Carregar OWL ≠ consistência; consistência ≠ qualidade científica.

## 5. Reprodução obrigatória de erros
Analisar o erro → identificar o cenário → **reproduzir** → guardar o traceback → causa-raiz → corrigir → **repetir a operação original** → confirmar que o erro desapareceu → regressão → evidências. Se não for possível reproduzir, documentar a limitação (nunca inventar reprodução).

## 6. Matriz de validação (fim de cada tarefa)
| Funcionalidade | Implementada | Executada | Validada | Evidência |
Estados: **IMPLEMENTADO** (código sem execução comprovada) · **EXECUTADO** (correu, sem implicar sucesso) · **VALIDADO** (critérios comprovados por execução e resultados) · **FALHOU** · **BLOQUEADO** (impedimento documentado) · **NÃO TESTADO**.

## 7. Mocks e dados sintéticos
Servem para testes unitários/desenvolvimento; nunca como evidência com dados reais, nem substituir ficheiros reais em silêncio, nem simular APIs para afirmar integração real, nem apresentar métricas sintéticas como resultados experimentais. Distinguir sempre testes simulados de reais.

## 8. Critérios de conclusão e fórmulas
Concluída só com: implementação completa, critérios avaliados, testes relevantes executados, fluxo real exercitado quando aplicável, sem falhas críticas conhecidas no âmbito validado, evidências documentadas. Caso contrário:
- «Implementação concluída. Validação funcional pendente.»
- «Implementação realizada. Validação falhou.»
- «Implementação realizada. Execução não efetuada devido a [impedimento concreto].»

## 9. Relatório final obrigatório
A. Resumo da implementação · B. Ficheiros modificados · C. Comandos executados · D. Resultados observados (e logs) · E. Matriz de validação · F. Problemas encontrados · G. Estado final (VALIDADA / PARCIALMENTE VALIDADA / FALHOU / BLOQUEADA, delimitando o âmbito comprovado) · H. Próximos passos (só o que falta).

## 10. Regra final — evidência antes da afirmação
Antes de «verificado/validado/testado/corrigido/funcional»: executei a operação? tenho evidências? os resultados satisfazem os critérios? testei com os ficheiros e condições relevantes? a afirmação corresponde exatamente ao comprovado? Se alguma resposta for negativa, usar uma descrição mais restrita e verdadeira.
Nunca confundir implementação com validação, carregamento com treino, execução com sucesso, consistência ontológica com qualidade científica; nunca inventar evidências.
