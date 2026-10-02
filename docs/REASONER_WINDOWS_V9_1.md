# Reasoner OWL no Windows

O enriquecimento OWL utiliza HermiT/Pellet através do Owlready2. Estes
reasoners são distribuídos com o pacote Python, mas necessitam de um processo
Java externo.

## Configuração recomendada

1. Instale um JRE/JDK de 64 bits (Java 17 recomendado).
2. Defina `JAVA_HOME` para a pasta do JDK/JRE, por exemplo:
   `C:\Program Files\Eclipse Adoptium\jdk-17...`.
3. Alternativamente, defina `OWLREADY2_JAVA_EXE` diretamente para
   `...\bin\java.exe`.
4. Reinicie o BIURI depois de alterar variáveis de ambiente.

O BIURI também procura automaticamente Java no `PATH` e em instalações comuns
da Oracle/OpenJDK, Eclipse Adoptium, Microsoft, Amazon Corretto e BellSoft.

## Comportamento seguro

- Java encontrado e reasoner concluído: a ontologia segue para ABox audit e
  quality gate.
- Java ausente ou reasoner falhou: a ontologia não é ativada, não é assumida
  como consistente e não gera features.
- Se ARFF e OWL foram escolhidos juntos, o ARFF continua a carregar em modo
  sem OWL após o aviso. Nesse modo, TREPAN Reloaded espelha TREPAN Original.
- Uma inconsistência lógica real continua a ser um erro bloqueante da OWL.

Não existe fallback estrutural apresentado como reasoning OWL DL: isso
preserva a validade científica do enriquecimento.
