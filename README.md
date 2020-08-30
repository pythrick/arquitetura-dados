# Trabalho da disciplina de Arquitetura de Dados

Base de dados escolhida: [Weight Lifting Exercises monitored with Inertial Measurement Units Data Set](http://archive.ics.uci.edu/ml/datasets/Weight+Lifting+Exercises+monitored+with+Inertial+Measurement+Units)


## TODO
1. Escolher 3 algorimos dos tipos:
    * Regressão
    * Árvore de decisão
    * Rede Neural
    - Fazer o uso de cross-validation na base completa (treino e teste), para ter uma ideia inicial de quanto ele pode chegar,
    o máximo que ele poderia aprender com aqueles dados.
2. Executar as predições para cada modelo e salvar os resultados (estado inicial)
    - O estado mais próximo do dado de origem
    - Realizar apenas a mínima adequação da base para rodar os modelos
    - Não mudar escala
    - Não aplicar média
    - Não fezer associação entre produtos
    - Não normalizar
    - Conseguiu relacionar padrão com classe? Chegou no estado inicial.
3. Aplicar técnicas/estratégias de tratamento de dados na base:
    - Seleção
        - Forward Selection / Backward Selection
    - Limpeza
        - Entropia
        - Normalização (Min/Max)
        - Eliminar Outliers
        - Boxplot
    - Codificação (para casos de texto)
    - Enriquecimento
4. Executar as predições para cada modelo e comparar com os resultados anteriores
5. Desenvolver artigo com base no template [RITA](https://www.overleaf.com/latex/templates/revista-de-informatica-teorica-e-aplicada-rita/fhxyfwnmhxzj)
    - Fará vídeo dicas de escrita de artigos
    - Relatório em formato de artigo
    - Mostrar os resultado dos experimentos em formato de tabelas e gráficos
    - Metodologia (ponto de partida):
        - Descrição da base de dados que estamos utilizando
        - Exemplo: "Obtivemos a base tal, de tal lugar, esse conjunto de dados apresenta tantos padrões,
        tantos registros, com os seguintes atributos, esses atributos possuem valores contínuos 
        ou possuem valores categorizados com tais rótulos e tais termos."
        - Algo breve, sucinto que descreve o problema que estamos trabalhando e o que propomos classificar.
    - Resultados em formato de tabelas e gráficos
        - Tabelas usando as métricas geradas a partir da matrix de confusão:
            - Acurácia, Precisão, Recall, F1 Score, Especialidade, Sensibilidade
            - Uma linha para o estado inicial
            - Aplicação da técnica de normalização, obtivemos o seguinte resultado
            - Uma linha para cada tipo de tratamento
            - Tendo uma linha com estado inicial e uma linha com estado final que apresente um ganho, 
            um acerto ou uma melhora, nós já atingimos o objetivo da disciplina
            - Se conseguirmos fazer múltiplas linhas nesta tabela, melhor, por que iremos demonstrar 
            quais ténicas foram efetivas para melhorar o problema
    - Sessões:
        - Título
        - Resumo
        - Introdução
        - Metodologia
        - Resultado
        - Discussão (ou Resultado e Discussão juntos)
        - Conclusão
    - Mais importante as sessões do que a formatação gráfica
    - Dica: Se utilizar Word, manter imagens, gráficos em arquivos separados
    - Entregar em formato PDF
    - Não quer muitas páginas, o mais sucinto/objetivo possível
    - De modo que seja fácil para os demais colegas avaliar / reproduzir
