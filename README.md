# analise_de_sentimentos_bancos_digitais
Código utilizado para coletar e analisar dados do Twitter sobre bancos digitais, incluindo uma comparação com os bancos tradicionais. Utilização do algoritmo Naive Bayes para classificação dos tweets como positivos, negativos e neutros. <br><br>
A construção do código e todas as conclusões estão explicadas [nesse artigo](https://medium.com/@marianamannes/analisando-sentimentos-de-tweets-sobre-bancos-digitais-dac1e5d1ff01).

## Pré-requisitos
- Python 3
- Bibliotecas em <b>requirements.txt</b>
- Adicionar a base de treino e teste em /csvfiles. Em meu modelo, foi utilizado o dataset [Portuguese Tweets for Sentiment Analysis](https://www.kaggle.com/augustop/portuguese-tweets-for-sentiment-analysis), disponível no Kaggle.

## getoldtweets3.py
Utilizado para a coleta de dados do twitter, com a biblioteca GetOldTweets3.

## timeseries.py
Análise da periodicidade dos tweets com conceitos de séries temporais.

## modelo.py
Criação efetiva do modelo de Machine Learning e aplicação na base de dados.

## csvfiles
Pasta com os arquivos de treino e teste para o modelo.
