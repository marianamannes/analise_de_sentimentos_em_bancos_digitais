import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from matplotlib.colors import ListedColormap

nubank = pd.read_csv(r'csvfiles\nubank.csv')
inter = pd.read_csv(r'csvfiles\inter.csv')
original = pd.read_csv(r'csvfiles\original.csv')

base = pd.read_csv(r"csvfiles\Train3classes.csv", sep=";")
teste = pd.read_csv(r"csvfiles\Test3classes.csv", sep=";")
base = base.drop_duplicates()
teste = teste.drop_duplicates()

nubank = nubank.drop_duplicates()
nubank = nubank.iloc[0:1899, 1]
inter = inter.drop_duplicates()
inter = inter.iloc[0:1899, 1]
original = original.drop_duplicates()
original = original.iloc[0:1899, 1]

stops = set(nltk.corpus.stopwords.words('portuguese'))
stops_nuvem = list(stops) + ["eu", "voce", "voces", "nos", "eles", "ele", "ela",
                             "sim", "nao", "banco", "nubank", "inter",
                             "original", "neon", "ta", "to", "pra",
                             "pro", "ma", "me", "ja", "porque", "por que",
                             "do", "da", "so", "ai", "ate", "é", "sao"]


def limpar_texto(texto):
    texto = str(texto).lower()
    texto = texto.replace("pq", "porque")
    texto = texto.replace("vdd", "verdade")
    texto = texto.replace("vc", "você")
    texto = texto.replace("vcs", "vocês")
    texto = texto.replace("tb", "também")
    texto = texto.replace("tbm", "também")
    texto = texto.replace("add", "adicionar")
    texto = texto.replace("vlw", "valeu")
    texto = texto.replace("abs", "abraço")
    texto = texto.replace("qdo", "quando")
    texto = texto.replace("qd", "quando")
    texto = texto.replace("msg", "mensagem")
    texto = texto.replace("blz", "beleza")
    texto = texto.replace("dps", "depois")
    texto = texto.replace("dpois", "depois")
    texto = texto.replace("qto", "quanto")
    texto = texto.replace("qt", "quanto")
    texto = texto.replace("qts", "quantos")
    texto = texto.replace("td", "tudo")
    texto = texto.replace("tds", "todos")
    texto = texto.replace("naum", "não")
    texto = texto.replace("obs", "observação")
    texto = texto.replace("niver", "aniversário")
    texto = texto.replace("bjo", "beijo")
    texto = texto.replace("bj", "beijo")
    texto = texto.replace("pc", "computador")
    texto = texto.replace("cmg", "comigo")
    texto = texto.replace("mto", "muito")
    texto = texto.replace("agr", "agora")
    texto = texto.replace("algm", "alguém")
    texto = texto.replace("ctg", "contigo")
    texto = texto.replace("ctz", "certeza")
    texto = texto.replace("dsd", "desde")
    texto = texto.replace("enqto", "enquanto")
    texto = texto.replace("img", "imagem")
    texto = texto.replace("hr", "hora")
    texto = texto.replace("hrs", "horas")
    texto = texto.replace("mt", "muito")
    texto = texto.replace("eh", "é")
    texto = texto.replace(":)", "positivo")
    texto = texto.replace("(:", "positivo")
    texto = texto.replace("):", "negativo")
    texto = texto.replace(":(", "negativo")
    texto = re.sub(r'[😌🤨😒😞😔😟😕🙁☹️😣😖😫😩🥺😢😭😤😠😡🤬🤯😳🥵🥶😱😨😰😥😓🤔🤥😶😐😑😬🙄😯😦😧😮😲🥱😴🤤😪😵🤐🥴🤢🤮🤧😷🤒🤕😈👺🤡💩👻💀☠️👽👾]', "negativo", texto)
    texto = re.sub(r'[❤️😀😃😄😁😆😅😂🤣☺️😊🙂😉😍🥰😘😗😙😚😋😛😝😜🤪🤓😎🤩🥳😏🤗🤑🤠]', "positivo", texto)
    texto = re.sub(r'[áâàã]', "a", texto)
    texto = re.sub(r'[éêè]', "e", texto)
    texto = re.sub(r'[íîì]', "i", texto)
    texto = re.sub(r'[óôòõ]', "o", texto)
    texto = re.sub(r'[úûù]', "u", texto)
    texto = re.sub(r'http\S+', "", texto)
    texto = re.sub(r'@\S+ ?', "", texto)
    texto = re.sub(r'\n', "", texto)
    texto = re.sub(r'[-./?!,";\']', "", texto)
    return texto


def stemming(texto):
    stemmer = nltk.stem.RSLPStemmer()
    palavras = []
    for w in texto.split():
        if w not in stops:
            palavras.append(stemmer.stem(w))
    return (" ".join(palavras))


def preprocessar(tweets):
    tweets = [limpar_texto(str(tweet)) for tweet in tweets]
    tweets = [stemming(tweet) for tweet in tweets]
    return tweets


mapacores = ListedColormap(["mediumorchid", "beige", "orchid"])
nuvem = WordCloud(background_color="black", colormap=mapacores, stopwords=stops_nuvem, width=1000, height=600, max_words=200, max_font_size=120, min_font_size=1)
nuvem.generate(" ".join([limpar_texto(i) for i in nubank]))
fig, ax = plt.subplots(figsize=(20, 15))
plt.imshow(nuvem, interpolation='bilinear')
ax.set_axis_off()

mapacores = ListedColormap(["darkorange", "beige", "bisque"])
nuvem = WordCloud(background_color="black", colormap=mapacores, stopwords=stops_nuvem, width=1000, height=600, max_words=200, max_font_size=120, min_font_size=1)
nuvem.generate(" ".join([limpar_texto(i) for i in inter]))
fig, ax = plt.subplots(figsize=(20, 15))
plt.imshow(nuvem, interpolation='bilinear')
ax.set_axis_off()

mapacores = ListedColormap(["forestgreen", "beige", "lightgreen"])
nuvem = WordCloud(background_color="black", colormap=mapacores, stopwords=stops_nuvem, width=1000, height=600, max_words=200, max_font_size=120, min_font_size=1)
nuvem.generate(" ".join([limpar_texto(i) for i in original]))
fig, ax = plt.subplots(figsize=(20, 15))
plt.imshow(nuvem, interpolation='bilinear')
ax.set_axis_off()

x_treino, y_treino = base["tweet_text"], base["sentiment"]
x_teste, y_teste = teste["tweet_text"], teste["sentiment"]

vectorizer = CountVectorizer(analyzer="word", tokenizer=word_tokenize)

x_treino = preprocessar(x_treino[:])
vetor = vectorizer.fit_transform(x_treino)

modelo = MultinomialNB()
modelo.fit(vetor, y_treino)

x_teste = preprocessar(x_teste[:])
x_teste_vetor = vectorizer.transform(x_teste)
previsoes = modelo.predict(x_teste_vetor)

print(metrics.accuracy_score(y_teste, previsoes))


def pretty_confusion(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(16, 10))
    sns.set(font_scale=1.4)
    sns.heatmap(matrix, annot=True, fmt="d", annot_kws={'size': 10},
                cmap=plt.cm.Blues, linewidths=0.2)
    class_names = ["Negativo", "Positivo", "Neutro"]
    tick_marks = np.arange(len(class_names))
    tick_marks2 = tick_marks + 0.5
    plt.xticks(tick_marks, class_names, rotation=25)
    plt.yticks(tick_marks2, class_names, rotation=0)
    plt.xlabel('Classes Predições')
    plt.ylabel('Classes Reais')
    plt.tight_layout()
    plt.show()


pretty_confusion(y_teste, previsoes)


frases = ["o roxinho é tudo pra mim hoje em dia, resolve minha vida e tem várias vantagens.", "pq me fazes sofrer com confirmações de sms que nuuunca chegam e com um telemarketing que não me atende?", "para quem promete desburocratizar, deixou muito a desejar 🙁", "o marketing deles tá de parabéns com essa assinatura e cartinha escrita a mão, tudo muito fofo.", "Banco Inter lança área de pesquisa e análise de ações com relatórios gratuitos"]
frases_prep = preprocessar(frases[:])
frases_vetor = vectorizer.transform(frases_prep)
previsoes = modelo.predict(frases_vetor)
for f, p in zip(frases, previsoes):
    if p == 0:
        p = "Negativo"
    if p == 1:
        p = "Positivo"
    if p == 2:
        p = "Neutro"
    print(f"Frase: {f}\nSentimento: {p}\n")

sentimentonubank = preprocessar(nubank[:])
sentimentonubank = vectorizer.transform(sentimentonubank)
sentimentonubank = modelo.predict(sentimentonubank)

print("Negativos: " + str((sentimentonubank == 0).sum()))
print("Positivos: " + str((sentimentonubank == 1).sum()))
print("Neutros: " + str((sentimentonubank == 2).sum()))

sentimentointer = preprocessar(inter[:])
sentimentointer = vectorizer.transform(sentimentointer)
sentimentointer = modelo.predict(sentimentointer)

print("Negativos: " + str((sentimentointer == 0).sum()))
print("Positivos: " + str((sentimentointer == 1).sum()))
print("Neutros: " + str((sentimentointer == 2).sum()))

sentimentooriginal = preprocessar(original[:])
sentimentooriginal = vectorizer.transform(sentimentooriginal)
sentimentooriginal = modelo.predict(sentimentooriginal)

print("Negativos: " + str((sentimentooriginal == 0).sum()))
print("Positivos: " + str((sentimentooriginal == 1).sum()))
print("Neutros: " + str((sentimentooriginal == 2).sum()))

bancos = ["Nubank", "Inter", "Original"]
qtde_pos = [(sentimentonubank == 1).sum(), (sentimentointer == 1).sum(), (sentimentooriginal == 1).sum()]
qtde_neg = [(sentimentonubank == 0).sum(), (sentimentointer == 0).sum(), (sentimentooriginal == 0).sum()]

plt.figure(figsize=(5, 10))
plt.subplot(211)
plt.bar(bancos, qtde_pos, color = "navy")
plt.ylabel("Quantidade de Tweets")
plt.title("Sentimento Positivo")
plt.subplot(212)
plt.bar(bancos, qtde_neg, color = "navy")
plt.ylabel("Quantidade de Tweets")
plt.title("Sentimento Negativo")

bradesco = pd.read_csv(r'csvfiles\bradesco.csv')
itau = pd.read_csv(r'csvfiles\itau.csv')
bancodobrasil = pd.read_csv(r'csvfiles\bancodobrasil.csv')

bradesco = bradesco.drop_duplicates()
itau = itau.drop_duplicates()
bancodobrasil = bancodobrasil.drop_duplicates()

bradesco = bradesco.iloc[0:1899, 1]
itau = itau.iloc[0:1899, 1]
bancodobrasil = bancodobrasil.iloc[0:1899, 1]

sentimentobradesco = preprocessar(bradesco[:])
sentimentobradesco = vectorizer.transform(sentimentobradesco)
sentimentobradesco = modelo.predict(sentimentobradesco)

print("Negativos: " + str((sentimentobradesco == 0).sum()))
print("Positivos: " + str((sentimentobradesco == 1).sum()))
print("Neutros: " + str((sentimentobradesco == 2).sum()))


sentimentoitau = preprocessar(itau[:])
sentimentoitau = vectorizer.transform(sentimentoitau)
sentimentoitau = modelo.predict(sentimentoitau)

print("Negativos: " + str((sentimentoitau == 0).sum()))
print("Positivos: " + str((sentimentoitau == 1).sum()))
print("Neutros: " + str((sentimentoitau == 2).sum()))


sentimentobb = preprocessar(bancodobrasil[:])
sentimentobb = vectorizer.transform(sentimentobb)
sentimentobb = modelo.predict(sentimentobb)

print("Negativos: " + str((sentimentobb == 0).sum()))
print("Positivos: " + str((sentimentobb == 1).sum()))
print("Neutros: " + str((sentimentobb == 2).sum()))

bancos = ["Bradesco", "Itaú", "Banco do Brasil"]
qtde_pos = [(sentimentobradesco == 1).sum(), (sentimentoitau == 1).sum(), (sentimentobb == 1).sum()]
qtde_neg = [(sentimentobradesco == 0).sum(), (sentimentoitau == 0).sum(), (sentimentobb == 0).sum()]

plt.figure(figsize=(5, 10))
plt.subplot(211)
plt.bar(bancos, qtde_pos, color = "navy")
plt.ylabel("Quantidade de Tweets")
plt.title("Sentimento Positivo")
plt.subplot(212)
plt.bar(bancos, qtde_neg, color = "navy")
plt.ylabel("Quantidade de Tweets")
plt.title("Sentimento Negativo")
