import GetOldTweets3 as got
import re
import pandas as pd


def pegar_tweets(banco, user, nomebanco):
    lista_tweets = []
    lista_bancos = ["bradesco", "banco do brasil", "santander", "itaú", "caixa",
                    "agibank", "inter", "neon", "nubank", "next", "original"]
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(banco)\
                                               .setMaxTweets(5000)\
                                                .setEmoji("unicode")
    tweet = got.manager.TweetManager.getTweets(tweetCriteria)
    for i in range(0, len(tweet)):
        if tweet[i].username != user and "RT" not in tweet[i].text:
            data = tweet[i].date.strftime("%d-%m-%y")
            tt = (str(tweet[i].text)).lower()
            tt = re.sub(r'http\S+', "", tt)
            tt = re.sub(r'@\S+ ?', "", tt)
            tt = re.sub(r'\n', "", tt)
            lista_tweets.append([data, tt])
    lista_bancos.remove(nomebanco)
    for b in lista_bancos:
        for t in lista_tweets:
            if b in t[1]:
                lista_tweets.remove(t)
    return lista_tweets


pd.DataFrame(pegar_tweets("banco nubank", "nubank", "nubank")).to_csv(r'csvfiles\nubank.csv', index = False)
pd.DataFrame(pegar_tweets("banco inter", "Bancointer", "inter")).to_csv(r'csvfiles\inter.csv', index = False)
pd.DataFrame(pegar_tweets("banco original", "BancoOriginal", "original")).to_csv(r'csvfiles\original.csv', index = False)

pd.DataFrame(pegar_tweets("bradesco", "bradesco", "bradesco")).to_csv(r'csvfiles\bradesco.csv', index = False)
pd.DataFrame(pegar_tweets("banco itaú", "itau", "itaú")).to_csv(r'csvfiles\itau.csv', index = False)
pd.DataFrame(pegar_tweets("banco do brasil", "BancodoBrasil", "banco do brasil")).to_csv(r'csvfiles\bancodobrasil.csv', index = False)
