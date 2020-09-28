import pandas as pd
from matplotlib import pyplot as plt

dateparse = lambda dates: pd.datetime.strptime(dates, "%d-%m-%y")
nubank = pd.read_csv(r'csvfiles\nubank.csv', parse_dates=[0], index_col=[0], date_parser=dateparse)
inter = pd.read_csv(r'csvfiles\inter.csv', parse_dates=[0], index_col=[0], date_parser=dateparse)
original = pd.read_csv(r'csvfiles\original.csv', parse_dates=[0], index_col=[0], date_parser=dateparse)

nubank = nubank.dropna(how='all')
inter = inter.dropna(how='all')
original = original.dropna(how='all')

nubank.columns = ["tweet"]
nubank = nubank.drop_duplicates()
inter.columns = ["tweet"]
inter = inter.drop_duplicates()
original.columns = ["tweet"]
original = original.drop_duplicates()

bancos = ["Nubank", "Inter", "Original"]
qtde = [len(nubank), len(inter), len(original)]

fig, ax = plt.subplots(figsize=(10, 7))
ax.bar(bancos, qtde, width=0.5, color = "navy")
plt.ylabel("Quantidade de Tweets")

gf1 = nubank
gf1.iloc[:, 0] = 1
gf1 = gf1.resample("W").sum()
gf1.index = gf1.index.strftime('%d/%m/%Y')
gf2 = inter
gf2.iloc[:, 0] = 1
gf2 = gf2.resample("W").sum()
gf2.index = gf2.index.strftime('%d/%m/%Y')
gf3 = original
gf3.iloc[:, 0] = 1
gf3 = gf3.resample("W").sum()
gf3.index = gf3.index.strftime('%d/%m/%Y')

plt.figure(figsize=(20, 12))
plt.subplot(311)
plt.bar(gf2.index, gf2["tweet"], color = "navy")
plt.title("Banco Inter")
plt.xticks([])
plt.subplot(312)
plt.bar(gf1.index, gf1["tweet"], color = "navy")
plt.title("Banco Nubank")
plt.xticks([])
plt.subplot(313)
plt.bar(gf3.index, gf3["tweet"], color = "navy")
plt.title("Banco Original")
plt.xticks([])
