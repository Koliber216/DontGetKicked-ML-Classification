# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 11:42:34 2023

@author: tymot
"""

import pandas as pd
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
train_df = pd.read_csv('training.csv') # split for test and train set is ready
test = pd.read_csv('test.csv')




train_df.info()


sns.heatmap(train_df.corr(method="spearman"))

my_columns = [1,10,11]+(list(range(18, 29))) + [31]

df = train_df.iloc[:,my_columns]


# zrobimy sobie EDA z tych danych

df.info()

y = df["IsBadBuy"]
X = df.copy()

del X["IsBadBuy"]


X_train, X_val, y_train, y_val = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)


train = X_train.join(y_train)
val = X_val.join(y_val)

train.info()

# opisy kolumn 
# IsBadBuy - czy git zakup czy nie - target

plt.hist(y_val)
plt.hist(y_train)

# dobre rozłożenie targetu

# typy kolumn

train.head()


##### Color - kolumna zawiera kolor sprzedawanego samochodu




### Transmission - typ skrzyni biegów

train["Transmission"].unique()

# przyjmuje jedynie dwie wartosci - z dokladnoscia do rozmiaru znakow

transmission_dict = {'AUTO':0, 'MANUAL':1}

train['Transmission'] = train['Transmission'].str.upper().map(transmission_dict)


plt.hist(train["Transmission"]) # znacznie mniej jest manuali
train[["Transmission", "IsBadBuy"]].corr(method="spearman") # bardzo mala korelacja

# co ciekawe manuale są tańsze niz automaty

# MMR rzeczy - na pozniej

MMR_train = train.iloc[:, 2:10].join(train["IsBadBuy"]).join(train["VehBCost"])

MMR_corr = MMR_train.corr()
# jak sie okazuje wszystkie są wybitnie mocno skorelowane

# postaram się wniknąć w zależnoci między tymi zmiennymi - co tam się w srodku dzieje


# patrząc na macierz korelacji widać, że clean prices są tak bardzo skorelowane, ze chyba nie ma sensu ich trzymać
mmr_columns_to_drop = ["MMRAcquisitionAuctionCleanPrice", "MMRAcquisitonRetailCleanPrice",
                   "MMRCurrentAuctionCleanPrice", "MMRCurrentRetailCleanPrice"]

MMR_train = MMR_train.drop(columns = mmr_columns_to_drop)

MMR_corr = MMR_train.corr()

# znacznie lepiej

mmr_by_target = train.groupby(["IsBadBuy"]).mean()
# ciekawe jest to, że bad buys ma zarówno niższe ceny na rynkach jak i te ceny zakupu

sns.displot(kind="kde", data=MMR_train,x="IsBadBuy", facet_kws={'sharey': False, 'sharex': False})
MMR_melt = MMR_train.melt(id_vars=['IsBadBuy'])

MMR_melt["variable"] = MMR_melt["variable"] + MMR_melt["IsBadBuy"].astype("str")

sns.displot(kind='kde', data=MMR_melt, col='variable', col_wrap=2, x='value', hue="variable", facet_kws={'sharey': False, 'sharex': False})


# niektóre obserwacje mają cene równą 0 - troche to jest głupie
no_data_train = train.loc[train["MMRAcquisitionAuctionAveragePrice"]==0]
no_data_train = train.loc[train["MMRCurrentRetailCleanPrice"]==0]

train.mean()
no_data_train.mean()
# tam gdzie są braki jest troszke wiecej zlych zakupów
no_data_train.hist(fill="IsBadBuy")

sns.displot(MMR_train.loc[MMR_train["IsBadBuy"]==0].loc[:, MMR_train.columns != "IsBadBuy"], kind="kde")
sns.displot(MMR_train.loc[MMR_train["IsBadBuy"]==1].loc[:, MMR_train.columns != "IsBadBuy"], kind="kde")

# tutaj widać jasno, że Auction Prices są bardzo powiązane

sns.distplot(train.loc[train["IsBadBuy"]==0]["VehBCost"], hist=False, rug=True)
sns.distplot(train.loc[train["IsBadBuy"]==1]["VehBCost"], hist=False, rug=True)

plt.show()

g = sns.FacetGrid(train, col="IsBadBuy")
g.map(sns.kdeplot, "VehBCost")


# PRIMEUNIT - czy samochód bylby w większym zapotrzebowaniu

train["PRIMEUNIT"].unique()

primeunit_dict = {'NO':0, 'YES':1}

train['PRIMEUNIT'] = train['PRIMEUNIT'].str.upper().map(primeunit_dict)

# AUCGUART - czy jest gwarancja


train["AUCGUART"].unique()

aucguart_dict = {'GREEN':3, 'YELLOW':2, "RED":1, float("nan"):0}

train['AUCGUART'] = train['AUCGUART'].str.upper().map(aucguart_dict)


# BYRNO - buyer number - unique for buyers

buyers = train.groupby("BYRNO").IsBadBuy.agg(["sum", "count"])
buyers["ratio"] = buyers["sum"]/buyers["count"]
buyers["ratio"].mean() # sredni około 15% u kazdego kupującego jest cytryn


train[["BYRNO", "IsBadBuy"]].corr() # w teorii brak korelacji

# wobec tego wydaje mi sie ze mozna wywalić kupującego - nieistotna kolumna



columns_to_drop = ["BYRNO"]



def encode_categorical(df):
    
    # TRANSMISSION
    df["IsPrimeunitNA"] = np.where(df["PRIMEUNIT"].isna(), 1, 0)
    
    primeunit_dict = {'NO':0, 'YES':1, float("nan"):0} # nie jestem pewien czy nie lepiej bedzie zamienic na odwrot
    df['PRIMEUNIT'] = df['PRIMEUNIT'].str.upper().map(primeunit_dict)
    
    
    aucguart_dict = {'GREEN':2, 'YELLOW':1, "RED":0, float("nan"):0}
    df['AUCGUART'] = df['AUCGUART'].str.upper().map(aucguart_dict)
    
    
    


def drop_columns(df):
    
    return df.drop(columns = columns_to_drop)
