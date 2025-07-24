#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: burakertan
"""
# %% Titanic Projesi
#Titanik Veri Seti


#Kullanacağımız Kütüphaneler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
#%% Data'yı Okuma Kısmı

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
test_passengerid = test_df["PassengerId"]
#%%Data hakkında ilk fikirler
train_df.columns
train_df.head()
train_df.info()
train_df.describe()

#%%Datanın Featureları
#yazılan bilgiler train_df.info() ile elde edildi
#PassengerId = yolcu numarası uniq (int türünde)
#Survived = Yaşayıp/yaşamama durumu 0/1 (int türünde) #0-549 1-342
#Pclass = Yolcunun sınıfı 1/2/3 (int türünde) #1-216 2-184 3-491
#Name = Yolcunun ismi (Obje türünde)
#Sex = Cinsiyet male/female (Obje türünde) #male-577 female-314
#Age  = Yolcunun yaşı (float türünde)
#SibSp = Kardeş ve eşlerin sayısı (int türünde)
#0-608 1-209 2-28 3-16 4-18 5-5 8-7
#Parch = Parent ve Çocuk sayısı (int türünde)
#0-678 1-118 2-80 3-5 4-4 5-5 6-1
#Ticket = ticket kodu (float türünde)
#Fare = Bilet için harcanan para (float türünde)
#Cabin = Kaldıkları odalar (Obje türünde)
#Embarked = Yolcuların nereden bindiği Q/S/C (Obje türünde) #S-644 C-168 Q-77

#Kategorik olanlar: Survived,Sex,Pclass,Embarked,Cabin,Name,Ticket,SibSp ve Parch
#Sayısal olanlar :Fare,Age ve PassengerId

#%%Kategorik Değişkenlerin Görselleştirmesi

def bar_plot(variable):
    var = train_df[variable]
    varValue = var.value_counts()
    plt.figure(figsize=(9,3))
    plt.bar(varValue.index,varValue)
    plt.xticks(varValue.index,varValue.index.values)
    plt.ylabel("Sıklık")
    plt.title(variable)
    plt.show()
    print("{}: \n {}".format(variable,varValue))
    
categorys = ["Survived","Sex","Pclass","Embarked","Cabin","Name","Ticket","SibSp","Parch"]

for each in categorys:
    bar_plot(each)
#%%Sayısal Değişkenlerin Görselleştirmesi

numericVar = ["Fare","Age","PassengerId"]

def hist_plot(variable):
    plt.figure(figsize=(9,3))
    plt.hist(train_df[variable],bins=200)
    plt.xlabel(variable)
    plt.ylabel("Sıklık")
    plt.title("{} dağılımı".format(variable))
    plt.show()


for each in numericVar:
    hist_plot(each)

#%%Data'nın Detaylı Analizi

#Pclass vs Survived
train_df[["Pclass","Survived"]].groupby(["Pclass"],as_index = False).mean().sort_values(by="Survived",ascending = False)
#%%
#Sex vs Survived
train_df[["Sex","Survived"]].groupby(["Sex"],as_index = False).mean().sort_values(by="Survived",ascending = False)
#%%
#SibSp vs Survived
train_df[["SibSp","Survived"]].groupby(["SibSp"],as_index = False).mean().sort_values(by="Survived",ascending = False)
#%%
#Parch vs Survived
train_df[["Parch","Survived"]].groupby(["Parch"],as_index = False).mean().sort_values(by="Survived",ascending = False)
#%% Outlier Tespiti

def detect_outlier(df,features):
    outlier_indices = []
    for each in features:
        Q1 = np.percentile(df[each],25)
        Q3 = np.percentile(df[each],75)
        IQR = Q3 - Q1
        outlier_step = IQR * 1.5
        outlier_list_col = df[(df[each] < Q1-outlier_step) | (df[each] > Q3-outlier_step)].index
        outlier_indices.extend(outlier_list_col)
    
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i,v in outlier_indices.items() if v > 2)
    
    return multiple_outliers

train_df.loc[detect_outlier(train_df,["Age","SibSp","Parch","Fare"])]
train_df = train_df.drop(detect_outlier(train_df,["Age","SibSp","Parch","Fare"]),axis = 0).reset_index(drop = True)
#%% Eksik Değerleri Bulma

train_df_len = len(train_df)
train_df = pd.concat([train_df,test_df],axis = 0).reset_index(drop = True)
train_df.columns[train_df.isnull().any()]
train_df.isnull().sum()
#%% Eksik Değerleri Doldurma
#Embarked 2 boş değer var
#Fare 1 boş değer var

train_df[train_df["Embarked"].isnull()]
train_df.boxplot(column = "Fare",by = "Embarked")
plt.show()
#C'den binmiş olma olasılıkları yüksek
train_df["Embarked"] = train_df["Embarked"].fillna("C")
train_df[train_df["Embarked"].isnull()] #Index: []
train_df[train_df["Fare"].isnull()]
np.mean(train_df[train_df["Pclass"]==3]["Fare"])
train_df["Fare"] = train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"]==3]["Fare"]))
train_df[train_df["Fare"].isnull()] #Index: []
#%%Korelasyon Matrisi
#SibSp-Parch-Age-Fare-Survived arasındaki korelasyon

list_corr = ["SibSp","Parch","Age","Fare","Survived"]
sns.heatmap(train_df[list_corr].corr(),annot=True,fmt = ".2f")

#En iyi korelasyonlar Fare-Survived,Parch-SibSp,Age-Fare
#%%SibSp-Survived arasındaki ilişki

g = sns.catplot(x = "SibSp",y = "Survived",data = train_df,kind = "bar",height= 6)
g.set_ylabels("Survived Probabilty")
plt.show()
#garip bir sonuç çıktı data kaynaklı büyük ihtimalle
#buradan yeni bir feature çıkabilir
#%%Parch-Survived arasındaki ilişki

g = sns.catplot(x = "Parch",y = "Survived",data = train_df,kind = "bar",height= 6)
g.set_ylabels("Survived Probabilty")
plt.show()
#%%Parch-Survived arasındaki ilişki

g = sns.catplot(x = "Parch",y = "Survived",data = train_df,kind = "bar",height= 6)
g.set_ylabels("Survived Probabilty")
plt.show()
#%%Pclass-Survived arasındaki ilişki

g = sns.catplot(x = "Pclass",y = "Survived",data = train_df,kind = "bar",height= 6)
g.set_ylabels("Survived Probabilty")
plt.show()
#%%Pclass-Survived arasındaki ilişki

g = sns.catplot(x = "Pclass",y = "Survived",data = train_df,kind = "bar",height= 6)
g.set_ylabels("Survived Probabilty")
plt.show()
#%%Age-Survived arasındaki ilişki

g = sns.FacetGrid(train_df,col ="Survived")
g.map(sns.displot,"Age",bins =25)
plt.show()
#%%Pclass-Survived-Age arasındaki ilişki

g = sns.FacetGrid(train_df,col ="Survived",row = "Pclass")
g.map(plt.hist,"Age",bins =25)
g.add_legend()
plt.show()
#Pclass model eğitimi için önemli

#%% Embarked -- Sex -- Pclass--Survived
g = sns.FacetGrid(train_df,row = "Embarked")
g.map(sns.pointplot,"Pclass","Survived","Sex")
g.add_legend()
plt.show()
#kadınlar daha yüksek yaşama sahip
#pclass 3 te erkekler daha yüksek yaşama sahip
#embarked ve sex eğitimde kullanılabilir
#%%Embarked -- Sex -- Fare-- Survived
g = sns.FacetGrid(train_df,row = "Embarked",col = "Survived")
g.map(sns.barplot,"Sex","Fare")
g.add_legend()
plt.show()
#Daha çok para ödeyen daha çok hayatta kalmış
#fare kategorik eğitim için kullanılabilir
#%%Eksik Yaş Verisini Doldurma
sns.catplot(x = "Sex",y = "Age",data = train_df,kind = "box")
plt.show()
#çok yakın mantıklı iş çıkmaz
sns.catplot(x = "Sex",y = "Age",hue="Pclass",data = train_df,kind = "box")
plt.show()
#en yaşlılar p1 sonra p2 sonra p3'te
#p2 de olan bir yolcunun cinsiyet fark etmeksizin 30larında diyebiliriz
sns.catplot(x = "Parch",y = "Age",data = train_df,kind = "box")
plt.show()
sns.catplot(x = "SibSp",y = "Age",data = train_df,kind = "box")
plt.show()
train_df["Sex"] = [1 if i == "male" else 0 for i in train_df["Sex"]]
sns.heatmap(train_df[["Age","Sex","SibSp","Parch","Pclass"]].corr(),annot=True)
plt.show()
#Age-Pclass arası güçlü bir ilişki

from sklearn.linear_model import LinearRegression
features_for_age = ["Pclass", "Sex", "SibSp", "Parch", "Fare"]
age_not_nan = train_df.dropna(subset=["Age"])
X_train = age_not_nan[features_for_age]
y_train = age_not_nan["Age"]
lr = LinearRegression()
lr.fit(X_train, y_train)

index_nan_age = list(train_df["Age"][train_df["Age"].isnull()].index)
predicted_ages = []
for i in index_nan_age:
    row = train_df.loc[i, features_for_age].values.reshape(1, -1)
    age_pred = lr.predict(row)[0]
    train_df.loc[i, "Age"] = age_pred
    predicted_ages.append((i, age_pred))

#lineer model kullanarak yaşı doldurdum
#%% Feature Engineering Kısmı

name = train_df["Name"]
train_df["Title"] = [i.split(".")[0].split(",")[-1].strip() for i in name]
sns.countplot(x = "Title",data = train_df)
plt.xticks(rotation = 60)
plt.show()

train_df["Title"] = train_df["Title"].replace(["Lady","the Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"],"other")
train_df["Title"] = [0 if i == "Master" else 1 if i == "Miss" or i == "Ms" or i == "Mlle" or i == "Mrs" else 2 if i == "Mr" else 3 for i in train_df["Title"]]
sns.countplot(x = "Title",data = train_df)
plt.xticks(rotation = 60)
plt.show()


g = sns.catplot(x = "Title",y = "Survived",data = train_df,kind = "bar")
g.set_xticklabels(["Master","Mrs","Mr","Other"])
g.set_ylabels("Surv Prob")
plt.show()
#%%
train_df.drop(labels = ["Name"],axis = 1,inplace = True)

#%%Feature Engineering Family Size

train_df["F_size"] = train_df["Parch"] + train_df["SibSp"] + 1

g = sns.catplot(x = "F_size",y = "Survived",data = train_df,kind = "bar")
g.set_ylabels("Survived")
plt.show()

train_df["family_size"] = [1 if i < 3 else 0 for i in train_df["F_size"]]

sns.countplot(x = "family_size",data = train_df)
plt.show()

g = sns.catplot(x = "family_size",y = "Survived",data = train_df,kind = "bar")
g.set_ylabels("Survived")
plt.show()
#%% Feature Engineerin Embarked

train_df = pd.get_dummies(train_df, columns=["Embarked"])
#%%
train_df.head()
train_df.loc[:, "Embarked_C":"Embarked_S"] = train_df.loc[:, "Embarked_C":"Embarked_S"].astype(int)
train_df.head()
#%%Feature Engineering Ticket

tickets =[]

for i in list(train_df.Ticket):
    if not i.isdigit():
        tickets.append(i.replace("."," ").replace("/"," ").strip().split(" ")[0])
    else:
        tickets.append("x")
train_df["Tickets"] = tickets

train_df = pd.get_dummies(train_df,columns=["Tickets"],prefix = "T")

#%% Feature Engineerin Pclass

sns.countplot(x = "Pclass",data = train_df)
train_df["Pclass"] = train_df["Pclass"].astype("int")
train_df = pd.get_dummies(train_df,columns=["Pclass"])
#%%
train_df.head()
train_df[["Pclass_1", "Pclass_2", "Pclass_3"]] = train_df[["Pclass_1", "Pclass_2", "Pclass_3"]].astype(int)
train_df.head()

#%% Feature Engineerin Sex

train_df = pd.get_dummies(train_df,columns=["Sex"])


#%%
train_df.head()
train_df[["Sex_0", "Sex_1"]] = train_df[["Sex_0", "Sex_1"]].astype(int)
train_df.head()

#%% Feature Engineering Drop PassengerID ve Cabin

train_df.drop(labels = ["PassengerId","Cabin","Ticket"],axis = 1, inplace = True)

#%%Modelling finally

from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
#%%

test = train_df[train_df_len:]
test.drop(labels = ["Survived"],axis = 1,inplace = True)
#%%
test = train_df[train_df_len:].drop(labels=["Survived"], axis=1)

# Eğitim veri çerçevesini ayırma
train = train_df[:train_df_len]

# x_train ve y_train için sütunları ayırma
x_train = train.drop(labels=["Survived"], axis=1)
y_train = train["Survived"]

# Eğitim ve test setlerini bölme
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.33, random_state=42)

# Çıktıları kontrol etme
print("x_train", len(x_train))
print("x_test", len(x_test))
print("y_train", len(y_train))
print("y_test", len(y_test))
print("test", len(test))
#%%Simple Logistic Regression

logreg = LogisticRegression()
logreg.fit(x_train,y_train)
acc_for_train = round(logreg.score(x_train,y_train)*100,2)
acc_for_test = round(logreg.score(x_test,y_test)*100,2)

print("Train için Accuracy  = {}".format(acc_for_train))
print("Test için Accuracy  = {}".format(acc_for_test))

#%% HyperParameter Tuning -- GridSearch --CrossValidation

random_state = 42

classifier = [DecisionTreeClassifier(random_state = random_state),
              SVC(random_state=random_state,max_iter=20000),
              RandomForestClassifier(random_state=random_state),
              LogisticRegression(random_state=random_state,max_iter=1000),
              KNeighborsClassifier()]

dt_param_grid = {"min_samples_split" : range(10,500,20),
                "max_depth": range(1,20,2)}

svc_param_grid =    { "kernel": ["rbf"],
    "gamma": [0.01, 0.1, 1],
    "C": [1, 10, 100]}

rf_param_grid = {"max_features": [1,3,10],
                "min_samples_split":[2,3,10],
                "min_samples_leaf":[1,3,10],
                "bootstrap":[False],
                "n_estimators":[100,300],
                "criterion":["gini"]}

logreg_param_grid = {"C":np.logspace(-3,3,7),
                    "penalty": ["l1","l2"]}

knn_param_grid = {"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(),
                 "weights": ["uniform","distance"],
                 "metric":["euclidean","manhattan"]}
classifier_param = [dt_param_grid,
                   svc_param_grid,
                   rf_param_grid,
                   logreg_param_grid,
                   knn_param_grid]

cv_result = []
best_estimators = []

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

for i in range(len(classifier)):
    if isinstance(classifier[i], (SVC, KNeighborsClassifier, LogisticRegression)):
        clf = GridSearchCV(classifier[i], param_grid=classifier_param[i], cv=StratifiedKFold(n_splits=10),
                           scoring="accuracy", n_jobs=-1, verbose=1)
        clf.fit(x_train_scaled, y_train)
    else:
        clf = GridSearchCV(classifier[i], param_grid=classifier_param[i], cv=StratifiedKFold(n_splits=10),
                           scoring="accuracy", n_jobs=-1, verbose=1)
        clf.fit(x_train, y_train)
    cv_result.append(clf.best_score_)
    best_estimators.append(clf.best_estimator_)
    print(cv_result[i])
    print(best_estimators[i])

for model in best_estimators:
    test_accuracy = model.score(x_test_scaled, y_test)
    print(f"{model.__class__.__name__} Test Accuracy: {test_accuracy:.4f}"),

cv_results = pd.DataFrame({"Cross Validation Means":cv_result, "ML Models":["DecisionTreeClassifier", "SVM","RandomForestClassifier",
             "LogisticRegression",
             "KNeighborsClassifier"]})

g = sns.barplot(x="Cross Validation Means", y="ML Models", data=cv_results)
g.set_xlabel("Mean Accuracy")
g.set_title("Cross Validation Scores")

#%%Modeling Ensemble

votingC = VotingClassifier(estimators = [("dt",best_estimators[0]),
                                        ("rfc",best_estimators[2]),
                                        ("lr",best_estimators[3])],
                                        voting = "soft", n_jobs = -1)
votingC = votingC.fit(x_train, y_train)
print(accuracy_score(votingC.predict(x_test),y_test))

#%%prediction
test_survived = pd.Series(votingC.predict(test), name = "Survived").astype(int)
results = pd.concat([test_passengerid, test_survived],axis = 1)
results.to_csv("titanic.csv", index = False)

train_accuracy = votingC.score(x_train, y_train)
test_accuracy = votingC.score(x_test, y_test)

y_pred = votingC.predict(x_test)
test_accuracy_manual = accuracy_score(y_test, y_pred)

print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Manual Test Accuracy (using accuracy_score): {test_accuracy_manual:.4f}")

print(results)

















