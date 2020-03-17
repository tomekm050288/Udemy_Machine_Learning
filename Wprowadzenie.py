# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 21:05:54 2020

@author: tm
"""


import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
%matplotlib inline

iris = pd.read_csv(r"C:\Users\Tomasz\Desktop\udemy cheatsheet\Machine Learning\Dane Kurs\iris.data"
                   ,header = None,names = ['petal length', 'petal width','sepal length', 'sepal width', 'species'])

iris.head()

# można sprawdzic rozmiar wczytanego zbioru
# jeśli obiekt ma więcej wymiarów, to można niezależnie sprawdzać każdy z nich
# W skryptach ML, często trzeba zainicjować rozmiary innych obiektów zależnie od
# rozmiaru danych wejściowych. Robi się to korzystając własnie z właściwości shape
iris.shape
iris.shape[0]
iris.shape[1]

# dalej przygotowujemy wykres - tutaj wyznaczenie wartości min i max dla 
# 2 wybranych kolumn z rozmiarami kwiatów. Kiedy chcesz się odwołać do całej kolumny w data frame,
# to w nawiasie kwadratowym podajesz nazwę tej kolumny
x_min, x_max = iris['petal length'].min() - .5, iris['petal length'].max() + .5
y_min, y_max = iris['petal width'].min() - .5, iris['petal width'].max() + .5
 
# każdy gatunek ma być wyświetlony w innym kolorze - definiujemy słownik
colors = {'Iris-setosa':'red', 'Iris-versicolor':'blue', 'Iris-virginica':'green'}
 
# tworzymy obiekt odpowiedzialny za rysowany wykres i jego współrzędne
# instrukcje odtąd aż do plt.show() uruchom zaznaczając cały ten blok kodu
fig, ax = plt.subplots(figsize=(8, 6))
 
# grupujemy dane ze względu na gatunek i rysujemy dane. Korzystamy tu z metody groupby obiektu data frame
# funkcja zwraca klucz identyfikujący nazwę grupy (tutaj jest to nazwa gatunku kwiatu) oraz
# próbki wchodzące w skład tej grupy. To pozwala rysować każdą grupę w innym kolorze
for key, group in iris.groupby(by='species'):
    plt.scatter(group['petal length'], group['petal width'], 
                c=colors[key], label=key)

iris.groupby(by='species').head()



# dodajemy legendę i opis osi
ax.legend()
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
ax.set_title("IRIS DATASET CATEGORIZED")
 
plt.show()
 
# teraz podobny wykres można sporządzić dla sepal
# pamiętaj o uruchomieniu mając zaznaczony blok kodu odtąd aż do plt.show()
# kroki są takie same jak w poprzednim przykładzie
x_min, x_max = iris['sepal length'].min() - .5, iris['sepal length'].max() + .5
y_min, y_max = iris['sepal width'].min() - .5, iris['sepal width'].max() + .5
 
colors = {'Iris-setosa':'red', 'Iris-versicolor':'blue', 'Iris-virginica':'green'}
 
fig, ax = plt.subplots(figsize=(8, 6))
 
for key, group in iris.groupby(by='species'):
    # funkcja scatter przyjmuje argumenty - współrzędne X punktów, współrzędne Y punktów,
    # kolor i nazwę rysowanej grupy
    plt.scatter(group['sepal length'], group['sepal width'], 
                c=colors[key], label=key)
 
ax.legend()
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
ax.set_title("IRIS DATASET CATEGORIZED")
plt.show()
 