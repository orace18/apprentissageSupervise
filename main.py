import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sns


titanic = sns.load_dataset('titanic')
titanic.shape
titanic.head()

titanic = titanic[['survived', 'pclass', 'sex', 'age']]
titanic.dropna(axis=0, inplace= True)
titanic['sex'].replace(['male', 'female'], [0,1], inplace=True)
titanic

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=10)
y = titanic['survived']
X = titanic.drop('survived', axis=1)
model.fit(X, y)
model.score(X, y)


def tonsiunmintcheke(model, pclass=1, sex=0, age=23):
  x = np.array([pclass, sex, age]).reshape(1,3)
  print(model.predict(x))

tonsiunmintcheke(model)
