import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import pandas as pd
from sklearn.decomposition import PCA

def linear_regression(x, y):
    model = LinearRegression()
    model.fit(x, y)
    plt.scatter(x, y)
    plt.plot(x, model.predict(x), color='purple')
    plt.xlabel('Благоприятность погоды')
    plt.ylabel('Спрос на аренду велосипедов')
    plt.show()
    return model

def predict(model, y):
    cnt = model.predict([[y]])
    return cnt


"""def plot_2d_graph(X):
    pca = PCA(n_components=2)
    pca.fit(X)
    X_pca = pca.transform(X)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data['cnt'], cmap='viridis')
    plt.xlabel('Главный компонент 1')
    plt.ylabel('Главный компонент 2')
    plt.colorbar()
    plt.show()"""

def lasso(x, y):
    lasso = Lasso(alpha=0.1)
    lasso.fit(x, y)
    max_coef_idx = np.argmax(lasso.coef_)
    max_coef_name = x.columns[max_coef_idx]
    return max_coef_name

def graph(x):
    y = x.drop('cnt', axis=1)

    # Уменьшить размерность пространства
    data_2d = PCA(n_components=2).fit_transform(y)

    # Добавить столбец с целевым признаком
    data_2d = np.hstack((data_2d, data['cnt'].values.reshape(-1, 1)))

    # Построить график
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=data_2d[:, 2])
    plt.xlabel('Первая главная компонента')
    plt.ylabel('Вторая главная компонента')
    plt.colorbar()
    plt.show()

data = pd.read_csv('bikes_rent.csv')

x = data[['weathersit']]
y = data['cnt']
x2 = data[['weathersit', 'cnt']]
model = linear_regression(x.to_numpy(), y)

for i in range(1, 4):
    print('Спрос на аренду велосипедов при благоприятности погоды: ' + f'{i} ' + f'{predict(model, i)}')

#plot_2d_graph(data)
graph(data)
data = data.drop('cnt', axis=1)
breakp = lasso(data, y)
print('Признак, оказывающий наибольшее влияние на спрос: ' + f'{breakp}')