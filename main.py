import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy import stats
from scipy.stats import ttest_ind
from scipy.stats import binomtest
from sklearn.linear_model import LinearRegression
if __name__ == '__main__':
    print('hello')
    cars = pd.read_csv('khl.csv')
    columns3 = cars.columns.tolist()
    columns9 = cars.select_dtypes(include=[int, float]).columns.to_numpy()
    print(columns9)
    print(columns3)


    print("1-ое ЗАДАНИЕ:\n")
    print("Количество наблюдений: " + f"{cars.shape[0]}")
    print("Количество переменных: " + f"{cars.shape[1]}")
    print("Типы данных: " + f"{set(list(map(lambda x: 'text' if x == 'object' else 'number', cars.dtypes.values)))}")
    print("Количество пропущенных значений: " + f"{cars.isnull().values.sum()}\n")
    cars.fillna(cars.mean(numeric_only=True), inplace=True)

    print("3-е ЗАДАНИЕ:\n")
    # Гистограммы
    # for column in columns9:
    #   sns.histplot(cars, x=column, kde=True)
    #   plt.show()
    #   time.sleep(0.8)

    # Боксплоты
    # for column in columns9:
    #   sns.boxplot(cars, y=column)
    #   plt.show()

    # Диаграммы рассеивания
    #  pair_grid_plot = sns.PairGrid(cars)
    #  pair_grid_plot.map(plt.scatter)
    # plt.show()

    print("4-ое ЗАДАНИЕ\n")
    for column in columns9:
        q1 = cars[column].quantile(0.25)
        q3 = cars[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        print("Выбросы в столбце " + f"{column}: " + f"{cars[(cars[column] < lower) | (cars[column] > upper)]}\n")
        cars = cars[(cars[column] > lower) & (cars[column] < upper)]
        print(cars)

    print("2-ое ЗАДАНИЕ:\n")
    print("     Средние значения\n" + f"{cars.mean(numeric_only= True)}\n")
    print("     Дисперсии\n" + f"{cars.var(numeric_only=True)}\n")
    print("     Корреляции\n" + f"{cars.corr(numeric_only=True)}\n")
    print("     Минимумы\n" + f"{cars.min(numeric_only=True)}\n")
    print("     Максимумы\n" + f"{cars.max(numeric_only=True)}\n")
    print("     Квартили\n" + f"{cars.quantile([0.25, 0.5, 0.75],numeric_only=True)}\n")


    print("5-ое ЗАДАНИЕ\n")
    # гипотеза о том, что нападающие играют лучше защитников
    defense_data = cars[cars['position'] == 'D']
    forward_data = cars[cars['position'] == 'F'][:len(defense_data)]
    print(defense_data['pl_min'].var(), forward_data['pl_min'].var())
    print(len(forward_data), len(defense_data))
    test, p = stats.mannwhitneyu(forward_data['pl_min'], defense_data['pl_min'])

    print(p)
    print(test)
    # Устанавливаем уровень значимости
    alpha = 0.05

    # Оцениваем p-значение
    if p < alpha:
        print("Гипотеза отвергается. Защитники играют лучше нападающих.")
    else:
        print("Гипотеза не отвергается.")

    mean = np.mean(cars['pl_min'])
    sс = np.std(cars['pl_min'])

    # Вычисляем ожидаемые частоты для нормального распределения
    datas = sorted(set(cars['pl_min']))
    print(datas)
    expected_freq_norm = [len(cars['pl_min']) * stats.norm.cdf(data, loc=mean, scale=sс) for data in datas]
    # Рассчитываем наблюдаемые частоты и частоты ожидаемые для распределения Пуассона
    observed_freq, _ = np.histogram(cars['pl_min'], bins=len(expected_freq_norm))
    observed_freq_arr = np.array(observed_freq)
    expected_freq_arr = np.array(expected_freq_norm)
    expected_freq_arr_norm = np.sum(observed_freq_arr) / np.sum(expected_freq_norm) * expected_freq_arr
    expected_freq_puas = [sum(observed_freq_arr) * ((mean**data * math.e**(-mean))/data) for data in datas[0:17]+datas[18:32]]
    expected_freq_puas.insert(17,sum(observed_freq_arr) * ((mean**0 * math.e**(-mean))))
    print(mean)
    print(sum(observed_freq_arr))
    expected_freq_arr_puas = np.array(expected_freq_puas)
    print(expected_freq_arr_puas)
    # Выполняем критерий Пирсона
    chi2_stat, p_val = stats.chisquare(observed_freq_arr, f_exp=expected_freq_arr_norm)
    ks_stat, ks_pvalue = stats.chisquare(observed_freq_arr, f_exp=expected_freq_arr_puas)
    print("Хи-квадрат статистика:", chi2_stat)
    print("p-value:", p_val)

    # Вывести значение критерия и p-значение
    print("Хи-Квадрат:", ks_stat)
    print("p-значение:", ks_pvalue)
    # Проверяем статистическую значимость
    alpha = 0.05
    if p_val < alpha:
        print("Данные не подчиняются нормальному распределению")
    else:
        print(" ")

    if ks_pvalue < alpha:
        print("Нулевая гипотеза отвергается, распределение не является распределением Пуассона.")
    else:
        print("Нулевая гипотеза не отвергается, распределение может быть распределением Пуассона.")

    print("6-ое ЗАДАНИЕ\n")
    x = cars['pl_min'].values.reshape(-1,1)
    y = cars['shoots'].values.reshape(-1,1)
    model = LinearRegression()
    model.fit(x,y)
    plt.scatter(x, y, color='blue')
    plt.plot(x, model.predict(x), color='red', linewidth=3)
    plt.xlabel('Коэффициент полезности')
    plt.ylabel('Броски по воротам')
    plt.title('Линейная регрессия')
    plt.show()

