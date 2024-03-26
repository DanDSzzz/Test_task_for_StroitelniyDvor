import pandas as pd
import numpy as np
import seaborn as sns

np.random.seed(42)  # Для воспроизводимости результатов

# Определяем размер датафрейма
n_rows = 100

# Функция для генерации даты рождения и возраста
def generate_dob(n, start_year=1950, end_year=2000):
    years = np.random.randint(start_year, end_year, n)
    months = np.random.randint(1, 13, n)
    days = np.random.randint(1, 29, n)
    return [pd.Timestamp(year=years[i], month=months[i], day=days[i]) for i in range(n)]

# Функция для генерации дат увольнения и трудоустройства
def generate_dates(n, start_year=2010, end_year=2020):
    start_dates = generate_dob(n, start_year, end_year)
    end_dates = [date + pd.DateOffset(years=np.random.randint(1, 5)) for date in start_dates]
    return start_dates, end_dates

# Создаем датафрейм
df = pd.DataFrame({
    'Увольнение': generate_dates(n_rows, 2015, 2020)[1],
    'Трудоустройство': generate_dates(n_rows, 2010, 2015)[0],
    'Стаж в организации': np.random.randint(1, 10, n_rows),
    'Стаж общий': np.random.randint(1, 30, n_rows),
    'Дата рождения': generate_dob(n_rows),
    'Пол': np.random.choice(['Муж', 'Жен'], n_rows),
    'Семейное положение': np.random.choice(['Женат/Замужем', 'Не женат/Не замужем'], n_rows),
    'Наличие детей': np.random.choice([True, False], n_rows),
    'Регион': np.random.choice(['Регион 1', 'Регион 2', 'Регион 3'], n_rows),
    'Заработная плата': np.random.randint(30000, 200000, n_rows),
    'Выполнение KPI': np.random.choice(['Выполнил', 'Не выполнил'], n_rows),
    'Должность': np.random.choice(['Менеджер', 'Аналитик', 'Разработчик'], n_rows),
    'Количество повышений за год': np.random.randint(0, 3, n_rows),
    'Бывший студент/нанятый сотрудник': np.random.choice(['Бывший студент', 'Нанятый сотрудник'], n_rows),
    'Списочная численность персонала': np.random.randint(100, 1000, n_rows),
    'Выручка на одного сотрудника': np.random.randint(500000, 5000000, n_rows) / np.random.randint(100, 1000, n_rows)
})

# Подсчет возраста сотрудников на момент увольнения
df['Возраст'] = ((df['Увольнение'] - df['Дата рождения']).dt.days / 365.25).astype(int)

# Вывод первых 5 строк
print(df.head()) # Рассчитываем стаж в организации как разницу между датой увольнения и датой трудоустройства
df['Стаж в организации (годы)'] = (df['Увольнение'] - df['Трудоустройство']).dt.days / 365.25

# Округляем стаж до полных лет
df['Стаж в организации (годы)'] = df['Стаж в организации (годы)'].round()

# Выводим обновленный датафрейм
print(df[['Увольнение', 'Трудоустройство', 'Стаж в организации (годы)']].head())


import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt

# Многофакторный ANOVA анализ
model = ols('Q("Стаж в организации (годы)") ~ C(Q("Пол")) + C(Q("Семейное положение")) + C(Q("Наличие детей")) + C(Q("Должность"))', data=df).fit()
anova_results = sm.stats.anova_lm(model, typ=2)
print("Многофакторный ANOVA анализ:")
print(anova_results)

# Извлекаем суммы квадратов для каждого фактора из таблицы ANOVA
sum_sq = anova_results['sum_sq']
total_sum_sq = sum_sq.sum()

# Создание круговой диаграммы
labels = sum_sq.index
sizes = sum_sq.values / total_sum_sq * 100  # Процентное соотношение каждого фактора

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title('Влияние различных факторов на стаж в организации')
plt.show()

# Визуализация влияния каждого фактора на стаж в организации
for var in ['Пол', 'Семейное положение', 'Наличие детей', 'Должность']:
    plt.figure(figsize=(8, 6))
    sns.barplot(x=var, y='Стаж в организации (годы)', data=df, ci=None, palette='pastel')
    plt.title(f'Влияние {var} на стаж в организации')
    plt.show()
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Список категориальных переменных для проведения ANOVA
categorical_vars = ['Пол', 'Семейное положение', 'Наличие детей', 'Должность']

# Проведение однофакторного ANOVA для каждой категориальной переменной
for var in categorical_vars:
    model = ols(
        'Q("Стаж в организации (годы)") ~ C(Q("Пол"))', data=df).fit()
    anova_results = sm.stats.anova_lm(model, typ=2)  # Используем тип 2 ANOVA, т.к. он обычно предпочтителен
    print(f"ANOVA для влияния '{var}' на 'Стаж в организации (годы)':")
    print(anova_results)
    print("\n")