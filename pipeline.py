import base_classes
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import ast

# 1. Ввод данных
# Считываем датасет
df = pd.read_csv('data/dataset_example.csv')

# Задаем алгоритмы и параметры для GridSearchCV
svm_params = [{'kernel': ('linear', 'rbf', 'sigmoid'),
               'C': [0.01, 0.1, 1, 1.5, 5, 10],
               'gamma': ['scale', 'auto'],
               'class_weight': ('balanced', None),
               'random_state': [21],
               'probability': [True]}]
tree_params = {"max_depth": range(1, 50),
               "class_weight": ['balanced', None],
               "criterion": ['entropy', 'gini'],
               "random_state": [21]}
rf_params = {"n_estimators": [5, 10, 50, 100],
             "max_depth": range(1, 50),
             "class_weight": ['balanced', None],
             "criterion": ['entropy', 'gini'],
             "random_state": [21]}

gs_svm = GridSearchCV(estimator=SVC(), param_grid=svm_params,
                      n_jobs=-1, scoring='accuracy', cv=5)
gs_rf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=rf_params,
                     n_jobs=-1, scoring='accuracy', cv=5)
gs_tree = GridSearchCV(estimator=DecisionTreeClassifier(),
                       param_grid=tree_params,
                       n_jobs=-1, scoring='accuracy', cv=5)

grids = [gs_svm, gs_tree, gs_rf]
grid_dict = {0: 'SVM', 1: 'Decision Tree', 2: 'Random Forest'}

# 2. Работа программы
# Предобработка датасета и сплит на подвыборки
preprocessing = Pipeline([
        ("feature_extractor", base_classes.FeatureExtractor()),
        ("onehot_encoder", base_classes.MyOneHotEncoder("dayofweek"))
])
X, y = preprocessing.fit_transform(df)
X_train, X_valid, X_test, y_train, y_valid, y_test = \
    base_classes.TrainValidationTest.div(X, y)

# Поиск наилучшей модели на основе кросс-валидации
selector = base_classes.ModelSelection(grids, grid_dict)
selector.choose(X_train, y_train, X_valid, y_valid)
result = selector.best_results()
print(result)
# Принятие решения пользователем и ввод
# соответствующих данных
model_name = input(
        "На основе полученных результатов, пожалуйста, "
        "выберите модель и введите только название "
        "(без скобок):")
# Переводим название модели из типа str в переменную
strc = model_name
model = locals()[strc]
params = input(
        "Для выбранной модели введите "
        "соответсвующие наилучшие параметры "
        "в виде словаря (скопировав из консоли):")
# Переводим параметры из типа str в словарь
params = ast.literal_eval(params)

# Расчет accuracy на тестовой подвыборке и сохранение модели
estimator = model(**params)
final_result = base_classes.Finalize(estimator)
clf, accuracy = final_result.final_score(X_train, y_train, X_test, y_test)

# можно добавить path (путь для сохранения модели) и задать в качестве
# аргумента при вызове метода save_model(path)
final_result.save_model()
