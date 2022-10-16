import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import (train_test_split)
from sklearn.base import (BaseEstimator,
                          TransformerMixin)
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Предобработка датасета
    """
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, df):
        return self

    def transform(self, df):
        # меняем формат столбца "timestamp"
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        # извлекаем день недели и час коммита
        df["dayofweek"] = df["timestamp"].dt.weekday
        df["hour"] = df["timestamp"].dt.hour
        # удаляем столбец, ставший не нужным
        df = df.drop(columns='timestamp')

        return df


class MyOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    Данный класс определяет категориальные признаки и
    выполняет OneHotEncoding для них
    """

    def __init__(self, key):
        self.key = key

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        y = df[self.key]
        # определяем столбцы с категориальными признаками
        categorical_feature_mask = df.dtypes == object
        categorical_features = df.columns[categorical_feature_mask].tolist()
        try:
            categorical_features = categorical_features.remove(self.key)
        except ValueError:
            pass
        # Fit OneHotEncoder to X, then transform X
        encoder = OneHotEncoder()
        # get_feature_names_out([input_features])get output
        # feature names for transformation
        df1 = pd.DataFrame(encoder.fit_transform(
            df[categorical_features]).todense(),
                           columns=encoder.get_feature_names_out())
        df = df.join(df1)
        # удаляем столбцы, ставшие не нужными
        df = df.drop(labels=categorical_features, axis=1)
        df = df.drop(labels=self.key, axis=1)

        return df, y


class TrainValidationTest:
    """
    Данный класс выполняет разделение на подвыборки:
    тренировочную, валидационную и тестовую
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def div(X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=21, stratify=y)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, y_train, test_size=0.2, random_state=21,
            stratify=y_train)

        return X_train, X_valid, X_test, y_train, y_valid, y_test


class ModelSelection:
    """
    Данный класс выполняет работу по выбору модели:
    на основе кросс-валидации ищет наилучшее сочетание
    гиперпараметров для каждой заданной модели
    и выводит наилучший результат + соответствующие
    гиперпараметры для каждой модели
    """
    # задаем lists для формирования датафрейма через статические атрибуты
    model_lst = []
    params_lst = []
    valid_score_lst = []

    def __init__(self, grids, grid_dict):
        self.grids = grids
        self.grid_dict = grid_dict

    def choose(self, X_train, y_train, X_valid, y_valid):
        max_valid_score = 0
        best_model = " "
        for i in range(len(self.grids)):

            grid = self.grids[i]
            grid.fit(X_train, y_train)
            # выводим на печать результаты Gridsearch
            print(f'Estimator: {self.grid_dict[i]}')
            print(f'Best params: {grid.best_params_}')
            print(f'Best training accuracy: {round(grid.best_score_, 3)}')
            print(f'Validation set accuracy score for best params:'
                  f' {round((grid.score(X_valid, y_valid)), 3)}')
            print('  ')  # чтобы была пустая строка между моделями

            # формируем lists для датафрейма в следующем методе
            ModelSelection.model_lst.append(self.grid_dict[i])
            ModelSelection.params_lst.append(grid.best_params_)
            ModelSelection.valid_score_lst.append(grid.score(X_valid, y_valid))

            # определяем наилучшую модель на основе кросс-валидации
            if max_valid_score < grid.score(X_valid, y_valid):
                max_valid_score = grid.score(X_valid, y_valid)
                best_model = self.grid_dict[i]
            else:
                max_valid_score = max_valid_score

        print(f'Classifier with best validation set accuracy: {best_model}')

        return best_model

    def best_results(self):
        # формируем датафрейм с результатами valid_score для каждой модели
        df0 = pd.DataFrame(ModelSelection.model_lst, columns=['model'])
        df1 = pd.DataFrame({'params': ModelSelection.params_lst})
        df2 = pd.DataFrame(
            ModelSelection.valid_score_lst, columns=['valid_score'])
        df_result = pd.concat([df0, df1, df2], axis=1)

        return df_result


class Finalize:
    """
    Данный класс финализирует работу:
    рассчитывает accuracy для выбранной модели на
    тестовой подвыборке и сохраняет модель
    """

    def __init__(self, estimator):
        self.estimator = estimator
        self.accuracy = 0

    def final_score(self, X_train, y_train, X_test, y_test):

        self.estimator.fit(X_train, y_train)

        # расчет accuracy  на тестовой подвыборке
        y_pred = self.estimator.predict(X_test)
        self.accuracy = round(accuracy_score(y_test, y_pred), 4)

        print(f'Accuracy of the final model on test is {self.accuracy}')
        return self.estimator, self.accuracy

    def save_model(self, path=''):

        # сохранение финальной модели:
        # по умолчанию модель будет сохранена в текущей папке
        # либо можно указать path
        name = f"best_model_accuracy_{self.accuracy}.sav"
        path_name = Path(path, name)

        joblib.dump(self.estimator, path_name)
        print(f"best_model_accuracy_{self.accuracy}.sav - model saved!")
        return name
