import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer # т.н. преобразователь колонок
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import root_mean_squared_error
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
# import torch


random_state = 42


def preprocessing_data_frame(frame, process_only: bool = False):
    df = frame.copy()

    # impossible years
    if 394748 in df.index:
        df = df.drop(394748)
    if 1474557 in df.index:
        df = df.drop(1474557)

    df = df.drop(["prev_sold_date", "status", "street", "brokered_by", "zip_code", "city"], axis=1)
    df = df.dropna()

    cat_columns = ['state']
    num_columns = ['price', 'bed', 'acre_lot', 'house_size']

    question = df[(df["price"] < 10_000) | (df["price"] > 1_000_000)]
    df = df.drop(question.index)
    question = df[(df["bed"] > 5)]
    df = df.drop(question.index)
    question = df[(df["bath"] > 5)]
    df = df.drop(question.index)

    df = df.reset_index(drop=True)

    if process_only:
        with open("encoder.pkl", "rb") as file:
            encoder = pickle.load(file)
    else:
        encoder = OneHotEncoder(sparse_output=False)
        encoder.fit(df[cat_columns])
        with open("encoder.pkl", "wb") as file:
            pickle.dump(encoder, file)

    encoded = encoder.transform(df[cat_columns])
    df_encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())
    df = pd.concat([df.drop(columns=['state']), df_encoded], axis=1)
    print(1, df)

    df["acre_lot_log"] = np.log1p(df["acre_lot"])  # log(1 + x) to avoid log(0)
    df[["acre_lot", "acre_lot_log"]].describe()
    df = df.drop("acre_lot", axis=1)
    print(2, df)

    return df


def scale_frame(frame, process_only: bool = False):
    df = frame.copy()

    X, Y = df.drop(columns=['price']), df['price']

    if process_only:
        with open("scaler_x.pkl", "rb") as file:
            scaler_x: StandardScaler = pickle.load(file)
        with open("scaler_y.pkl", "rb") as file:
            scaler_y: StandardScaler = pickle.load(file)

        print(3, X)
        X_scale = scaler_x.transform(X.values)
        print(4, X_scale)
        Y_scale = scaler_y.transform(Y.values.reshape(-1, 1))

    else:
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        X_scale = scaler_x.fit_transform(X.values)
        Y_scale = scaler_y.fit_transform(Y.values.reshape(-1, 1))

        with open("scaler_x.pkl", "wb") as file:
            pickle.dump(scaler_x, file)
        with open("scaler_y.pkl", "wb") as file:
            pickle.dump(scaler_y, file)

    return X_scale, Y_scale, scaler_y

def getYPredicted(y_pred):
    with open("scaler_y.pkl", "rb") as file:
        scaler_y: StandardScaler = pickle.load(file)
    return scaler_y.inverse_transform(y_pred.reshape(-1, 1))


if __name__ == "__main__":
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print('Using device:', device)
    # print()
    #
    # # Additional Info when using cuda
    # if device.type == 'cuda':
    #     # torch.set_default_tensor_type(torch.cuda.FloatTensor)
    #     torch.set_default_tensor_type(torch.cuda.FloatTensor)
    #     torch.set_default_device(device)
    #     print(torch.cuda.get_device_name(0))
    #     print('Memory Usage:')
    #     print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    #     print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

    df = pd.read_csv("realtor-data.csv")
    df.head()


    df_proc = preprocessing_data_frame(df)
    X, Y, scaler_y = scale_frame(df_proc)
    # разбиваем на тестовую и валидационную выборки
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                      test_size=0.01,
                                                      random_state=random_state)


    def SGD():
        return SGDRegressor(
            loss='squared_error',
            max_iter=10000,
            # tol=1e-12,
            # # shuffle=False,
            # eta0=0.001,
            learning_rate='adaptive',
            random_state=random_state
        )

    from sklearn.linear_model import LinearRegression
    def Linear():
        return LinearRegression()

    from sklearn.linear_model import ElasticNet
    def Elastic():
        return ElasticNet()


    model = SGD()
    model.fit(X_train, Y_train.reshape(-1))

    with open("model.pkl", "wb") as file:
        pickle.dump(model, file)