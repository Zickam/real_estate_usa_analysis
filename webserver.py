from fastapi import FastAPI

from pandas import DataFrame
from dataclasses import dataclass, asdict
import pickle

from sklearn.linear_model import SGDRegressor
from clean_and_train import preprocessing_data_frame, scale_frame, getYPredicted


@dataclass()
class Data:
    brokered_by: str
    status: str
    price: float
    bed: int
    bath: int
    acre_lot: float
    street: int
    city: str
    state: str
    zip_code: int
    house_size: float
    prev_sold_date: str

app = FastAPI()
with open("model.pkl", "rb") as file:
    model: SGDRegressor = pickle.load(file)

@app.get("/predict")
def read_root(kwargs: Data):
    df = DataFrame(data=[asdict(kwargs)])
    print(df.head())
    print(df.shape)

    df_proc = preprocessing_data_frame(df, True)
    print(df_proc.head())
    X, Y, scaler_y = scale_frame(df_proc, True)
    print(X)
    print(Y)

    pred = model.predict(X)
    pred_norm = scaler_y.inverse_transform(pred.reshape(-1, 1))
    return {"message": {"pred_norm": str(float(pred_norm)), "pred": str(pred)}}

if __name__ == "__main__":
    import subprocess

    with open("uvicorn.log", "w") as log:
        process = subprocess.Popen(
            ["uvicorn", "webserver:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "2"],
            stdout=log,
            stderr=log,
            text=True
        )

    print(f"Uvicorn started with PID {process.pid}")