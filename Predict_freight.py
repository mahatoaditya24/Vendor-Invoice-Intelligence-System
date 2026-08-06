import joblib
import pandas as pd

MODEL_PATH = r"C:\Users\ADITYA\Flipkart_2\Freight Cost Prediction\models\predict_freight_model.pkl"


def load_model(model_path=MODEL_PATH):
    return joblib.load(model_path)


def predict_freight_cost(input_data):
    model = load_model()

    input_df = pd.DataFrame(input_data)

    input_df["Predict_Freight"] = model.predict(input_df).round(2)

    return input_df


if __name__ == "__main__":

    sample_data = {
        "Quantity": [100, 50, 10, 25, 80],
        "Dollars": [18500, 9000, 300, 2500, 7000]
    }

    prediction = predict_freight_cost(sample_data)
    print(prediction)