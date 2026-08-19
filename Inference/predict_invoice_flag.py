import joblib
import pandas as pd

MODEL_PATH = r"C:\Users\ADITYA\Flipkart_2\Invoice Flagging\models\predict_flag_invoice.pkl"
SCALER_PATH = r"C:\Users\ADITYA\Flipkart_2\Invoice Flagging\models\scaler.pkl"

# Load once
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


def predict_invoice_flag(input_data):
    input_df = pd.DataFrame(input_data)

    input_scaled = scaler.transform(input_df)

    input_df["Predict_Flag"] = model.predict(input_scaled)

    return input_df


if __name__ == "__main__":

    sample_data = {
        "invoice_quantity": [50],
        "invoice_dollars": [352.95],
        "Freight": [1.73],
        "total_item_quantity": [162],
        "total_item_dollars": [2476.0]
    }

    prediction = predict_invoice_flag(sample_data)
    print(prediction)