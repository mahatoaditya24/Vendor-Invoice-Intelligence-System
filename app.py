import streamlit as st
from Predict_freight import predict_freight_cost
from predict_invoice_flag import predict_invoice_flag

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Vendor Invoice Intelligence Portal",
    page_icon="📦",
    layout="wide"
)

# -------------------------------
# Header
# -------------------------------
st.title("📦 Vendor Invoice Intelligence Portal")
st.subheader("AI-Driven Freight Cost Prediction & Invoice Risk Flagging")

st.markdown("""
This internal analytics portal leverages machine learning to:

- 🚛 Forecast freight costs accurately
- 🚨 Detect risky vendor invoices
- 💰 Reduce financial leakage
- ⚡ Improve finance operations
""")

st.divider()

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("🔍 Model Selection")

selected_model = st.sidebar.radio(
    "Choose Prediction Module",
    (
        "Freight Cost Prediction",
        "Invoice Manual Approval Flag"
    )
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### Business Impact
- 📈 Better Cost Forecasting
- 🧾 Fraud Detection
- ⚙️ Faster Finance Operations
""")

# ===========================================================
# Freight Prediction
# ===========================================================
if selected_model == "Freight Cost Prediction":

    st.header("🚛 Freight Cost Prediction")

    with st.form("freight_form"):

        col1, col2 = st.columns(2)

        with col1:
            quantity = st.number_input(
                "Quantity",
                min_value=1,
                value=1200
            )

        with col2:
            dollars = st.number_input(
                "Invoice Dollars",
                min_value=1.0,
                value=18500.0
            )

        submit = st.form_submit_button("Predict Freight Cost")

    if submit:

        input_data = {
            "Quantity": [quantity],
            "Dollars": [dollars]
        }

        result = predict_freight_cost(input_data)

        freight = result["Predict_Freight"][0]

        st.success("Prediction Successful!")

        st.metric(
            "Estimated Freight Cost",
            f"${freight:,.2f}"
        )

# ===========================================================
# Invoice Flag Prediction
# ===========================================================
else:

    st.header("🚨 Invoice Manual Approval Prediction")

    with st.form("invoice_form"):

        col1, col2, col3 = st.columns(3)

        with col1:

            invoice_quantity = st.number_input(
                "Invoice Quantity",
                min_value=1,
                value=50
            )

            freight = st.number_input(
                "Freight",
                min_value=0.0,
                value=1.73
            )

        with col2:

            invoice_dollars = st.number_input(
                "Invoice Dollars",
                min_value=1.0,
                value=352.95
            )

            total_item_quantity = st.number_input(
                "Total Item Quantity",
                min_value=1,
                value=162
            )

        with col3:

            total_item_dollars = st.number_input(
                "Total Item Dollars",
                min_value=1.0,
                value=2476.0
            )

        submit = st.form_submit_button("Evaluate Invoice")

    if submit:

        input_data = {
            "invoice_quantity": [invoice_quantity],
            "invoice_dollars": [invoice_dollars],
            "Freight": [freight],
            "total_item_quantity": [total_item_quantity],
            "total_item_dollars": [total_item_dollars]
        }

        result = predict_invoice_flag(input_data)

        flag = int(result["Predict_Flag"][0])

        if flag == 1:
            st.error("🚨 Invoice requires Manual Approval")
        else:
            st.success("✅ Invoice Approved Automatically")

        st.subheader("Prediction Details")
        st.dataframe(result, use_container_width=True)