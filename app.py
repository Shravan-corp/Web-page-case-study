import streamlit as st
import pandas as pd
import joblib

# Load model, scaler and column names
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

continent_cols = [
    "continent_Asia",
    "continent_Europe",
    "continent_North America",
    "continent_Oceania",
    "continent_South America",
]


def preprocess_input(data, scaler, continent_columns):
    df = pd.DataFrame([data])
    for col in continent_columns:
        df[col] = 0
    continent_col = f"continent_{data['continent']}"
    if continent_col in continent_columns:
        df[continent_col] = 1
    df.drop("continent", axis=1, inplace=True)
    df["beer_spirit_interaction"] = df["beer_servings"] * df["spirit_servings"]
    df["total_alcohol_servings"] = (
        df["beer_servings"] + df["spirit_servings"] + df["wine_servings"]
    )
    num_cols = [
        "beer_servings",
        "spirit_servings",
        "wine_servings",
        "beer_spirit_interaction",
        "total_alcohol_servings",
    ]
    df[num_cols] = scaler.transform(df[num_cols])
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df


st.title("🍺 Alcohol Consumption Predictor")
st.markdown("### Predict Total Litres of Pure Alcohol Consumed per Person")

continent = st.selectbox(
    "Continent", ["Asia", "Europe", "North America", "Oceania", "South America"]
)
beer_servings = st.slider("Beer Servings", 0, 500, 100)
spirit_servings = st.slider("Spirit Servings", 0, 500, 50)
wine_servings = st.slider("Wine Servings", 0, 500, 30)

input_data = {
    "beer_servings": beer_servings,
    "spirit_servings": spirit_servings,
    "wine_servings": wine_servings,
    "continent": continent,
}

if st.button("Predict"):
    processed_input = preprocess_input(input_data, scaler, continent_cols)
    prediction = model.predict(processed_input)[0]
    st.success(f"Predicted total litres of pure alcohol: **{prediction:.2f} L/person**")

st.markdown("### 📊 Data Insights")
st.image("static/images/histograms.png", caption="Distributions")
st.image("static/images/scatter_matrix.png", caption="Scatter Matrix")
st.image("static/images/alcohol_by_continent.png", caption="Alcohol by Continent")
