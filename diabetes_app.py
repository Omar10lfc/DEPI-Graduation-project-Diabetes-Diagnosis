import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from joblib import load
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


ARTIFACT_MODEL_PATH = os.path.join("artifacts", "best_model.joblib")
ARTIFACT_METADATA_PATH = os.path.join("artifacts", "model_metadata.json")


@st.cache_data
def load_data():
    data = pd.read_csv("diabetes_dataset.csv")
    data.drop(
        labels=[
            1611,
            2550,
            2787,
            4385,
            5064,
            7975,
            19647,
            19658,
            22784,
            32002,
            52722,
            58482,
            59534,
            64571,
            69173,
            69763,
            70701,
            70863,
        ],
        axis=0,
        inplace=True,
    )
    data.drop_duplicates(inplace=True)

    df_race = data[["race:AfricanAmerican", "race:Asian", "race:Caucasian", "race:Hispanic", "race:Other"]]
    by_race = pd.from_dummies(df_race)
    data = data.drop(columns=["race:AfricanAmerican", "race:Asian", "race:Caucasian", "race:Hispanic", "race:Other"])
    data.insert(2, "race", by_race)
    data["race"] = data["race"].str.replace("race:", "", regex=False)
    data["race"] = data["race"].str.replace("AfricanAmerican", "African-American", regex=False)

    bins = [0, 18, 25, 35, 45, 55, 65, np.inf]
    age_order = ["0-18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    data.insert(3, "age_cat", pd.cut(data["age"], bins, labels=age_order))

    bins = [0, 18.5, 25, 30, 35, 40, np.inf]
    bmi_order = ["Underweight", "Normal", "Overweight", "Moderately Obese", "Severely Obese", "Morbidly Obese"]
    data.insert(9, "bmi_cat", pd.cut(data["bmi"], bins, labels=bmi_order))

    data["smoking_history"] = data["smoking_history"].str.title()
    data = data[data["smoking_history"] != "No Info"]

    for col in ["gender", "race", "age_cat", "location", "smoking_history", "bmi_cat"]:
        if col in data.columns:
            data[col] = data[col].astype("category")

    columns = list(data.columns)
    new_names = [col.title() for col in columns]
    data = data.rename(columns=dict(zip(columns, new_names)))
    data = data.rename(
        columns={
            "Age_Cat": "Age Group",
            "Heart_Disease": "Heart Disease",
            "Smoking_History": "Smoking History",
            "Bmi_Cat": "BMI Category",
            "Bmi": "BMI",
            "Hba1C_Level": "HbA1C Level",
            "Blood_Glucose_Level": "Blood Glucose Level",
            "Race": "Race",
        }
    )

    # Keep balancing behavior aligned with notebook app experiments.
    diabetes_yes = data[data["Diabetes"] == 1]
    diabetes_no = data[data["Diabetes"] == 0].sample(n=diabetes_yes.shape[0], ignore_index=True, random_state=42)
    data = pd.concat([diabetes_yes, diabetes_no], axis=0).reset_index(drop=True)

    for col in ["BMI", "HbA1C Level"]:
        percentile25 = data[col].quantile(0.25)
        percentile75 = data[col].quantile(0.75)
        iqr = percentile75 - percentile25
        upper_limit = percentile75 + 1.5 * iqr
        lower_limit = percentile25 - 1.5 * iqr
        data[col] = np.clip(data[col], lower_limit, upper_limit)

    return data


def prepare_model_dataframe(raw_df):
    model_df = raw_df.copy()
    drop_cols = ["Year", "Location", "BMI Category", "Age Group", "Race"]
    model_df = model_df.drop(columns=[c for c in drop_cols if c in model_df.columns])

    encoders = {}
    for col in ["Gender", "Smoking History"]:
        if col in model_df.columns:
            le = LabelEncoder()
            model_df[col] = le.fit_transform(model_df[col])
            encoders[col] = le

    return model_df, encoders


@st.cache_resource
def load_prediction_artifacts(_model_df):
    feature_columns = [col for col in _model_df.columns if col != "Diabetes"]
    metadata = {}

    if os.path.exists(ARTIFACT_MODEL_PATH):
        model = load(ARTIFACT_MODEL_PATH)
        if os.path.exists(ARTIFACT_METADATA_PATH):
            with open(ARTIFACT_METADATA_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                feature_columns = metadata.get("feature_columns", feature_columns)
        return model, feature_columns, metadata, True

    # Fallback if notebook artifact has not been generated yet.
    x = _model_df[feature_columns]
    y = _model_df["Diabetes"]
    X_train, _, y_train, _ = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42, shuffle=True)
    model = GradientBoostingClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model, feature_columns, metadata, False


data = load_data()
model_df, label_encoders = prepare_model_dataframe(data)
model, feature_columns, model_metadata, using_artifact = load_prediction_artifacts(model_df)


def build_input_dataframe(feature_cols, reference_df, encoders):
    input_payload = {}
    for col in feature_cols:
        if col == "Gender" and "Gender" in encoders:
            label = st.selectbox("Gender", list(encoders["Gender"].classes_))
            input_payload[col] = int(encoders["Gender"].transform([label])[0])
        elif col == "Smoking History" and "Smoking History" in encoders:
            label = st.selectbox("Smoking History", list(encoders["Smoking History"].classes_))
            input_payload[col] = int(encoders["Smoking History"].transform([label])[0])
        elif col in ["Hypertension", "Heart Disease"]:
            label = st.selectbox(col, ["No", "Yes"])
            input_payload[col] = 1 if label == "Yes" else 0
        else:
            if col in reference_df.columns and pd.api.types.is_numeric_dtype(reference_df[col]):
                col_min = float(reference_df[col].min())
                col_max = float(reference_df[col].max())
                default = float(reference_df[col].median())
                step = 1.0 if col in ["Age", "Blood Glucose Level"] else 0.1
                input_payload[col] = st.number_input(col, min_value=col_min, max_value=col_max, value=default, step=step)
            else:
                input_payload[col] = st.number_input(col, value=0.0)

    return pd.DataFrame([input_payload], columns=feature_cols)


def diagnosis_page():
    st.title("Diabetes Diagnosis")

    if using_artifact:
        selected = model_metadata.get("best_model", "saved model")
        st.caption(f"Using saved notebook model artifact: {selected}")
    else:
        st.warning("Saved artifact not found. Using fallback model trained in app.")

    input_data = build_input_dataframe(feature_columns, model_df, label_encoders)

    if st.button("Diagnose"):
        try:
            result = model.predict(input_data)[0]
            proba = None
            if hasattr(model, "predict_proba"):
                proba = float(model.predict_proba(input_data)[0, 1])

            if result == 1:
                st.error("The patient is likely to be diabetic.")
            else:
                st.success("The patient is not diabetic.")

            if proba is not None:
                st.info(f"Predicted probability (Diabetes=1): {proba:.3f}")
        except Exception as e:
            st.error(f"Error in processing the input: {e}")

def visualization_page():
    st.title("Diabetes Data History Visualizations")

    st.write("Diabetes Trend Analysis:")
    pst_cases = data[data['Diabetes'] == 1]
    trends = pst_cases.groupby('Year')[['Blood Glucose Level', 'Age', 'HbA1C Level', 'BMI']].mean()
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    sns.lineplot(data=trends, x=trends.index, y='Age', ax=axes[0, 0])
    axes[0, 0].set_title("Average Age of Diabetic Cases Over the Years")
    sns.lineplot(data=trends, x=trends.index, y='Blood Glucose Level', ax=axes[0, 1])
    axes[0, 1].set_title("Average Blood Glucose Level of Diabetic Cases Over the Years")
    sns.lineplot(data=trends, x=trends.index, y='HbA1C Level', ax=axes[1, 0])
    axes[1, 0].set_title("Average HbA1c Level of Diabetic Cases Over the Years")
    sns.lineplot(data=trends, x=trends.index, y='BMI', ax=axes[1, 1])
    axes[1, 1].set_title("Average BMI of Diabetic Cases Over the Years")
    plt.tight_layout()
    st.pyplot(fig)

    
    st.write("Blood Glucose Level Distribution:")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(data[data['Diabetes'] == 0]['Blood Glucose Level'], label='Non-diabetic', color='skyblue', bins=5, ax=ax)
    sns.histplot(data[data['Diabetes'] == 1]['Blood Glucose Level'], label='Diabetic', color='salmon', bins=5, ax=ax)
    ax.set_title('Blood Glucose Level Distribution')
    ax.set_xlabel('Blood Glucose Level')
    ax.set_ylabel('Frequency')
    ax.legend()
    st.pyplot(fig)

    st.write("BMI Distribution:")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(data[data['Diabetes'] == 0]['BMI'], label='Non-diabetic', color='skyblue', bins=20, ax=ax)
    sns.histplot(data[data['Diabetes'] == 1]['BMI'], label='Diabetic', color='salmon', bins=20, ax=ax)
    ax.set_title('BMI Distribution')
    ax.set_xlabel('BMI')
    ax.set_ylabel('Frequency')
    ax.legend()
    st.pyplot(fig)

    st.write("HbA1c Level vs BMI:")
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.scatterplot(x='BMI', y='HbA1C Level', hue='Diabetes', data=data, palette='coolwarm', ax=ax)
    ax.set_title('HbA1c Level vs BMI by Diabetes Status')
    ax.set_xlabel('BMI')
    ax.set_ylabel('HbA1c Level')
    st.pyplot(fig)

    st.write("Count of Diabetic and Non-diabetic Patients by Year:")
    yearly_diabetes = data.groupby(['Year', 'Diabetes']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 6))
    yearly_diabetes.plot(kind='bar', stacked=False, color=['skyblue', 'salmon'], width=0.8, ax=ax)
    ax.set_title('Count of Diabetic and Non-diabetic Patients by Year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Count')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.legend(title='Diabetes', labels=['Non-diabetic', 'Diabetic'])
    ax.grid(axis='y')
    st.pyplot(fig)

    st.write("Diabetes Across Age Groups:")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Age Group', hue='Diabetes', data=data, palette='coolwarm', ax=ax)
    ax.set_title('Diabetes Prevalence Across Age Groups')
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Count')
    ax.legend(title='Diabetes', labels=['Non-diabetic', 'Diabetic'])
    st.pyplot(fig)

    st.write("Trend of Diabetic and Non-diabetic People Over Time:")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(yearly_diabetes.index, yearly_diabetes[0], marker='o', linestyle='-', color='skyblue', label='Non-diabetic')
    ax.plot(yearly_diabetes.index, yearly_diabetes[1], marker='o', linestyle='-', color='salmon', label='Diabetic')
    ax.set_title('Trend of Diabetic and Non-diabetic People Over Time', fontsize=16)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.grid(True)
    ax.legend(title='Diabetes Status')
    plt.tight_layout()
    st.pyplot(fig)


    st.write("Age Against Diabetes:")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Diabetes', y='Age', data=data, palette='coolwarm', hue='Diabetes', ax=ax)
    ax.set_title('Age vs Diabetes')
    ax.set_xlabel('Diabetes (0: Non-diabetic, 1: Diabetic)')
    ax.set_ylabel('Age')
    st.pyplot(fig)



def main():
    page = st.selectbox("App Navigation", ["Diagnosis", "Dataset History"])
    if page == "Diagnosis":
        diagnosis_page()
    elif page == "Dataset History":
        visualization_page()

if __name__ == "__main__":
    main()
