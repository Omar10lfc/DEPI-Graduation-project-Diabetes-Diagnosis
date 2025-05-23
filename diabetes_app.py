import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

@st.cache_data
def load_data():
    data = pd.read_csv('diabetes_dataset.csv')
    data.drop(labels=[1611, 2550, 2787, 4385, 5064, 7975, 19647, 19658, 22784, 32002, 52722, 58482, 59534, 64571, 69173, 69763, 70701, 70863], axis=0, inplace=True)
    data.drop_duplicates(inplace=True)

    
    df_race = data[["race:AfricanAmerican", "race:Asian", "race:Caucasian", "race:Hispanic", "race:Other"]]
    by_race = pd.get_dummies(df_race) 
    data = data.drop(columns=["race:AfricanAmerican", "race:Asian", "race:Caucasian", "race:Hispanic", "race:Other"])
    data = pd.concat([data, by_race], axis=1) 

    # Process age categories
    bins = [0, 18, 25, 35, 45, 55, 65, np.inf]
    age_order = ["0-18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    data.insert(3, "age_cat", pd.cut(data["age"], bins, labels=age_order))
    
    
    bins = [0, 18.5, 25, 30, 35, 40, np.inf]
    bmi_order = ["Underweight", "Normal", "Overweight", "Moderately Obese", "Severely Obese", "Morbidly Obese"]
    data.insert(9, "bmi_cat", pd.cut(data["bmi"], bins, labels=bmi_order))
    
    data["smoking_history"] = data["smoking_history"].str.title()
    data = data[data["smoking_history"] != "No Info"]

    
    columns = ['gender', 'age_cat', 'location', 'smoking_history', 'bmi_cat']
    for col in columns:
        if col in data.columns:
            data[col] = data[col].astype("category")

    # Rename columns for consistency
    data.columns = [col.title() for col in data.columns]
    data = data.rename(columns={"Age_Cat": "Age Group",
                                "Heart_Disease": "Heart Disease",
                                "Smoking_History": "Smoking History",
                                "Bmi_Cat": "BMI Category",
                                "Bmi": "BMI",
                                "Hba1C_Level": "HbA1C Level",
                                "Blood_Glucose_Level": "Blood Glucose Level"
                                })
    
    # Balance the dataset
    diabetes_yes = data[data["Diabetes"] == 1]
    diabetes_no = data[data["Diabetes"] == 0].sample(n=diabetes_yes.shape[0], ignore_index=True, random_state=42)
    data = pd.concat([diabetes_yes, diabetes_no], axis=0).reset_index(drop=True)

    # Removing the outliers
    columns = ['BMI', 'HbA1C Level']
    for col in columns:
        percentile25 = data[col].quantile(0.25)
        percentile75 = data[col].quantile(0.75)
        iqr = percentile75 - percentile25
        upper_limit = percentile75 + 1.5 * iqr
        lower_limit = percentile25 - 1.5 * iqr
        data[col] = np.clip(data[col], lower_limit, upper_limit)

    return data

data = load_data()

@st.cache_resource
def train_model(data):
    data.drop(columns=['Year', 'Location', 'BMI Category', 'Age Group'], axis=1, inplace=True)
    l = LabelEncoder()
    data['Gender'] = l.fit_transform(data['Gender'])
    data['Smoking History'] = l.fit_transform(data['Smoking History'])
    x = data[['Age', 'Gender', 'Blood Glucose Level', 'Smoking History', 'Hypertension', 'Heart Disease', 'HbA1C Level']]
    y = data['Diabetes']

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42, shuffle=True)

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    model = GradientBoostingClassifier(n_estimators=100)
    model.fit(X_train_scaled, y_train)

    return model, scaler_X

model, scaler_X = train_model(data)

def diagnosis_page():
    st.title("Diabetes Diagnosis")

    age = st.number_input("Age", 1, 80)
    gender = st.selectbox("Gender", ["Male", "Female"])
    blood_glucose = st.number_input("Blood Glucose Level", 0.0, 300.0)
    smoking_history = st.selectbox("Smoking History", ["Never Smoked", "Former Smoker", "Current Smoker"])
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    hba1c_level = st.number_input("HbA1c Level", 0.0, 20.0)

    gender = 1 if gender == "Male" else 0
    hypertension = 1 if hypertension == "Yes" else 0
    heart_disease = 1 if heart_disease == "Yes" else 0
    smoking_map = {"Never Smoked": 0, "Former Smoker": 1, "Current Smoker": 2}
    smoking_history = smoking_map[smoking_history]

    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Blood Glucose Level': [blood_glucose],
        'Smoking History': [smoking_history],
        'Hypertension': [hypertension],
        'Heart Disease': [heart_disease],
        'HbA1C Level': [hba1c_level]
    })

    if st.button("Diagnose"):
        try:
            input_scaled = scaler_X.transform(input_data)
            result = model.predict(input_scaled)
            if result[0] == 1:
                st.error("The patient is likely to be diabetic.")
            else:
                st.success("The patient is not diabetic.")
        except ValueError as e:
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
    page = st.selectbox(" App Navigation", ["Diagnosis", "Dataset History"])
    if page == "Diagnosis":
        diagnosis_page()
    elif page == "Dataset History":
        visualization_page()

if __name__ == "__main__":
    main()
