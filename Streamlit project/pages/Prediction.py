import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load and preprocess dataset
url = "https://raw.githubusercontent.com/Yuvaraja1604/Streamlit-project-ML-/refs/heads/main/Students%20Social%20Media%20Addiction%201.csv"
df = pd.read_csv(url)
df.drop(['Student_ID', 'Country'], axis=1, inplace=True)

# Encode categorical columns
df['Affects_Academic_Performance'] = df['Affects_Academic_Performance'].map({'Yes': 1, 'No': 0})
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Academic_Level'] = df['Academic_Level'].astype('category')
df['Academic_Level'] = df['Academic_Level'].cat.codes
df['Most_Used_Platform'] = df['Most_Used_Platform'].astype('category')
platform_mapping = dict(enumerate(df['Most_Used_Platform'].cat.categories))
df['Most_Used_Platform'] = df['Most_Used_Platform'].cat.codes

# Train model
X = df.drop('Affects_Academic_Performance', axis=1).values
y = df['Affects_Academic_Performance'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Streamlit App
st.title("Student Social Media Impact Prediction")

with st.form("feedback_form"):
    name = st.text_input("Name *")
    age = st.number_input("Age *", min_value=1)
    Gen = st.selectbox("Gender *", ["", "Male", "Female"])
    Aca = st.selectbox("Education *", ["", "Graduate", "High School", "Undergraduate"])
    platform_input = st.selectbox("Preferred Social Media Platform *", ["", *platform_mapping.values()])
    Avg = st.number_input("Avg_Daily_Usage_Hours (Mobile) *", min_value=0.0)

    submit = st.form_submit_button("Submit")

    if submit:
        # Validation
        if not name or Gen == "" or Aca == "" or platform_input == "" or age == 0 or Avg == 0:
            st.warning("Please fill in all required fields marked with *.")
        else:
            # Encode inputs
            st.success("Form submitted successfully!")
            st.write("### Submitted Data:")
            st.write(f"**Name:** {name}")
            st.write(f"**Age:** {age}")
            st.write(f"**Gender:** {Gen}")
            st.write(f"**Education:** {Aca}")
            st.write(f"**Platform:** {platform_input}")
            st.write(f"**Avg_Daily_Usage_Hours(Mobile):** {Avg}")

            def gender(s):
                return 1 if s == "Male" else 0

            def education(s):
                return {"Graduate": 0, "High School": 1, "Undergraduate": 2}[s]

            def get_platform_code(s):
                for code, label in platform_mapping.items():
                    if s == label:
                        return code
                return -1  # fallback

            g = gender(Gen)
            a = education(Aca)
            p = get_platform_code(platform_input)

            new_sample = np.array([[age, g, a, Avg, p]])
            new_pred = clf.predict(new_sample)
            st.write("### Prediction Result:")
            st.write("Academic Performance Affected:" if new_pred[0] == 1 else "No Impact on Academic Performance")
