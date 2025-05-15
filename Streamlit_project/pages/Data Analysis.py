import streamlit as st

st.title("📊 Data Analysis")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import plotly.express as px
import altair as alt

# 1. Load dataset using pandas
url ="https://raw.githubusercontent.com/Yuvaraja1604/Streamlit-project-ML-/refs/heads/main/Students%20Social%20Media%20Addiction%201.csv"
df = pd.read_csv(url)
chart_data = df[['Age', 'Avg_Daily_Usage_Hours']].set_index('Avg_Daily_Usage_Hours')
line_data=df[['Academic_Level','Avg_Daily_Usage_Hours']].set_index('Academic_Level')
# Display the bar chart
st.title("Age vs Avg_Daily_Usage_Hours")
st.bar_chart(chart_data)
st.title("Academic_Level vs Avg_Daily_Usage_Hours")
st.bar_chart(line_data)
# Create a pie chart: e.g., distribution of social media platform usage
st.title("Age Usage Distribution")
fig = px.pie(df, names='Age')

# Display the pie chart
st.plotly_chart(fig)
st.title("Most_Used_Platform vs Avg_Daily_Usage_Hours ")
l=df[['Most_Used_Platform','Avg_Daily_Usage_Hours']].set_index('Most_Used_Platform')
st.bar_chart(l)
chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["Academic_Level", "Most_Used_Platform", "Avg_Daily_Usage_Hours"])
st.line_chart(chart_data)

st.title("Scatter Plot of Age vs. Daily Usage")
scatter = alt.Chart(df).mark_circle(size=80).encode(
    x='Avg_Daily_Usage_Hours:Q',
    y='Age:O',
      # Optional: color by gender
    tooltip=['Age', 'Avg_Daily_Usage_Hours']
).interactive()

# Show in Streamlit
st.altair_chart(scatter, use_container_width=True)