import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Title
st.title("🎓 Student Performance Prediction Dashboard")

# Load dataset
data = pd.read_csv("student_performance_dataset.csv")
data.columns = data.columns.str.strip()
st.write(data.columns)


# Remove extra spaces from column names
data.columns = data.columns.str.strip()

# Show dataset preview
st.subheader("Dataset Preview")
st.dataframe(data.head())

# Select useful columns
X = data[['Study_Hours_per_Week','Attendance_Rate','Past_Exam_Scores']]
y = data['Final_Exam_Score']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Accuracy
score = r2_score(y_test, model.predict(X_test))
st.subheader("Model Accuracy")
st.write(round(score,2))

# User input
st.subheader("Enter Student Details")

hours = st.slider("Hours Per Week", 0, 50, 10)
attendance = st.slider("Attendance Rate", 0, 100, 75)
past_score = st.slider("Past Exam Score", 0, 100, 60)

if st.button("Predict Final Exam Score"):
    
    prediction = model.predict([[hours, attendance, past_score]])
    st.success(f"Predicted Final Exam Score: {round(prediction[0],2)}")

# Graph
st.subheader("Hours per Week vs Final Exam Score")

fig, ax = plt.subplots()
ax.plot(data['Study_Hours_per_Week'], data['Final_Exam_Score'])
ax.set_xlabel("Hours Per Week")
ax.set_ylabel("Final Exam Score")
st.pyplot(fig)
