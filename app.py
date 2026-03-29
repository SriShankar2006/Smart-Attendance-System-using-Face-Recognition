import streamlit as st
import subprocess
import sys
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart Attendance System", layout="wide")

st.title("🎓 Smart Attendance System")
st.write("Face Recognition Based Attendance Management")

# Sidebar Navigation
menu = st.sidebar.selectbox(
    "Navigation",
    [
        "Register Face",
        "Student Management",
        "Encode Faces",
        "Train Model",
        "Take Attendance",
        "Analytics Dashboard",
        "Attendance History"
    ]
)

# ===============================
# 📸 REGISTER FACE
# ===============================

if menu == "Register Face":

    st.header("📸 Register Student Face")

    name = st.text_input("Student Name")
    student_id = st.text_input("Student ID")

    picture = st.camera_input("Capture Face")

    if picture and name and student_id:

        file_bytes = np.asarray(bytearray(picture.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        folder = f"dataset/{student_id}_{name}"
        os.makedirs(folder, exist_ok=True)

        count = len(os.listdir(folder)) + 1
        path = f"{folder}/{count}.jpg"

        cv2.imwrite(path, img)

        st.success("Face saved successfully")

# ===============================
# 👥 STUDENT MANAGEMENT
# ===============================

elif menu == "Student Management":

    st.header("👥 Registered Students")

    dataset = "dataset"

    if os.path.exists(dataset):

        students = os.listdir(dataset)

        if students:
            for student in students:

                col1, col2 = st.columns([3,1])

                col1.write(student)

                if col2.button("Delete", key=student):

                    import shutil
                    shutil.rmtree(os.path.join(dataset, student))

                    st.warning(f"{student} deleted")
                    st.experimental_rerun()

        else:
            st.info("No students registered")

# ===============================
# ⚙ ENCODE FACES
# ===============================

elif menu == "Encode Faces":

    st.header("⚙ Encode Faces")

    if st.button("Start Encoding"):
        subprocess.run([sys.executable, "module2_encode_faces.py"])
        st.success("Encoding completed")

# ===============================
# 🧠 TRAIN MODEL
# ===============================

elif menu == "Train Model":

    st.header("🧠 Train Recognition Model")

    if st.button("Train Model"):
        subprocess.run([sys.executable, "module3_train_model.py"])
        st.success("Training completed")

    charts = [
        "training_results.png",
        "confusion_matrix.png",
        "model_accuracy.png"
    ]

    for chart in charts:
        if os.path.exists(chart):
            st.image(chart)

# ===============================
# 🎥 TAKE ATTENDANCE
# ===============================

elif menu == "Take Attendance":

    st.header("🎥 Live Attendance")

    st.info("Camera window will open")

    if st.button("Start Attendance"):
        subprocess.run([sys.executable, "module4_take_attendance.py"])

# ===============================
# 📊 ANALYTICS DASHBOARD
# ===============================

elif menu == "Analytics Dashboard":

    st.header("📊 Attendance Analytics")

    if st.button("Generate Analytics"):
        subprocess.run([sys.executable, "module5_visualization.py"])

    charts = [
        "reports/1_daily_attendance.png",
        "reports/2_student_percentage.png",
        "reports/3_pie_latest_day.png",
        "reports/4_heatmap.png",
        "reports/5_dashboard.png"
    ]

    for chart in charts:
        if os.path.exists(chart):
            st.image(chart, width=800)

# ===============================
# 📅 ATTENDANCE HISTORY
# ===============================

elif menu == "Attendance History":

    st.header("📅 Attendance Records")

    file = "attendance.csv"

    if os.path.exists(file):

        df = pd.read_csv(file)

        st.write("Columns detected:", df.columns)

        st.dataframe(df)

        # Automatically detect name column
        possible_columns = ["Name", "Student", "StudentName"]

        name_column = None
        for col in possible_columns:
            if col in df.columns:
                name_column = col
                break

        if name_column:

            student = st.selectbox(
                "Filter by student",
                ["All"] + list(df[name_column].unique())
            )

            if student != "All":
                st.dataframe(df[df[name_column] == student])

        else:
            st.warning("No student name column found in CSV.")

    else:
        st.warning("No attendance file found")