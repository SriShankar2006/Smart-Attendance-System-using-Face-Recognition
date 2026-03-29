# 🎓 Smart Attendance System using Face Recognition
A Python-based automated attendance system that registers student faces, encodes facial features, trains machine learning models, performs real-time face recognition for attendance, and generates analytical reports and visualizations.

The system captures student faces, converts them into numerical encodings, trains recognition models, detects students via webcam, and generates attendance reports and visualizations.

# 🚀 Features

✅ Face Registration using Webcam
✅ Automatic Face Encoding (128-D facial descriptors)
✅ Machine Learning Model Training (KNN & SVM)
✅ Real-time Face Recognition Attendance
✅ CSV Attendance Storage
✅ Attendance Analytics and Visual Reports
✅ Menu-driven Command Line Interface
✅ Dashboard & Heatmap Visualization

# 🧠 System Workflow

# 1️⃣ Face Registration
Capture student face images via webcam
Store images in structured dataset folders
Save student records in students.json
# 2️⃣ Face Encoding
Convert face images into 128-dimensional embeddings
Uses face_recognition (dlib)
Stores encodings in encodings.pkl
# 3️⃣ Model Training
Train KNN and SVM classifiers
Compare model accuracy
Save best model to trained_model.pkl
# 4️⃣ Live Attendance
Detect faces in real time using webcam
Match faces with trained encodings
Mark attendance automatically in attendance.csv
# 5️⃣ Reports & Visualization
Daily attendance charts
Student attendance percentage
Attendance heatmap
Pie charts
Dashboard summary
Text attendance report

All reports are saved inside the reports/ folder.

# 📂 Project Structure
Smart-Attendance-System
│
├── main.py
├── module1_register_face.py
├── module2_encode_faces.py
├── module3_train_model.py
├── module4_take_attendance.py
├── module5_visualization.py
│
├── dataset/
├── reports/
│
├── students.json
├── encodings.pkl
├── trained_model.pkl
├── attendance.csv
🖥️ Main Menu Interface

Run:

python main.py

Menu options:

1  Register Student Faces
2  Encode Faces
3  Train Recognition Model
4  Take Attendance
5  View Reports & Visualizations
0  Exit

The menu automatically checks required files before running modules.

⚙️ Requirements
Python Version
Python 3.8+
Download cmake: https://cmake.org/download/
📦 Required Python Libraries

Install dependencies:

pip install opencv-python
pip install face-recognition
pip install numpy
pip install matplotlib
pip install scikit-learn
pip install pandas
pip install seaborn
pip install cmake
pip install dlib
pip install imutils

Or install everything at once:

pip install opencv-python face-recognition numpy matplotlib scikit-learn pandas seaborn cmake dlib imutils
# 🧩 External Dependencies
Some libraries require system packages:
Windows
Install Visual Studio Build Tools if dlib fails to install.
Linux / Mac
Install:
cmake
gcc
build-essential

# The system generates:
📊 Daily Attendance Chart
📉 Student Attendance Percentage
🧭 Attendance Heatmap
🥧 Present vs Absent Pie Chart
📋 Attendance Dashboard
📄 Text Attendance Report
🔐 Recognition Settings

# 🧑‍💻 Technologies Used
Python
OpenCV
face_recognition
dlib
scikit-learn
NumPy
Pandas
Matplotlib
Seaborn
# 🎯 Applications
Classroom attendance automation
College attendance systems
Smart classroom projects
Computer vision learning projects

# 📚 Learning Outcomes
This project demonstrates:
Computer Vision
Face Recognition
Machine Learning
Data Visualization
Python Project Architecture

# 🏁 Future Improvements
Possible upgrades:
GUI using Tkinter / PyQt
Database storage (MySQL / MongoDB)
Cloud deployment
Mobile integration
Multi-camera support
