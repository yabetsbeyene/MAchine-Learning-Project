 Project Overview
The Salary Prediction System is a machine learning project that predicts employee annual salaries based on 10 professional and demographic features. It uses a Random Forest Regressor trained on 10,000 synthetic employee records, achieving an R² score of 0.8581 (85.8% accuracy).
The system includes:

A Jupyter notebook for data analysis, training, and evaluation
Three serialized model files (.pkl)
An interactive Streamlit web application for real-time predictions


 Project Structure
salary_project/
│
├── salary_prediction.ipynb       # Training notebook (EDA + models + evaluation)
├── app-1.py                      # Streamlit web application
├── generate_model.py             # Standalone script to regenerate .pkl files
├── salary_dataset.csv            # 10,000-row synthetic dataset
│
├── salary_prediction_model.pkl   # Trained Random Forest model
├── salary_scaler.pkl             # Fitted StandardScaler
└── salary_label_encoders.pkl     # LabelEncoders for categorical features

 Machine Learning Pipeline
Dataset (10,000 rows)
       ↓
Preprocessing & Encoding
       ↓
Train / Test Split (80/20)
       ↓
Model Training
  ├── Linear Regression    → R²: 0.78
  ├── Gradient Boosting    → R²: 0.84
  └── Random Forest      → R²: 0.8581  ← Best Model
       ↓
Save .pkl Files
       ↓
Streamlit App → Real-time Predictions

📊 Dataset Features
FeatureTypeValuesAgeNumeric22 – 55GenderCategoricalMale, FemaleEducationCategoricalHigh School, Bachelor's, Master's, PhDJob RoleCategoricalIntern, Analyst, Engineer, Manager, DirectorDepartmentCategoricalIT, Finance, HR, Marketing, OperationsExperienceNumeric0 – 30 yearsJob LevelCategoricalEntry, Mid, SeniorRemote WorkCategoricalYes, NoCertificationsNumeric0 – 10Performance ScoreNumeric1 – 10Salary (Target)Numeric$20,000 – $200,000

📈 Model Results
ModelR² ScoreMAERMSELinear Regression0.7800$9,840$10,950Gradient Boosting0.8400$7,120$9,380Random Forest ✅0.8581$6,595$8,852
Top Features by Importance
Job Role          ████████████████████████  0.26
Experience        ████████████████████      0.18
Education         ████████████████          0.16
Job Level         ████████████              0.12
Age               ███████                   0.07
Certifications    ██████                    0.06
Department        █████                     0.05
Performance Score ████                      0.04
Remote Work       ███                       0.03
Gender            ███                       0.03

 Installation & Setup
1. Clone the Repository
bashgit clone https://github.com/your-username/salary-prediction.git
cd salary-prediction
2. Install Dependencies
bashpip install pandas numpy scikit-learn matplotlib seaborn joblib streamlit notebook
3. Generate the Model (run this first)
bashpython generate_model.py
This creates the 3 .pkl files needed by the app.
4. Run the Web App
bashpython -m streamlit run app-1.py
Open your browser at: http://localhost:8501

 Running the Notebook
bashpython -m jupyter notebook salary_prediction.ipynb
Or open it directly in VS Code and click Run All.

 Running the notebook will regenerate the .pkl files. Make sure to run it before launching the app if you skip generate_model.py.


 App Features
Once the Streamlit app is running you can:

Fill in employee details using dropdowns and sliders
Click 🔍 Predict Salary to get an instant prediction
View results as:

 Entry-Level Band (< $40,000)
 Mid-Level Band (< $80,000)
 Senior-Level Band (< $130,000)
 Executive Band (> $130,000)


See salary broken down into Annual / Monthly / Weekly / Daily
View a salary progress bar and key factors table


 How the .pkl Files Work
FileContentsPurposesalary_prediction_model.pklTrained Random ForestMakes predictionssalary_scaler.pklFitted StandardScalerNormalizes numeric inputsalary_label_encoders.pkl6 LabelEncoder objectsEncodes categorical input

The app uses hardcoded encoding maps to guarantee the same encoding as training — preventing the most common ML deployment bug.


 Tech Stack
ToolPurposePython 3.10+Core languagePandas
NumPyData handlingScikit-learnML models & preprocessingMatplotlib 
SeabornVisualizationsJoblibModel serializationStreamlitWeb applicationJupyter NotebookTraining & EDAVS CodeDevelopment IDE

 Quick Demo
python# Sample prediction (same encoding as app)
import pandas as pd, joblib

model = joblib.load("salary_prediction_model.pkl")

input_data = pd.DataFrame([{
    "Age": 28,               # 28 years old
    "Gender": 1,             # Male
    "Education": 2,          # Master's
    "Job_Role": 2,           # Engineer
    "Department": 2,         # IT
    "Experience": 5,         # 5 years
    "Job_Level": 1,          # Mid
    "Remote_Work": 1,        # Yes
    "Certifications": 3,
    "Performance_Score": 8
}])

salary = model.predict(input_data)[0]
print(f"Predicted Salary: ${salary:,.0f}")
# Output: Predicted Salary: $82,400
