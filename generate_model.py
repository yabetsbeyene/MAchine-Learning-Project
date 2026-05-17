import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

print("Starting model generation...")

# ── Generate Dataset ───────────────────────────────────────────────────────
np.random.seed(42)
n = 10000

education_levels = ['High School', "Bachelor's", "Master's", 'PhD']
job_roles        = ['Analyst', 'Engineer', 'Manager', 'Director', 'Intern']
departments      = ['IT', 'Finance', 'HR', 'Marketing', 'Operations']
genders          = ['Male', 'Female']
job_levels       = ['Entry', 'Mid', 'Senior']
remote_options   = ['Yes', 'No']

education   = np.random.choice(education_levels, n, p=[0.15, 0.45, 0.30, 0.10])
job_role    = np.random.choice(job_roles,        n, p=[0.25, 0.30, 0.20, 0.10, 0.15])
department  = np.random.choice(departments,      n)
gender      = np.random.choice(genders,          n)
job_level   = np.random.choice(job_levels,       n, p=[0.35, 0.40, 0.25])
remote_work = np.random.choice(remote_options,   n, p=[0.40, 0.60])
experience  = np.random.randint(0, 30, n)
age         = 22 + experience + np.random.randint(0, 5, n)
certs       = np.random.randint(0, 10, n)
perf_score  = np.random.randint(1, 11, n)

edu_map   = {'High School': 0, "Bachelor's": 1, "Master's": 2, 'PhD': 3}
role_map  = {'Intern': 0, 'Analyst': 1, 'Engineer': 2, 'Manager': 3, 'Director': 4}
level_map = {'Entry': 0, 'Mid': 1, 'Senior': 2}

base_salary = (
    30000
    + np.array([edu_map[e]   for e in education])  * 8000
    + np.array([role_map[r]  for r in job_role])   * 12000
    + np.array([level_map[l] for l in job_level])  * 7000
    + experience   * 1500
    + certs        * 1200
    + perf_score   * 800
    + np.random.normal(0, 5000, n)
)
salary = np.clip(base_salary, 20000, 200000).astype(int)

df = pd.DataFrame({
    'Age':               age,
    'Gender':            gender,
    'Education':         education,
    'Job_Role':          job_role,
    'Department':        department,
    'Experience':        experience,
    'Job_Level':         job_level,
    'Remote_Work':       remote_work,
    'Certifications':    certs,
    'Performance_Score': perf_score,
    'Salary':            salary
})

print(f"Dataset created: {df.shape}")

# ── Preprocessing ──────────────────────────────────────────────────────────
df['Age'].fillna(df['Age'].median(),              inplace=True)
df['Experience'].fillna(df['Experience'].median(), inplace=True)

label_encoders   = {}
categorical_cols = ['Gender', 'Education', 'Job_Role', 'Department', 'Job_Level', 'Remote_Work']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

print("Encoding done...")

# ── Train Model ────────────────────────────────────────────────────────────
X = df.drop('Salary', axis=1)
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

print("Model trained successfully!")
print(f"Features used: {list(X.columns)}")

# ── Save .pkl Files ────────────────────────────────────────────────────────
joblib.dump(model,          'salary_prediction_model.pkl')
joblib.dump(scaler,         'salary_scaler.pkl')
joblib.dump(label_encoders, 'salary_label_encoders.pkl')

print("")
print("✅ All 3 files saved successfully!")
print("   salary_prediction_model.pkl")
print("   salary_scaler.pkl")
print("   salary_label_encoders.pkl")
print("")
print("Now run: python -m streamlit run app-1.py")
