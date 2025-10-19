# 🕵️‍♂️ Crime Data Analysis Project

## 📖 Overview
This project analyzes crime data to uncover trends, relationships, and predictive insights between offenders and victims.
It uses a real-world dataset (crime_data.csv) containing demographic, status, and categorical information related to crime incidents.

The project performs:
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA) with visualizations
- Predictive modeling (Random Forest)
- Feature importance analysis
- Optional Streamlit dashboard for interactive exploration

------------------------------------------------------------

## 📂 Dataset
File: crime_data.csv
Columns:
- Disposition
- OffenderStatus
- Offender_Race
- Offender_Gender
- Offender_Age
- PersonType
- Victim_Race
- Victim_Gender
- Victim_Age
- Victim_Fatal_Status (target variable)
- Report Type
- Category

Each row represents a reported crime event, including offender and victim details.

------------------------------------------------------------

## ⚙️ Technologies Used
- Python 3.8+
- pandas — data handling
- numpy — numerical computations
- matplotlib / seaborn — visualization
- scikit-learn — machine learning
- joblib — model saving
- Streamlit (optional) — interactive dashboard

------------------------------------------------------------

## 🧠 Project Workflow

### 1️⃣ Data Loading & Cleaning
- Read and inspect crime_data.csv
- Standardize column names and handle missing values
- Convert age and categorical columns into usable formats

### 2️⃣ Exploratory Data Analysis (EDA)
- Visualize target variable distribution
- Analyze offender and victim demographics
- Study relationships between race, gender, and crime outcomes
- Display correlation heatmaps and frequency charts

### 3️⃣ Feature Engineering
- Select relevant predictors such as:
  Offender_Age, Victim_Age, Offender_Race, Victim_Race,
  Offender_Gender, Victim_Gender, Report Type, Category, Disposition
- Encode categorical variables with OneHotEncoder

### 4️⃣ Predictive Modeling
- Train a Random Forest Classifier to predict Victim_Fatal_Status
- Split dataset (80/20) for train-test evaluation
- Evaluate with accuracy, ROC-AUC, confusion matrix
- Analyze feature importances

### 5️⃣ Results
- Insights on which factors most influence fatal outcomes
- Graphs and plots stored in /mnt/data/crime_model_output/

### 6️⃣ Model Saving
- Trained pipeline saved as crime_victim_fatal_model.pkl
- Feature importances saved as feature_importances.csv

------------------------------------------------------------

## 💻 How to Run

In VS Code or Terminal:
1. Clone or download this repository.
2. Place your crime_data.csv file in the project directory.
3. Install required libraries:
   pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit
4. Run the main script:
   python Crime.py

Optional Streamlit Dashboard:
Uncomment the Streamlit section at the bottom of Crime.py, then run:
   streamlit run Crime.py

------------------------------------------------------------

## 📊 Example Insights
- Majority of incidents involve offenders aged 20–35.
- Fatal outcomes are more common in violent categories.
- Offender and victim race often correlate within incidents.
- Certain report types and categories predict higher fatality likelihood.

------------------------------------------------------------

## 📁 Outputs
All generated files and plots are saved in:
/mnt/data/crime_model_output/

Contents include:
- feature_importances.csv
- crime_victim_fatal_model.pkl
- Saved visualizations (PNG files)

------------------------------------------------------------

## 📈 Results & Key Findings

=== Evaluation on test set ===

Accuracy: 0.9984939759036144

ROC AUC: 0.9845804988662131

Classification report:
              precision    recall  f1-score   support

           0     0.9985    1.0000    0.9992      1323
           1     1.0000    0.6000    0.7500         5

    accuracy                         0.9985      1328
   macro avg     0.9992    0.8000    0.8746      1328
weighted avg     0.9985    0.9985    0.9983      1328

----------------------------------------------------------------------
| Classification results | Precision | Recall  | f1-score  | support |
|------------------------|-----------|---------|-----------|---------|
| 0 | 0.9985 | 1.0000 | 0.9992 | 1323 |
| 1 | 1.0000 | 0.6000 | 0.7500 | 5 |
----------------------------------------------------------------------
| accuracy | - | - | 0.9985 | 1328 |
| macro average | 0.9992 | 0.8000 | 0.8746 | 1328 |
| weighted average | 0.9985 | 0.9985 | 0.9983 | 1328 |
----------------------------------------------------------------------

------------------------------------------------------------

## 📌 Next Steps
- Experiment with gradient boosting models (XGBoost, LightGBM).
- Deploy dashboard publicly using Streamlit Cloud or Hugging Face Spaces.

------------------------------------------------------------

## 👨‍💻 Author
Triston Marta
Data Science & Statistics
Interested in data analysis, visualization, and predictive modelling.
