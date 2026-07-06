# 🚢 Titanic Survival Prediction

A machine learning project that predicts passenger survival on the Titanic using classic data cleaning, exploratory data analysis (EDA), and classification models.

## 📌 Overview

This project uses the famous [Titanic dataset](https://www.kaggle.com/c/titanic) to build a model that predicts whether a passenger survived the disaster based on features like age, sex, class, and fare. It walks through the full ML workflow — from raw data to a trained, evaluated model.

## 🗂️ Dataset

The dataset (`Titanic Dataset.csv`) contains 891 passenger records with the following original features:

| Column | Description |
|---|---|
| `PassengerId` | Unique identifier |
| `Survived` | Target variable (0 = No, 1 = Yes) |
| `Pclass` | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) |
| `Name` | Passenger name |
| `Sex` | Gender |
| `Age` | Age in years |
| `SibSp` | # of siblings/spouses aboard |
| `Parch` | # of parents/children aboard |
| `Ticket` | Ticket number |
| `Fare` | Passenger fare |
| `Cabin` | Cabin number |
| `Embarked` | Port of embarkation |

## 🧹 Data Preprocessing

- Dropped uninformative/high-cardinality columns: `PassengerId`, `Name`, `Ticket`, `Cabin`
- Capped `Age` outliers above 60 years
- Imputed missing `Age` values with the column mean
- Encoded `Pclass` and `Sex` into numerical/categorical formats
- One-hot encoded remaining categorical features (`Pclass`, `Embarked`)

## 📊 Exploratory Data Analysis

- Visualized `Age` distribution before/after cleaning (boxplot, histogram)
- Examined survival counts grouped by `Sex`
- Generated a correlation heatmap to understand feature relationships with `Survived`

## 🤖 Models Trained

Two classification models were trained on an 80/20 train-test split:

| Model | Accuracy |
|---|---|
| Logistic Regression | ~81.6% |
| Random Forest Classifier | ~82.7% |

Model evaluation includes accuracy score, confusion matrix, and classification report.

## 🛠️ Tech Stack

- **Python**
- **Pandas** & **NumPy** – data manipulation
- **Matplotlib** & **Seaborn** – visualization
- **Scikit-learn** – modeling & evaluation

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Run the notebook
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
jupyter notebook TITANIC.ipynb
```

> **Note:** Update the dataset path in the notebook (`/content/Titanic Dataset.csv`) to point to your local file location.

## 📁 Project Structure

```
├── TITANIC.ipynb          # Main analysis and modeling notebook
├── Titanic Dataset.csv    # Dataset (add your own copy)
└── README.md              # Project documentation
```

## 📈 Future Improvements

- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- Feature engineering (e.g., title extraction from names, family size)
- Cross-validation for more robust performance estimates
- Try additional models (XGBoost, SVM, Gradient Boosting)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
