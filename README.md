# Network Performance Prediction using Regression

A machine learning regression project that predicts network performance metrics using supervised learning techniques, achieving reliable predictive performance.  
This project was developed as part of a university Machine Learning course.

---

## 📋 Overview

This project demonstrates the use of regression models to predict network performance based on structured numerical data.  
The workflow covers data preprocessing, feature scaling, model training, and evaluation using standard regression metrics.

---

## 🎯 Objective

The primary objective of this project is to develop a regression model capable of accurately predicting network performance indicators from input features.  
The project focuses on understanding regression pipelines and evaluating model effectiveness.

---

## 📊 Dataset

### Data Source

The dataset `network performance2.csv` contains network-related records with multiple numerical features used to predict a continuous target variable representing network performance.

### Dataset Characteristics

- Type: Structured numerical data  
- Target: Network performance metric  

---

## 🔧 Requirements

```txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

### Installation
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

Or install all dependencies at once:
```bash
pip install -r requirements.txt
```

## 🚀 Methodology

### 1. Data Loading and Exploration
```python
import pandas as pd
data = pd.read_csv("network performance2.csv")
data.info()
data.describe()
```

Initial exploration was performed to understand feature distributions, data types, and detect missing values.

### 2. Data Preprocessing
#### Handling missing values
#### Feature selection
#### Feature scaling using StandardScaler to normalize numerical features

### 3. Data Splitting

The dataset was split into training and testing sets:

- **Training set**: 80%
- **Testing set**: 20%

This ensures proper evaluation of model generalization.

### 4. Model Training
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```

A regression model was trained to learn the relationship between the input features and the target variable.

## 📈 Results

### Model Performance

| Metric | Score |
|--------|-------|
|**R² Score** |	0.8813 |
|**Mean Squared Error** (MSE)	| 0.1358 |
|**Root Mean Squared Error** (RMSE) | 0.1850 |

**Analysis**
- The model shows strong predictive performance on unseen data
- Feature scaling improved training stability
- Regression techniques effectively captured relationships within the dataset

## 💻 Usage

### Running the Project

1. Clone the repository:
```bash
git clone https://github.com/Saif13m05/Network-Performance-Regression.git
cd Network-Performance-Regression
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Open the Jupyter notebook:
```bash
jupyter notebook regression_final.ipynb
```

4. Run all cells sequentially to train and evaluate the model.

### Making Predictions

To predict heart disease for a new patient:
```python
# Predict new samples
prediction = model.predict(new_data)
print(prediction)
```

## 📁 Project Structure
```
Heart-Disease-Classification-KNN/
│
├── regression_final.ipynb    # Main notebook with complete workflow
├── network performance2.csv       # Dataset file
├── README.md                      # Project documentation
└── requirements.txt               # Python dependencies
```

## 🔍 Key Insights

1. Data preprocessing significantly impacts regression performance
2. Feature scaling enhances model stability and accuracy
3. Regression models are effective for predicting continuous network metrics

## 🎓 Future Improvements

- Try advanced regression models (Ridge, Lasso, Random Forest Regressor)
- Apply hyperparameter tuning
- Perform feature importance analysis
- Use cross-validation for better generalization

## ⚠️ Disclaimer

This project is for educational purposes only and should not be used in production systems without further validation.

## 📧 Contact

For questions or suggestions, feel free to open an [issue](https://github.com/Saif13m05/Network-Performance-Regression/issues).

---

**Built with ❤️ using Machine Learning**




