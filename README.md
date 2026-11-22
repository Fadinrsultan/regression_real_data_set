
# 🚗 Used Car Price Prediction Using Machine Learning

### Regression Analysis, Model Comparison, and Hyperparameter Tuning on a Real-World Automotive Dataset

This project explores **machine learning techniques for predicting used car prices** using a real-world dataset containing numerical, categorical, and boolean vehicle attributes.

The project combines **Exploratory Data Analysis (EDA)**, data preprocessing, regression modeling, classification, hyperparameter selection, and model evaluation.

Four regression algorithms are compared:

1. Linear Regression
2. K-Nearest Neighbors (KNN) Regression
3. Decision Tree Regression
4. Random Forest Regression

The project also investigates **Logistic Regression** to predict whether a vehicle listing allows exchanges.

The experiments are implemented using **Python, Pandas, NumPy, Scikit-learn, TensorFlow, Matplotlib, and Seaborn**.

**Main finding:** Random Forest achieved the lowest recorded Mean Squared Error (MSE), approximately **3.83 million**, among the four regression approaches in the final experiment.

## 🚀 Run in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fadinrsultan/regression_real_data_set/blob/main/Regression_of_complex_(20%2B_features)_and_diverse_real_cars_dataset.ipynb)

**GitHub Repository:**

https://github.com/Fadinrsultan/regression_real_data_set

**Notebook:**

`Regression_of_complex_(20+_features)_and_diverse_real_cars_dataset.ipynb`

## 🎯 Project Objectives

The objectives of this project are to:

- Explore a real-world dataset with more than 20 attributes.
- Understand relationships between vehicle characteristics and selling prices.
- Perform exploratory data analysis and visualization.
- Handle missing numerical and categorical data.
- Encode categorical features.
- Normalize numerical variables.
- Implement several regression algorithms.
- Evaluate models using MSE, RMSE, and R².
- Compare different hyperparameter configurations.
- Identify the best-performing regression approach in the recorded experiment.
- Investigate binary classification using Logistic Regression.
- Explore classification thresholds and performance metrics.

## ✨ Key Features

- **Real-World Dataset:** 38,531 car listings with 30 columns.
- **Mixed Feature Types:** Numerical, categorical, and boolean attributes.
- **Price Prediction:** Estimates vehicle selling prices in USD.
- **Exploratory Data Analysis:** Histograms, scatter plots, correlation matrices, and box plots.
- **Multiple Regression Models:** Linear Regression, KNN, Decision Tree, and Random Forest.
- **TensorFlow Regression:** A simple neural-network implementation of linear regression.
- **Data Preprocessing:** Missing-value imputation, feature scaling, and categorical encoding.
- **Hyperparameter Exploration:** Tests alternative KNN and Decision Tree configurations.
- **Model Comparison:** Evaluates regression performance using Mean Squared Error.
- **Logistic Regression:** Predicts the `is_exchangeable` binary variable.
- **Classification Evaluation:** Includes accuracy, confusion matrix, precision, recall, and F1-score.

## 🧠 Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| Pandas | Data analysis and manipulation |
| NumPy | Numerical computing |
| Scikit-learn | Regression, classification, and preprocessing |
| TensorFlow / Keras | Neural-network-based regression |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| LinearRegression | Linear price prediction |
| KNeighborsRegressor | Distance-based regression |
| DecisionTreeRegressor | Tree-based regression |
| RandomForestRegressor | Ensemble regression |
| LogisticRegression | Binary classification |
| StandardScaler | Numerical feature scaling |
| SimpleImputer | Missing-value handling |
| OneHotEncoder | Categorical feature encoding |
| ColumnTransformer | Feature-specific preprocessing |
| Google Colab | Cloud-based notebook execution |

## 📊 Dataset Description

The project uses a used-car listings dataset containing:

| Property | Value |
|----------|-------|
| Records | 38,531 |
| Columns | 30 |
| Regression Target | `price_usd` |
| Classification Target | `is_exchangeable` |
| Data Types | Numerical, categorical, boolean |
| Price Currency | USD |

### Example Attributes

| Feature | Description |
|---------|-------------|
| `manufacturer_name` | Vehicle manufacturer |
| `model_name` | Vehicle model |
| `transmission` | Transmission type |
| `color` | Vehicle color |
| `odometer_value` | Recorded mileage |
| `year_produced` | Production year |
| `engine_fuel` | Fuel type |
| `engine_has_gas` | Gas-equipment indicator |
| `engine_type` | Engine type |
| `engine_capacity` | Engine displacement |
| `price_usd` | Listed vehicle price |
| `has_warranty` | Warranty availability |
| `is_exchangeable` | Exchange availability |
| `number_of_photos` | Number of listing photographs |
| `duration_listed` | Listing duration |

The dataset also includes several additional vehicle and listing attributes.

### Dataset Summary

Selected descriptive statistics from the notebook:

| Variable | Mean | Median |
|----------|------|--------|
| Price (USD) | $6,639.97 | $4,800 |
| Odometer | 248,864.64 | 250,000 |
| Production Year | 2002.94 | 2003 |
| Engine Capacity | 2.06 | 2.00 |
| Number of Photos | 9.65 | 8 |
| Listing Duration | 80.58 | 59 |

The dataset contains substantial variation in vehicle prices and characteristics.

### Dataset Access

The notebook references a CSV file named:

`cars.csv`

Dataset link provided in the notebook:

https://drive.google.com/file/d/1GQyK6FfTLUkAxXBJoIbu1n3Jq-EIBYiC/view

## 🏗️ Project Architecture

```text
                  Used Car Dataset
                         |
                         v
                   Load CSV Data
                         |
                         v
             Exploratory Data Analysis
                         |
           +-------------+-------------+
           |                           |
           v                           v
      Data Cleaning              Visualization
           |                           |
           v                           v
    Missing Values              Histograms
    Data Encoding               Scatter Plots
    Feature Scaling             Correlation
           |                    Box Plots
           v
      Train/Test Split
           |
           v
    Regression Algorithms
           |
    +------+------+------+------+
    |      |      |             |
    v      v      v             v
  Linear  KNN   Decision      Random
Regression       Tree         Forest
    |      |      |             |
    +------+------+-------------+
                  |
                  v
          Model Evaluation
                  |
                  v
           MSE Comparison
                  |
                  v
          Best Recorded Model
                  |
                  v
             Random Forest
```

The notebook also includes a separate binary-classification experiment.

## 🔍 Exploratory Data Analysis

The project examines the dataset through descriptive statistics and several visualization techniques.

### 1. Histograms

Histograms are used to explore distributions of numerical attributes such as:

- Vehicle prices
- Engine capacity
- Odometer values
- Production years
- Listing duration

### 2. Scatter Plots

Scatter plots visualize relationships between numerical attributes and the target variable `price_usd`.

An early experiment investigates the relationship between engine capacity and selling price.

### 3. Correlation Matrix

A correlation heatmap examines linear relationships among numerical features.

This helps identify variables that are associated with vehicle prices.

### 4. Box Plots

Box plots are used to explore data distributions and identify potential outliers.

These visualizations help characterize the dataset before applying machine learning algorithms.

## 🔄 Data Preprocessing

The final regression experiment prepares numerical and categorical features separately.

### Numerical Features

Numerical features are processed through:

- Mean imputation for missing values
- StandardScaler for feature normalization

### Categorical Features

Categorical variables are processed through:

- Most-frequent-value imputation
- One-hot encoding

The `OneHotEncoder` is configured to handle previously unseen categories.

### Scikit-learn Implementation

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)
```

The final experiment uses an 80/20 training-test split.

| Dataset | Records |
|---------|---------|
| Training | 30,824 |
| Testing | 7,707 |

The predictor matrix contains 29 original columns before preprocessing.

## 📐 Assignment 1: Linear Regression

The first experiment investigates the relationship between engine capacity and used-car prices.

### Objective

Predict vehicle price using a single numerical predictor:

`engine_capacity`

### Mathematical Model

Linear regression estimates the target using:

\[
\hat{y}=\beta_0+\beta_1x
\]

Where:

- `y` represents vehicle price.
- `x` represents engine capacity.
- `β0` represents the intercept.
- `β1` represents the regression coefficient.

### Scikit-learn Implementation

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

### Recorded Regression Equation

The notebook reports:

\[
Price(USD)=847.015+2810.485\times EngineCapacity
\]

### Recorded Results

| Metric | Result |
|--------|--------|
| RMSE | 6,169.94 |
| R² | 0.0971 |
| Intercept | 847.015 |
| Slope | 2,810.485 |

The low R² indicates that engine capacity alone explains only a limited proportion of the variation in listed vehicle prices.

### TensorFlow Implementation

The notebook also implements linear regression using a single Dense layer in TensorFlow.

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1])
])

model.compile(
    optimizer=Adam(learning_rate=0.1),
    loss="mean_squared_error"
)

model.fit(
    X_train,
    y_train,
    epochs=50,
    validation_split=0.2
)
```

The recorded TensorFlow test loss is approximately:

**38,291,712 MSE**

This demonstrates an alternative optimization-based implementation of linear regression.

### Seaborn Visualization

The notebook uses Seaborn's `lmplot` to visualize the linear relationship between engine capacity and price.

## 🔬 Assignment 2: K-Nearest Neighbors Regression

The second assignment investigates KNN regression using multiple vehicle characteristics.

### Input Features

- `odometer_value`
- `year_produced`
- `engine_capacity`

### Algorithm

KNN regression predicts a vehicle's price based on the prices of nearby training examples in feature space.

The initial implementation uses:

```python
KNeighborsRegressor(n_neighbors=5)
```

### Recorded Result

**RMSE: 2,744.69 USD**

This result is lower than the RMSE reported for the earlier single-feature linear regression experiment.

However, the KNN model uses additional predictors, so the comparison does not isolate the effect of the algorithm itself.

## 🧪 Binary Classification Using Logistic Regression

The notebook also implements Logistic Regression for predicting whether a vehicle listing is exchangeable.

### Classification Target

`is_exchangeable`

Possible values:

- `True`: The listing allows exchanges.
- `False`: The listing does not allow exchanges.

### Input Features

- Odometer value
- Production year
- Engine capacity

### Balanced Logistic Regression

The notebook uses class weighting to address the unequal distribution of the target classes.

```python
from sklearn.linear_model import LogisticRegression

logistic_model_balanced = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

logistic_model_balanced.fit(
    X_train_logistic,
    y_train_logistic
)
```

### Recorded Classification Results

| Metric | Result |
|--------|--------|
| Accuracy | 53.04% |
| Precision — True Class | 0.38 |
| Recall — True Class | 0.51 |
| F1-score — True Class | 0.44 |
| Test Samples | 7,705 |

### Confusion Matrix

| Actual / Predicted | False | True |
|--------------------|-------|------|
| False | 2,678 | 2,278 |
| True | 1,340 | 1,409 |

### Classification Thresholds

The notebook investigates classification thresholds to maximize recall and F1-score.

The recorded threshold sweep reports:

| Metric | Recorded Result |
|--------|-----------------|
| Best Recall | 1.000 |
| Best Recall Threshold | 0.00 |
| Best F1-score | 0.5259 |
| Best F1 Threshold | 0.00 |

A threshold of zero classifies all observations as positive. Therefore, the perfect recall result should not be interpreted as strong classification performance.

The threshold experiment illustrates the trade-offs between recall and precision.

## 🏆 Final Assignment: Comparing Four Regression Models

The final part of the project uses the available vehicle attributes to predict `price_usd`.

The evaluated algorithms are:

- Linear Regression
- K-Nearest Neighbors Regression
- Decision Tree Regression
- Random Forest Regression

### 1. Linear Regression

Linear Regression provides a baseline model using the preprocessed predictors.

**Recorded MSE:**

6,897,078.44

### 2. K-Nearest Neighbors Regression

The experiment evaluates several combinations of neighbor counts and weighting strategies.

Tested neighbor counts:

`3, 5, 7`

Tested weights:

`uniform, distance`

**Selected configuration:**

```python
KNeighborsRegressor(
    n_neighbors=7,
    weights="distance"
)
```

**Recorded MSE:**

6,709,090.07

### 3. Decision Tree Regression

The Decision Tree experiment evaluates alternative tree configurations using:

- `max_depth`
- `min_samples_split`
- `min_samples_leaf`

A selected configuration is:

```python
DecisionTreeRegressor(
    max_depth=30,
    min_samples_split=10,
    min_samples_leaf=4
)
```

The hyperparameter-selection cell reports an MSE of approximately **5,670,866.88** for this configuration.

The notebook's later model-comparison dictionary records **5,665,327.51** for Decision Tree Regression. Both outputs are retained here because the saved notebook contains this small discrepancy.

### 4. Random Forest Regression

Random Forest combines multiple decision trees to generate ensemble predictions.

The final experiment uses:

```python
from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
```

**Recorded MSE:**

3,833,573.52

## 📈 Model Comparison Results

The final model comparison dictionary reports:

| Regression Model | MSE |
|------------------|--------------:|
| Linear Regression | 6,897,078.44 |
| KNN Regression | 6,709,090.07 |
| Decision Tree Regression | 5,665,327.51 |
| **Random Forest Regression** | **3,833,573.52** |

### Best Recorded Model

**Random Forest Regression**

The lowest MSE reported in the final comparison is:

\[
MSE=3,833,573.52
\]

The corresponding RMSE is approximately:

\[
RMSE=\sqrt{MSE}\approx1,957.95\ USD
\]

### Results Interpretation

Random Forest achieved the lowest recorded prediction error among the evaluated regression methods.

This suggests that ensemble-based tree models captured useful nonlinear relationships among vehicle attributes in the selected dataset.

However, these results should be considered experimental because some hyperparameters were chosen by comparing performance directly on the test set.

A separate validation set or cross-validation would provide a more reliable basis for model selection.

## 🔁 Testing with a Different Data Split

The final assignment investigates whether model performance changes when using a different random train/test split.

The original split uses:

```python
random_state=42
```

The alternative split uses:

```python
random_state=52
```

Both use:

```python
test_size=0.2
```

The notebook records the new dataset sizes and begins a Random Forest re-evaluation.

However, the saved output does not include a completed second MSE result.

A complete comparison would require applying the same preprocessing pipeline to the new split and evaluating the trained model again.

## 📊 Performance Evaluation Metrics

### Mean Squared Error (MSE)

\[
MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
\]

MSE measures the average squared prediction error.

Lower values indicate better predictions.

### Root Mean Squared Error (RMSE)

\[
RMSE=\sqrt{MSE}
\]

RMSE expresses prediction error in the same unit as the target variable.

For this project, RMSE is measured in USD.

### Coefficient of Determination (R²)

\[
R^2=1-\frac{SS_{res}}{SS_{tot}}
\]

R² measures how well predictions explain variation in the target variable relative to a constant-mean baseline.

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Fadinrsultan/regression_real_data_set.git
cd regression_real_data_set
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

**Windows:**

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow notebook
```

### 4. Prepare the Dataset

Download the `cars.csv` file from the dataset link.

The notebook originally reads the dataset from Google Drive:

```python
df = pd.read_csv(
    "/content/drive/MyDrive/cars.csv"
)
```

For local execution, change the path to:

```python
df = pd.read_csv("cars.csv")
```

### 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the project notebook and run the cells sequentially.

### 6. Run in Google Colab

The notebook can also be opened directly in Google Colab using the badge at the beginning of this README.

## 🧪 Experimental Workflow

```text
1. Load the dataset
        |
        v
2. Explore dataset statistics
        |
        v
3. Visualize feature distributions
        |
        v
4. Handle missing values
        |
        v
5. Encode categorical variables
        |
        v
6. Scale numerical features
        |
        v
7. Split training and test data
        |
        v
8. Train regression models
        |
        v
9. Evaluate prediction errors
        |
        v
10. Compare model performance
        |
        v
11. Identify best recorded model
        |
        v
12. Investigate alternative data splits
```

## ⚠️ Limitations

The project has several limitations:

- The dataset contains historical car listings.
- Vehicle asking prices may differ from actual sale prices.
- The initial linear and KNN experiments use different feature sets.
- Some hyperparameters are selected using the test data.
- A dedicated validation strategy is needed for unbiased model selection.
- Some numerical features contain missing values.
- Extreme prices and outliers may influence squared-error metrics.
- Model performance may change with different train/test splits.
- The notebook includes some repeated code from earlier assignments.
- The Decision Tree outputs contain a small difference between reported MSE values.
- The alternative Random Forest split does not have a completed saved evaluation output.
- The results should not be treated as a verified production pricing system.

## 🔮 Future Improvements

Possible enhancements include:

- Cross-validation for robust model selection
- GridSearchCV or RandomizedSearchCV
- Full preprocessing and prediction pipelines
- Gradient Boosting Regression
- XGBoost, LightGBM, or CatBoost comparisons
- Feature importance analysis
- SHAP explanations
- Residual distribution analysis
- Outlier detection
- Log-transformed price prediction
- Additional evaluation using MAE and R²
- Separate validation and held-out test datasets
- Repeated experiments using different random seeds
- Model serialization
- Interactive car price prediction interface
- Deployment as a web application

## 🌐 Potential Applications

The methods investigated in this project are relevant to:

- Used-car price estimation
- Automotive market analysis
- Vehicle valuation research
- Regression benchmarking
- Predictive analytics
- Tabular machine learning
- Automated pricing systems
- Data-driven decision support

This implementation is primarily an educational machine learning project.

## 📚 References

### Machine Learning

**Géron, A.**

*Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow.*

https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/

### Scikit-learn Documentation

- [Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [KNN Regression](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
- [Decision Tree Regression](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
- [Random Forest Regression](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
- [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)

### Data Analysis and Visualization

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Documentation](https://matplotlib.org/stable/)
- [Seaborn Documentation](https://seaborn.pydata.org/)

### TensorFlow

- [TensorFlow Documentation](https://www.tensorflow.org/)

## 👨‍💻 Author

**Fadi Sultan**

AI & Machine Learning Developer

**GitHub Repository:**

https://github.com/Fadinrsultan/regression_real_data_set

---

⭐ If you find this project useful, consider giving the repository a star.
