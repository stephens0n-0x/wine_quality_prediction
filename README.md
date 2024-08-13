# Wine Quality Prediction

This repository contains a machine learning project that predicts wine quality based on various features like price, variety, and description. The model utilizes a Support Vector Regression (SVR) approach with different preprocessing techniques, including scaling, one-hot encoding, and TF-IDF for text features.

## Dataset

The dataset used in this project is available on Kaggle: [Wine Reviews](https://www.kaggle.com/zynicide/wine-reviews).

## Project Overview

The project is divided into several key steps:

1. **Data Loading and Splitting:**
   - The dataset is loaded into a pandas DataFrame, and 10% of the data is used for training (`sample_fraction = 0.1`). A smaller fraction was chosen to expedite model training and testing, while ensuring enough data was used to maintain predictive accuracy. This approach helps make the process more efficient by reducing computation time and resource usage.

2. **Data Preprocessing:**
   - **Numerical Features:** The 'price' feature is scaled using `StandardScaler` to normalize the data, making it easier for the model to converge.
   - **Categorical Features:** The 'variety' feature is one-hot encoded using `OneHotEncoder` to convert categorical data into a format that the SVR model can handle.
   - **Text Features:** The 'description' feature is transformed into numerical features using `TfidfVectorizer`. TF-IDF helps in representing the importance of words within the wine descriptions relative to the entire dataset. The `max_features` parameter is set to 500 to limit the vocabulary size, focusing on the most important terms.

3. **Model Training:**
   - An SVR model is used for predicting wine quality ('points'). SVR was chosen for its ability to handle non-linear relationships between the features and the target variable.
   - The model's hyperparameters are specified as `C=10`, `epsilon=0.2`, and `gamma='scale'`.

4. **Model Evaluation:**
   - The model's performance is evaluated using Mean Squared Error (MSE), which measures the average squared difference between the actual and predicted wine quality scores.

## Why These Choices?

- **SVR:** Chosen for its robustness in handling both linear and non-linear data.
- **TF-IDF:** TF-IDF was selected for text feature extraction because it effectively balances term frequency with the inverse document frequency, highlighting important words in the wine descriptions.
- **0.1 Sample Fraction:** A smaller fraction was used for initial model testing to reduce computation time, particularly given the computational demands of SVR, making the process more efficient.
