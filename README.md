### Combined Report: Fraud Detection System

#### 1.0 Introduction
This project aims to build a fraud detection system using machine learning techniques to identify fraudulent financial transactions from a dataset containing transaction details. The system leverages various preprocessing steps, feature engineering, and a range of machine learning models to detect fraud effectively. The primary objective is to identify patterns and key features that help distinguish fraudulent transactions from legitimate ones.

#### 2.0 Data Preparation

##### 2.1 Data Loading and Cleaning
The dataset `fraud_0.1origbase.csv` contains 636,262 rows and 11 columns, which represent transaction details. The columns are as follows:

- `step`: Transaction step
- `type`: Transaction type
- `amount`: Transaction amount
- `name_orig`: Origin account name
- `oldbalance_org`: Origin account balance before the transaction
- `newbalance_orig`: Origin account balance after the transaction
- `name_dest`: Destination account name
- `oldbalance_dest`: Destination account balance before the transaction
- `newbalance_dest`: Destination account balance after the transaction
- `is_fraud`: Binary indicator of whether the transaction is fraudulent (1 = 'yes', 0 = 'no')
- `is_flagged_fraud`: Binary indicator of whether the transaction was flagged as fraudulent by the system

The column names are converted into snake_case using the `inflection` library for consistency. Additionally, the categorical columns `is_fraud` and `is_flagged_fraud` are mapped from binary values (1 and 0) to 'yes' and 'no' labels.

##### 2.2 Data Splitting and Transformation
The data is split into training, validation, and test sets. Initially, an 80-20 split is applied for the training and test sets, ensuring stratification to maintain the class distribution of the target variable (`is_fraud`). The training set is further divided into training and validation sets using an 80-20 ratio.

One-hot encoding is applied to the `type` column (categorical variable), and Min-Max scaling is used for numerical features like `amount`, `oldbalance_org`, and `newbalance_orig`, ensuring all numerical features are scaled within the range [0, 1].

##### 2.3 Feature Selection
Features for model training are selected based on domain knowledge and feature importance analysis. The following features were selected for use in model training:

- `step`
- `oldbalance_org`
- `newbalance_orig`
- `newbalance_dest`
- `diff_new_old_balance`
- `diff_new_old_destiny`
- `type_TRANSFER`

These features are used for the machine learning models in the subsequent steps.

#### 3.0 Exploratory Data Analysis (EDA)

##### 3.1 Numerical Attributes
Descriptive statistics are calculated for numerical attributes, including mean, standard deviation, minimum, and maximum values. In addition, additional measures such as range, variation coefficient, skewness, and kurtosis are computed to understand the distribution of the numerical data. This analysis helps identify any potential outliers or imbalances that could affect model performance.

##### 3.2 Categorical Attributes
The categorical attributes in the dataset include:

- `type`: Transaction type
- `is_fraud`: Fraudulent transaction indicator
- `is_flagged_fraud`: Flagged fraudulent transaction indicator

The distribution of the target variable `is_fraud` is visualized using a bar chart, highlighting the imbalanced nature of the dataset, where non-fraudulent transactions significantly outnumber fraudulent ones.

#### 4.0 Machine Learning Modeling

##### 4.1 Baseline Model
A **DummyClassifier** is used as a baseline model to predict the majority class (non-fraudulent transactions). The performance of this model is poor, with all metrics (precision, recall, F1-score) for fraudulent transactions being 0, indicating the challenge posed by the imbalanced data.

##### 4.2 Logistic Regression
The **Logistic Regression** model performs moderately with a balanced accuracy of 0.584 and an F1-score of 0.288. However, the recall for the minority class (fraudulent transactions) is low at 0.168, suggesting that the model struggles to detect fraud effectively.

##### 4.3 K-Nearest Neighbors (KNN)
The **KNN** model shows similar performance to Logistic Regression, with a balanced accuracy of 0.565 and an F1-score of 0.23. It also faces challenges in detecting fraudulent transactions, as reflected in the low recall score.

##### 4.4 Support Vector Machine (SVM)
The **SVM** model performs poorly, achieving a balanced accuracy of 0.5, with very low precision and recall for fraudulent transactions. This indicates that SVM is ineffective for fraud detection in this dataset.

##### 4.5 Random Forest
The **Random Forest** model performs significantly better, with a balanced accuracy of 0.844 and an F1-score of 0.807. It demonstrates a good balance between precision and recall, successfully detecting fraudulent transactions.

##### 4.6 XGBoost
The **XGBoost** model performs excellently, achieving a balanced accuracy of 0.863 and an F1-score of 0.83. It demonstrates strong precision and recall, making it highly effective for detecting fraudulent transactions.

##### 4.7 LightGBM
The **LightGBM** model performs poorly compared to the others, with a balanced accuracy of 0.69 and a recall of 0.382, suggesting that it struggles to detect fraudulent transactions effectively.

##### 4.8 Model Performance Comparison
**XGBoost** and **Random Forest** are the top performers, with **XGBoost** slightly outperforming Random Forest in both recall and precision for fraudulent transactions. Other models such as **Logistic Regression**, **KNN**, and **Dummy** perform poorly, while **SVM** and **LightGBM** also lag behind.

#### 5.0 Hyperparameter Tuning

##### 5.1 Fine-Tuning XGBoost
Grid search is applied to fine-tune the hyperparameters of the **XGBoost** model. The best parameters identified are:

- Booster: `gbtree`
- Eta: 0.3
- Scale_pos_weight: 1

After applying these hyperparameters, the performance of the tuned **XGBoost** model remains strong, maintaining an excellent F1-score and balanced accuracy, similar to the untuned version.

#### 6.0 Conclusion
Based on the evaluation of several machine learning models, **XGBoost** and **Random Forest** emerge as the most effective models for detecting fraudulent transactions. Both models show strong performance, with **XGBoost** providing the highest balanced accuracy and F1-score for detecting fraudulent transactions. Hyperparameter fine-tuning further improves the performance of **XGBoost**, confirming its position as the most effective model. **Random Forest** also performs well and can be considered a strong alternative. 

Moving forward, further optimizations and techniques such as **Boruta** for feature selection and additional hyperparameter tuning can be explored to improve the models' performance. The results of this project provide a solid foundation for building a real-time fraud detection system, where the trained models can be deployed for monitoring transactions as they occur.

#### 7.0 Future Work
- **Hyperparameter Tuning**: Additional tuning for models like **Random Forest** and **LightGBM** could be beneficial.
- **Feature Engineering**: Techniques such as **Boruta** for feature selection can help improve model performance by focusing on the most impactful features.
- **Real-Time Implementation**: The optimized model can be deployed in a real-time fraud detection system to monitor transactions as they occur and flag suspicious activity.


