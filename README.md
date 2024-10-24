# Yiren-590Assignment7
# Predicting High-Value Retail Transactions Using Explainable Deep Learning

## Project Overview
This project aims to build a deep learning model to predict high-value transactions in retail sales data and use explainable AI techniques (specifically Integrated Gradients) to understand the importance of different features in the model’s predictions. By interpreting the model's behavior, this project helps identify which features are the most influential in determining high-value transactions, enhancing both model transparency and decision-making insights.

## Table of Contents
1. [Project Objectives](#project-objectives)
2. [Dataset Description](#dataset-description)
3. [Methodology](#methodology)
4. [Model Performance](#model-performance)
5. [Feature Importance Analysis](#feature-importance-analysis)
6. [Statistical Testing & Hypothesis Testing](#statistical-testing--hypothesis-testing)
7. [Conclusion](#conclusion)

## Project Objectives
The key objectives of this project are:
- To predict whether a transaction is high-value (i.e., total amount > 500) using deep learning.
- To apply Integrated Gradients, a popular explainable AI method, to quantify feature importance in model predictions.
- To evaluate the statistical significance of feature attributions and compare the attention weights of the 'Age' feature with other features, determining whether there is a significant difference.

## Dataset Description
The dataset used for this project contains retail sales transaction records, including the following fields:
- `Date`: The transaction date.
- `Gender`: The gender of the customer.
- `Age`: The age of the customer.
- `Product Category`: The category of the purchased product.
- `Total Amount`: The total amount of the transaction.

Additional features extracted from the data include:
- `Month`: Extracted from the 'Date' field.
- `DayOfWeek`: Extracted from the 'Date' field.

The target variable is binary, indicating whether a transaction is high-value (`1`) or not (`0`).

## Methodology
1. **Data Preprocessing**:
   - Categorical variables (`Gender`, `Product Category`) were label-encoded.
   - New features such as `Month` and `DayOfWeek` were extracted from the `Date` column.
   - Data was scaled using standard scaling to normalize feature distributions.

2. **Model Development**:
   - A deep neural network (DNN) with three hidden layers was built using TensorFlow/Keras for binary classification.
   - The model was trained on the processed dataset for 50 epochs with a batch size of 32.

3. **Explainability Analysis**:
   - Integrated Gradients was used to analyze feature importance and interpret the model’s decision-making.
   - Statistical testing was conducted to compare the model's attention on the 'Age' feature against other features.

## Model Performance
- **Test Accuracy**: `0.6800`
- **Final Training Accuracy**: `0.7109`
- **Final Validation Accuracy**: `0.6500`

The model performs reasonably well, although there is evidence of slight overfitting as the training accuracy exceeds the validation accuracy.

## Feature Importance Analysis
Based on Integrated Gradients, the feature importance ranking is as follows:
1. `Month`: `0.0544`
2. `Gender`: `0.0542`
3. `Age`: `0.0507`
4. `DayOfWeek`: `0.0477`
5. `Product_Category`: `0.0377`

This analysis indicates that `Month` and `Gender` are the most influential features, while `Product_Category` has the least impact on model predictions.

## Statistical Testing & Hypothesis Testing
- A t-test was used to compare the attention weights of the 'Age' feature with other features.
- **Significant result**:
  - `Product_Category` showed a statistically significant difference in attribution compared to `Age` (t-stat = `2.7614`, p-value = `0.0060`).
- **Hypothesis Testing Conclusion**:
  - The null hypothesis (H0) was rejected, indicating that the deep neural network shows significantly different attention weights for `Age` compared to some other features, particularly `Product_Category`.

## Conclusion
This project successfully demonstrates the use of explainable deep learning to predict high-value transactions in retail data. The model achieves reasonable accuracy, with `Month` and `Gender` being the most influential features. Integrated Gradients provides clear insights into feature importance, while statistical testing confirms significant differences in model attention weights for `Product_Category` compared to `Age`. Future improvements could include fine-tuning the model, collecting more data, and exploring additional explainable AI techniques to further enhance model performance and transparency.

