# Consumer Complaint Text Classification
**Please Read the python notebook for Detailed explanation of each code.**
[Google Colab link](https://colab.research.google.com/drive/1e3V_PtcRTNgAODTgkQz9COWQ-KTsdv3i?usp=sharing)

This repository contains a fine-tuned LinearSVC model and a Python colab notebook for training, evaluating, and deploying a text classification model for consumer complaints. The model's purpose is to categorize consumer complaints into specific product categories - ["Credit reporting, repair, or other", "Debt collection", "Consumer Loan", "Mortgage"]. This README.md file provides comprehensive documentation, explaining the code's functionality, implementation, and usage.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Code Overview](#code-overview)
  - [1. Explanatory Data Analysis and Feature Engineering](#1-explanatory-data-analysis-and-deature-engineering)
  - [2. Text Pre-processing](#2-text-pre-processing)
  - [3. Selection of Multi Classification model](#3-selection-of-multi-classification-model-and-4-Comparison-of-model-performance)
  - [4. Comparison of model performance](#3-selection-of-multi-classification-model-and-4-Comparison-of-model-performance)
  - [5. Model Evaluation](#5-model-evaluation)
  - [6. Model Deployment/Prediction](#6-model-deployment/Prediction)
- [Usage](#usage)
- [Example](#example)
- [Conclusion](#conclusion)

## Introduction

Consumer complaints provide valuable insights for businesses and regulatory bodies. This project focuses on developing a text classification model capable of automatically categorizing consumer complaints into predefined product categories. The key objectives of this project are as follows:

1. Data preparation and preprocessing of consumer complaint data.
2. Training and evaluating a machine learning model for text classification.
3. Deploying the trained model to predict new complaints' categories.

## Prerequisites

Before running the code, ensure that you have the following prerequisites:

- Python 3.x
- Jupyter Notebook (for interactive code execution)
- Required Python libraries (install via `pip install`):
  - pandas
  - numpy
  - scikit-learn
  - joblib
  - matplotlib
  - seaborn

## Code Overview

The code is organized into several distinct sections, each serving a specific purpose. Below is a comprehensive overview of these sections:

### 1. Explanatory Data Analysis and Feature Engineering

This section involves loading the consumer complaint dataset, conducting data cleaning, and optionally sampling the dataset for manageability. It also includes mapping similar product categories to consolidate them and generating a category mapping for reference.

### 2. Text Pre-processing

Text pre-processing is a critical step in natural language processing (NLP). This section focuses on cleaning and preparing the complaint text data. Key steps include converting text to lowercase, tokenization, punctuation removal, and generating TF-IDF (Term Frequency-Inverse Document Frequency) vectors for text representation.

### 3. Selection of Multi Classification model and 4. Comparison of model performance 

To classify consumer complaints, various machine learning models are explored, including Linear Support Vector Machine (LinearSVC), Logistic Regression, Random Forest, Multinomial Naive Bayes, K Neighbors Classifier, AdaBoost Classifier, and Bagging Classifier. Cross-validation is used to select the best-performing model based on accuracy.

### 5. Model Evaluation

The chosen model (LinearSVC) is further fine-tuned using GridSearchCV to optimize hyperparameters. Model performance is evaluated using a range of metrics, including accuracy, mean squared error (MSE), root mean squared error (RMSE), mean absolute error (MAE), R-squared (R2) score, and explained variance score. A confusion matrix visualizes the model's performance in categorizing each product category.

### 6. Model Deployment/Prediction

Once the model is trained and evaluated, it is saved along with the TF-IDF vectorizer and category mapping for future use. This section provides detailed instructions on how to load and employ the trained model for making predictions on new complaint text data.

## Usage

To utilize the code, follow these steps:

1. Install the required Python libraries, as specified in the Prerequisites section.

2. Download the consumer complaint dataset. You can use your dataset or obtain one from a relevant source.

3. Update the file path in the code to reference your dataset.

4. Execute the code within a Jupyter Notebook or any Python environment.

5. Refer to the code documentation and comments to understand each step of the process.

## Example

To illustrate how the code operates, consider the following typical workflow:

1. Load and preprocess the consumer complaint dataset.

2. Train and evaluate a LinearSVC text classification model.

3. Save the model, TF-IDF vectorizer, and category mapping for future use.

4. Utilize the trained model to predict the category of a new complaint text.

For a detailed example, please refer to the [example section](#example) below.

## Conclusion

As a part of the assignment given by Kaiburr, This project offers a comprehensive solution for classifying consumer complaints into product categories. By adhering to the steps outlined in this README.md, you can train, evaluate, and deploy your text classification model for similar tasks. The code's flexibility enables adaptation to various datasets and classification challenges.

For inquiries or further assistance, please feel free to reach out.

