# Hotel Reservation Cancellation Prediction 🏨📅

# Project Overview 📊

This project focuses on predicting hotel reservation cancellations using machine learning techniques. By analyzing historical booking data, we aim to develop a model that can accurately forecast whether a reservation will be canceled. This predictive capability can help hotel management optimize resource allocation, improve customer service, and maximize revenue.

# Introduction 🔍

Hotel reservation cancellations are a significant challenge for the hospitality industry, impacting revenue and operational efficiency. This project aims to build a predictive model to identify reservations at risk of cancellation, enabling hotels to take proactive measures to mitigate these risks.

# Data Preprocessing 🧹

The data preprocessing steps include:

- Loading the dataset.

- Checking for null values and handling them.

- Dropping duplicate rows.

- Encoding categorical variables.

- Scaling numerical features.

- Combining date columns into a single datetime column.

- Handling outliers using the Interquartile Range (IQR) method.

# Exploratory Data Analysis 📈

The EDA involves visualizing the data to gain insights into the distribution of various features and their relationships with the target variable (booking status). Key visualizations include:

- Distribution of the number of adults and children.

- Time spent at the hotel (weekend nights vs. week nights).

- Date of arrival analysis (year, month, day, and day of the week).

- Services taken by guests (meal plan, room type, car parking, special requests).

- Lead time analysis.

- Market segment analysis.

- Guest's previous experience with the hotel.

- Average room price distribution.

- Impact of lead time on cancellation rates.

- Influence of market segment and repeated guests on cancellation.

- Effect of booking timing (weekdays vs. weekends) on cancellations.

# Feature Engineering ⚙️

Feature engineering steps include:

- Label encoding categorical variables.

- Scaling numerical features using StandardScaler.

- Creating new features if necessary.

# Model Building 🤖

Three machine learning models were built and evaluated:

- Logistic Regression

- Decision Tree Classifier

- Random Forest Classifier

Each model was tuned using GridSearchCV to find the best hyperparameters.

# Model Evaluation 🏆

The models were evaluated using the following metrics:

- Accuracy Score

- Mean Absolute Error

- Mean Squared Error

- Confusion Matrix

- Classification Report

The Random Forest Classifier performed the best with the highest accuracy score.

# Dashboard

![Screenshot (367)](https://github.com/user-attachments/assets/b257bb4e-0b3f-4849-a256-edb3a6dad907)


