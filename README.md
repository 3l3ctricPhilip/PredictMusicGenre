# Music Genre Prediction

## Project Description
This project aims to develop a machine learning model capable of predicting the music genre based on numerical features of songs. By analyzing both acoustic properties and lyrical data, the model will help classify songs into their respective genres, providing valuable insights into music data.

## Problem Statement
In an era of growing digital music libraries, organizing and categorizing songs by genre has become crucial for better user experience. Manual classification of genres is labor-intensive and prone to errors. This project addresses the need for an automated solution to predict the genre of a song based on its numerical and textual features, making it easier to manage large music datasets efficiently.

## Data Source
The dataset used for this project is a comprehensive collection of songs with the following characteristics:
- **Size:** 28,372 records
- **Attributes:** 31 features, including:
  - Song details: artist name, track name, release year, genre (target variable)
  - Acoustic features: danceability, loudness, energy, valence, instrumentalness, and more
  - Lyrics analysis: features capturing emotions and themes, such as sadness, romanticism, and violence

### Justification for Data Choice
This dataset provides a rich variety of numerical features that are highly relevant for building machine learning models. The inclusion of lyrical analysis and acoustic properties ensures a multi-faceted approach to genre prediction, making the dataset ideal for this task.

## Project Goals
1. **Data Processing and Analysis:**
   - Explore the dataset to understand its structure and properties.
   - Clean and preprocess data to handle missing values and outliers.
2. **Model Development:**
   - Build and train a machine learning model to predict the genre of a song.
   - Evaluate model performance using appropriate metrics (e.g., accuracy, precision, recall).
3. **Model Optimization and Testing:**
   - Fine-tune the model to improve its accuracy.
   - Test the model on unseen data to ensure generalization.
4. **Deployment:**
   - Package the model for deployment in a user-friendly application or API.

## Workflow Overview
1. **Data Collection and Exploration:**
   - Load the dataset and explore key features.
2. **Data Preprocessing:**
   - Normalize and scale numerical features.
   - Encode categorical variables (e.g., genre).
3. **Model Training and Validation:**
   - Train multiple algorithms (e.g., Random Forest, Neural Networks) and compare results.
4. **Model Deployment:**
   - Develop an API or integrate the model into a web application.

## Diagram
```mermaid
graph TD
    A[Data Collection] --> B[Data Preprocessing]
    B --> C[Feature Engineering]
    C --> D[Model Training]
    D --> E[Model Evaluation]
    E --> F[Deployment]
```

---

## Requirements
- **Programming Language:** Python
- **Libraries:** pandas, scikit-learn, TensorFlow/PyTorch, matplotlib, seaborn
- **Environment:** Anaconda or virtualenv

## Expected Outcome
By the end of this project, we aim to deliver:
1. A trained and validated machine learning model capable of predicting music genres.
2. A documented analysis of the dataset and model performance.
3. A deployable version of the model (e.g., Flask API or web app).

## Data Analysis and Observations

This section summarizes the findings and insights gained from the automated data analysis report.

### **Dataset Overview**
- **Number of variables:** 26
- **Number of observations:** 19,860
- **Missing values:** None (0%)
- **Duplicate rows:** None (0%)
- **Dataset size in memory:** 4.1 MiB

### **Alerts and Highlights**
1. **High Correlations:**
   - `acousticness` is highly correlated with `energy` and `loudness`.
   - `release_date` is highly correlated with `loudness`.
   - `topic` is highly correlated with `world/life`.

2. **Variables with Zero Values:**
   - `genre`: 16.3% of the values are zeros.
   - `instrumentalness`: 27.7% of the values are zeros.
   - `topic`: 2.2% of the values are zeros.

### **Correlation Matrix Insights**
- The dataset contains several highly correlated variables, which may indicate redundancy:
  - For example, `acousticness`, `energy`, and `loudness` show significant correlations.
  - Similarly, `topic` and `world/life` exhibit high correlation.

### **Missing Values**
- There are no missing values in the dataset, eliminating the need for imputation.

### **Recommendations**
1. **Feature Reduction:**
   - Remove or aggregate highly correlated variables such as `acousticness`, `energy`, and `loudness` to avoid redundancy.
   - Consider combining `topic` and `world/life` into a single variable.

2. **Handling Zero Values:**
   - Analyze the impact of `instrumentalness` (27.7% zeros) and determine if it is necessary for the model.
   - Investigate zero values in `genre` and `topic` to assess their significance.

3. **Dimensionality Reduction:**
   - Apply techniques like PCA to reduce the dimensionality of the dataset if needed.

4. **Model Selection:**
   - Models like **Random Forest** and **Gradient Boosting** can handle the dataset's characteristics effectively.
   - For reduced dimensions, linear models might be explored.

### **Next Steps**
- Prepare the dataset for modeling by addressing redundancy and handling zero values.
- Perform additional feature engineering to optimize model performance.

