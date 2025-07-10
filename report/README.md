# Midterm Submission for SoS 25 – Image Segmentation &

This repository contains my weekly progress for the **Summer of Science 2025** program. The focus is on gaining hands-on experience in **data science**, **machine learning**, and **image segmentation** using Python

The work is organized into weekly folders, each containing notebooks and datasets relevant to the concepts practiced that week.

---

## 📁 week1 – Python Libraries (NumPy, Pandas, Matplotlib)

1. **Numpy.ipynb**  
   A comprehensive introduction to **NumPy**, covering:
   - Array creation and indexing
   - Vectorized operations
   - Matrix math

2. **intro_pandas.ipynb**  
   Basics of **Pandas** including:
   - Reading and exploring datasets
   - Column-wise operations
   - Descriptive statistics  
   Dataset used: *Ecommerce Purchases*

3. **pandas_practice.ipynb**  
   Practical notebook on **data cleaning & manipulation** using a medical insurance dataset. Covers:
   - Handling missing values
   - Feature transformation
   - Sorting and filtering

4. **Matplotlib.ipynb**  
   Visual guide to **Matplotlib**, showcasing:
   - Line, bar, scatter, and histogram plots
   - Styling and annotations
   - Plot customization for presentations

---

## 📁 week2 and week3 – Supervised ML & Data Preprocessing

1. **binary_classifier.ipynb**  
   Builds a **binary classification model** to predict hotel booking cancellations using Scikit-learn. Topics covered:
   - Data preprocessing
   - One-hot encoding
   - Model training and evaluation  
   Dataset used: `hotel.csv`

2. **handling_missing_data_replace.ipynb**  
   Focuses on **imputation strategies** using `pandas.replace()` with a weather dataset. Teaches how to:
   - Identify missing data
   - Replace specific values
   - Understand impact on model quality

3. **linear_regression_house_price_prediction.ipynb**  
   Implements **linear regression from scratch**, without any ML libraries. Trains on `house-prices.csv`:
   - Manual gradient descent
   - Mean squared error calculation
   - Plotting predicted vs actual prices

4. **random_forest_model.ipynb**  
   Uses **Random Forest Regression** to predict house prices using `train.csv`. Covers:
   - Feature importance
   - Hyperparameter tuning
   - Evaluation on a test set (`test (1).csv`)

---

## 📁 week4 – Neural Networks & Image Segmentation

1. **2_layer_NN.ipynb**  
   Implements a basic **two-layer neural network from scratch** using only NumPy and Pandas. The model is trained to predict heart disease presence using structured features such as age, blood pressure, and cholesterol levels from the `Heart_Disease_Prediction.csv` dataset.

2. **GradientDescent.ipynb**  
   Demonstrates the complete working of **gradient descent** for linear regression:
   - Random data generation
   - Visualization of loss surface
   - Step-by-step weight update visualization  
   Ideal for understanding how optimization works under the hood.

3. **image_segmentation_basics.ipynb_currently working**  
   Introduces foundational concepts in **image segmentation**, such as:
   - Pixel classification
   - Thresholding
   - Region-based segmentation  
   Also includes sample code and explanations for working with image datasets using OpenCV and skimage.

---

## 📦 Datasets Used

- **Heart_Disease_Prediction.csv**  
  Patient data for heart disease prediction (age, gender, cholesterol, etc.)

- **hotel.csv**  
  Hotel booking records, used for cancellation prediction (includes features like lead time, number of guests, etc.)

- **house-prices.csv**  
  Cleaned dataset for house price regression, including features like lot area, year built, etc.

- **train.csv / test (1).csv**  
  Training and testing splits from the house prices dataset (possibly from Kaggle House Prices Competition)

- **weather_data.csv**  
  A small dataset used for demonstrating missing value handling, with features like temperature, wind, and humidity.

---

## 📌 Summary

This repository is a consolidated submission of my work in the **Summer of Science 2025**. It reflects my learning in:

- Core data science libraries (NumPy, Pandas, Matplotlib)
- Classical machine learning algorithms (Linear Regression, Random Forests)
- Neural network implementation from scratch
- Practical image segmentation example

