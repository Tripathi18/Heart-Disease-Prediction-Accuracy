# Heart-Disease-Prediction-Accuracy


This project focuses on classifying heart disease based on various features using machine learning algorithms. The code provided performs data preprocessing, feature scaling, and applies different classification algorithms to the heart disease dataset.


## Table of Contents


- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)


## Introduction


Heart disease is a prevalent and serious medical condition that can have significant implications for individuals' health. Predicting and classifying heart disease can aid in early detection and prompt treatment. This project utilizes machine learning algorithms to build models that can classify heart disease based on various factors.


The main steps involved in the project are:


1. Data loading and exploration: The heart disease dataset is loaded from a CSV file. Information about the dataset, such as the number of instances and features, is displayed. Descriptive statistics and correlation matrix plots are generated to gain insights into the data.


2. Data preprocessing and feature scaling: The dataset is preprocessed by encoding categorical variables using one-hot encoding. Numerical features are scaled using standardization to ensure they have similar scales.


3. Splitting the dataset: The dataset is split into training and testing sets using the train-test split technique. This allows the evaluation of the models' performance on unseen data.


4. Classification algorithms:
   - Decision Tree Classifier: The Decision Tree Classifier is trained and evaluated for different numbers of maximum features. Accuracy scores are computed and plotted to determine the optimal number of features.
   - K Nearest Neighbors Classifier: The K Nearest Neighbors Classifier is trained and evaluated for different numbers of neighbors (k). Accuracy scores are computed and plotted to determine the optimal value of k.
   - Support Vector Machine Classifier: The Support Vector Machine Classifier is trained and evaluated for different kernel functions. Accuracy scores are computed and plotted to determine the optimal kernel.
   - Random Forest Classifier: The Random Forest Classifier is trained and evaluated for different numbers of estimators (decision trees). Accuracy scores are computed and plotted to determine the optimal number of estimators.


## Installation


Install the required dependencies:


pip install numpy pandas matplotlib scikit-learn


Download the heart disease dataset and place it in the project directory.


## Usage


Open the Python script heart_disease_classification.py in your preferred development environment.


Update the file path in the code to the location of the heart disease dataset:

dataset = pd.read_csv('path/to/heart.csv')

**Run the script to execute the code.**

python heart_disease_classification.py


The script will generate various plots and display the accuracy scores of the classification algorithms.


## Results


The results of the classification algorithms are displayed through plots and printed scores. The following results are obtained:


**Decision Tree Classifier:** The optimal number of maximum features and the corresponding accuracy score are determined. The scores are plotted against the number of features.
K Nearest Neighbors Classifier: The optimal number of neighbors (k) and the corresponding accuracy score are determined. The scores are plotted against different values of k.
Support Vector Machine Classifier: The optimal kernel function and the corresponding accuracy score are determined. The scores are visualized using a bar chart.
Random Forest Classifier: The optimal number of estimators (decision trees) and the corresponding accuracy score are determined. The scores are visualized using a bar chart.


## Contributing


Contributions to the project are welcome. If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.


Feel free to modify the README.md file according to your project's specifics and add any additional sections or information that you find relevant.
