# Predict Credit Risk using Machine Learning Models

![Credit Risk](Images/Credit-Rating-800x267.jpg)

## Files

[Resampling Notebook](Starter_Code/credit_risk_resampling.ipynb)

[Ensemble Notebook](Starter_Code/credit_risk_ensemble.ipynb)

[Lending Club Loans Data](Resources/LoanStats_2019Q1.csv.zip)

## Expectations 
Build and evaluate several machine learning models to predict credit risk using data provided, which resembles peer-to-peer lending services data. Credit risk tends to an imbalanced classification problem, so I needed to use imbalanced-learn and scikit-learn libraries to do so. We were asked to use the following techniques:

1. [Resampling](#Resampling)
2. [Ensemble Learning](#Ensemble-Learning)

## Resampling Technique:
### Requirements:
Use the [imbalanced learn](https://imbalanced-learn.readthedocs.io) library to resample the LendingClub data and build and evaluate logistic regression classifiers using the resampled data.

### Steps Taken:
1. Read in CSV data to a dataframe.
2. Split data into training and test sets.
3. Scale the training and testing data using `Standard Scaler` from sklearn.preprocessing.
4. Oversample the data using `Naive Random Oversampler` and `SMOTE` algorithms.
5. Undersample the data using `Cluster Centroids` algorithm.
6. Over and undersample using a combo `SMOTEEN` algorithm

### For each of the above we need to do the following:
1. Train a `logistic regression classifier` from `sklearn.linear_model` using the resampled data.
2. Calculate the `balanced accuracy score` from `sklearn.metrics`.
3. Display the `confusion matrix` from `sklearn.metrics`.
4. Print the `imbalanced classification report` from `imblearn.metrics`.

### Questions to Answer:
* Which model had the best balanced accuracy score?
> Assuming I did everything right, the Smote Oversampled and SMOTEEN combo have the best (and equal) balanced accuracy scores at 0.9946680739911509.
* Which model had the best recall score?
>Assuming I did everything right, all of the recall average scores are the same, at .99.
* Which model had the best geometric mean score?
>Assuming I did everything right, all versions have the same geometric Mean scores of .99.

## Ensemble Learning Technique:
### Requirements:
Train and compare two different ensemble classifiers to predict loan risk and evaluate each model. You will use the [Balanced Random Forest Classifier](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedRandomForestClassifier.html) and the [Easy Ensemble Classifier](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.EasyEnsembleClassifier.html).

### Steps Taken:
1. Read the data into a dataframe using the provided starter code.
2. Split the data into training and testing sets.
3. Scale the training and testing data using the `StandardScaler` from `sklearn.preprocessing`. 
### For each of the above we need to do the following:
1. Train the model using the quarterly data from LendingClub provided in the `Resource` folder.
2. Calculate the balanced accuracy score from `sklearn.metrics`.
3. Display the confusion matrix from `sklearn.metrics`.
4. Generate a classification report using the `imbalanced_classification_report` from imbalanced learn.
5. For the balanced random forest classifier only, print the feature importance sorted in descending order (most important feature to least important) along with the feature score.

### Questions to Answer:
* Which model had the best balanced accuracy score?
>Easy Ensemble Classifier
* Which model had the best recall score?
>Easy Ensemble Classifier
* Which model had the best geometric mean score?
>Easy Ensemble Classifier
* What are the top three features?
>The top 3 features are: total_rec_prncp, total_rec_int and total_pymnt_inv