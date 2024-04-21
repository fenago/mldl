# Predicting Breast Cancer Using RandomForest Classifier with sklearn

---

## Objectives:
- To learn how to implement a RandomForestClassifier with scikit-learn to predict whether breast cancer is malignant or benign.
- To practice training a machine learning model and making predictions.
- To evaluate the model's performance using accuracy as a metric.

## Time to Complete:
- Approximately 15 minutes.

## Prerequisites:
- Basic understanding of Python programming.
- Familiarity with machine learning concepts and the scikit-learn library.

---

## Lab Steps:

### Step 1: Import the necessary libraries
- **Why:** Importing necessary libraries to load the dataset and create a machine learning model.
```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
```

### Step 2: Load the dataset
- **Why:** Accessing the Breast Cancer Wisconsin (Diagnostic) dataset which contains the features and target labels for breast cancer classification.
```python
features, target = load_breast_cancer(return_X_y=True)
```

### Step 3: Inspect the features
- **Why:** Understanding the shape and type of data that we will use to train our model is important for preprocessing and model selection.
```python
print(features)
```

### Step 4: Inspect the target variable
- **Why:** Checking the target variable helps us to understand what we are trying to predict - whether cancer is malignant (harmful) or benign (not harmful).
```python
print(target)
```

### Step 5: Create and configure the model
- **Why:** Instantiating the RandomForestClassifier with a fixed random_state ensures reproducibility of the model's results.
```python
seed = 888
rf_model = RandomForestClassifier(random_state=seed)
```

### Step 6: Train the model
- **Why:** Training the model on the dataset allows it to learn the patterns associated with malignant and benign breast cancer.
```python
rf_model.fit(features, target)
```

### Step 7: Make predictions
- **Why:** Predicting on the training data gives us a quick understanding of how well the model has learned from the dataset.
```python
preds = rf_model.predict(features)
print(preds)
```

### Step 8: Evaluate the model
- **Why:** Calculating the accuracy of the model's predictions against the true labels helps in assessing the model's performance.
```python
print(accuracy_score(target, preds))
```

## Conclusion:
By completing this exercise, you have successfully trained a RandomForestClassifier to predict breast cancer malignancy with high accuracy. This exercise serves as a foundational practice in applying machine learning techniques for classification problems.
