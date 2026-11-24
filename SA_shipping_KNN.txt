import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler

# Evaluation metrics related methods
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

# Define a random seed to reproduce any random process
rs = 42

# Ignore any deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# Read datast in csv format
df = pd.read_excel('SA_shipping_data.xlsx')

# Split the dataset into input X and output y
X_raw = df[['Longitude','Latitude']]
y = df['Alarm Signal']

# If the features do not sit in the same range (some take values that are higher/lower), we need to scale.
scaler = MinMaxScaler()
X = scaler.fit_transform(X_raw)
print(f"The range of feature inputs are within {X.min()} to {X.max()}")

# Split 80% as training dataset and 20% as testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state = rs)

# Define a KNN classifier with a chosen number of neighbours
knn_model = KNeighborsClassifier(n_neighbors=2)

# We can train the model with X_train and y_train, and we use ravel() method to convert the data frame y_train to a vector.
knn_model.fit(X_train, y_train.values.ravel())

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=None, n_neighbors=2, p=2,
           weights='uniform')

# We can make predictions on the X_test dataframe.
preds = knn_model.predict(X_test)

# To evaluate the KNN classifier, we provide a pre-defined method to return the commonly used evaluation metrics such as accuracy, recall, precision, f1score, and so on, based on the true classes in the 'y_test' and model predictions.
def evaluate_metrics(yt, yp):
    results_pos = {}
    results_pos['accuracy'] = accuracy_score(yt, yp)
    precision, recall, f_beta, _ = precision_recall_fscore_support(yt, yp, average='binary')
    results_pos['recall'] = recall
    results_pos['precision'] = precision
    results_pos['f1score'] = f_beta
    return results_pos

results = evaluate_metrics(y_test, preds)

print(results)

