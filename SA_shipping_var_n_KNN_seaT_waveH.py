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
X_raw = df[['Avg Wave Height','Avg Sea Temp']]
y = df['Alarm Signal']

# If the features do not sit in the same range (some take values that are higher/lower), we need to scale.
scaler = MinMaxScaler()
X = scaler.fit_transform(X_raw)
print(f"The range of feature inputs are within {X.min()} to {X.max()}")

# Split 80% as training dataset and 20% as testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state = rs)

# Try K from 1 to 50
max_k = 50
# Create an empty list to store f1score for each k
f1_scores = []

for k in range(1, max_k + 1):
    # Create a KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    # Train the classifier
    knn = knn.fit(X_train, y_train.values.ravel())
    preds = knn.predict(X_test)
    # Evaluate the classifier with f1score
    f1 = f1_score(preds, y_test)
    f1_scores.append((k, round(f1_score(y_test, preds), 4)))
# Convert the f1score list to a dataframe
Sea_Wave_f1 = pd.DataFrame(f1_scores, columns=['K', 'Sea/Wave f1'])
Sea_Wave_f1.set_index('K')

# Plot F1 results
ax = Sea_Wave_f1.plot(figsize=(12, 12))
ax.set(xlabel='Num of Neighbors', ylabel='F1 Score')
ax.set_xticks(range(1, max_k, 2));
plt.ylim((0.5, 1))
plt.title('KNN F1 Score')

plt.show()
