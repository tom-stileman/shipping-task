import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# For more comprehensive models, including scaling, encoding etc.
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay, precision_recall_fscore_support, precision_score, recall_score

# Sample data (if you have a dataframe, you could use X_raw = df.iloc[:,:-1] and y_raw = df.iloc[:,-1:], in the situation where your target is in the last column and your predictors features are in the others)
df = pd.read_excel('SA_shipping_data.xlsx')
X_raw = df[["Avg Wave Height"]]
y_raw = df["Alarm Signal"]

# Target Label Encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw.values.ravel())
# Note that ravel() function flattens the vector.
np.unique(y, return_counts=True)
# No ensure no target labels are double values.

# Split data into training and testing sets (stratify makes sure the split is representative)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"Training dataset shape, X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Testing dataset shape, X_test: {X_test.shape}, y_test: {y_test.shape}")

# Initialize Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

