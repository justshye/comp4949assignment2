import pandas as pd
from matplotlib import pyplot as plt
from pycaret.classification import *

# Load the dataset
data = pd.read_csv("match_data_v5.csv")

# Drop the first column
data = data.drop(columns=["matchId"])

# Separate features and target variable
X = data.drop(columns=["blueWin"])
y = data["blueWin"]

from sklearn.ensemble import RandomForestClassifier

# Initialize Random Forest Classifier
rf_classifier = RandomForestClassifier()

# Fit the model
rf_classifier.fit(X, y)

# Plot feature importance
plt.figure(figsize=(12, 8))  # Increase figure size
feat_importances = pd.Series(rf_classifier.feature_importances_, index=X.columns)
feat_importances.nlargest(27).plot(kind='barh')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()  # Adjust layout to prevent cropping
plt.show()
