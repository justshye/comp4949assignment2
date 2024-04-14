import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

# Load the dataset
data = pd.read_csv("match_data_v5.csv")

# Drop the first column
data = data.drop(columns=["matchId"])

# Separate features and target variable
X = data.drop(columns=["blueWin"])
y = data["blueWin"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize the model
model = LogisticRegression(max_iter=2000)

# Number of features to try
num_features_to_try = 27  # Total number of features available

best_accuracy = 0
best_num_features = 0

# Iterate through different numbers of features
for num_features in range(1, num_features_to_try + 1):
    # Initialize RFE
    rfe = RFE(model, n_features_to_select=num_features)

    # Fit RFE
    rfe.fit(X_train, y_train)

    # Get selected features
    selected_features = X.columns[rfe.support_]

    # Train model with selected features
    model.fit(X_train[selected_features], y_train)

    # Evaluate model
    accuracy = model.score(X_test[selected_features], y_test)

    # Check if this is the best accuracy so far
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_num_features = num_features

# Print the best result
print("Best Accuracy:", best_accuracy)
print("Number of Features:", best_num_features)

# BEST NUMBER OF FEATURES: 9
