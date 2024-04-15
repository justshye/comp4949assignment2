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
model = LogisticRegression(max_iter=10000)

# Initialize RFE
rfe = RFE(
    model, n_features_to_select=5
)  # You can adjust the number of features to select

# Fit RFE
rfe.fit(X_train, y_train)

# Get selected features
selected_features = X.columns[rfe.support_]

# Train model with selected features
model.fit(X_train[selected_features], y_train)

# Evaluate model
accuracy = model.score(X_test[selected_features], y_test)
print("Accuracy:", accuracy)

# Print selected features
print("Selected features:", selected_features)
