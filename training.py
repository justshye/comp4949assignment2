# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("match_data_v5.csv")

# Define features and target variable
features = [
    "blueTeamTotalKills",
    "blueTeamDragonKills",
    "redTeamDragonKills",
    "redTeamTotalKills",
]
target = "blueWin"

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data[features], data[target], test_size=0.2, random_state=42
)

# Initializing the Logistic Regression model
model = LogisticRegression(max_iter=2000)

# Training the model
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
