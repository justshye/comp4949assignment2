import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
data = pd.read_csv("match_data_v5.csv")

# Selecting features and target variable
features = ["blueTeamTotalKills", "blueTeamDragonKills", "redTeamDragonKills", "redTeamTotalKills"]
target = "blueWin"

# Splitting data into features and target variable
X = data[features]
y = data[target]

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Exporting the trained model as a .pkl file
with open("model_pkl_logistic_regression", "wb") as file:
    pickle.dump(model, file)
