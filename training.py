import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
data = pd.read_csv("match_data_v5.csv")

# Selecting features and target variable
features = ["blueTeamTotalKills", "blueTeamDragonKills", "redTeamDragonKills", "redTeamTotalKills"]
target = "blueWin"
data = data[features + [target]]

# Separate features and target variable
X = data.drop(columns=[target])
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the LDA model
lda = LinearDiscriminantAnalysis()

# Fit the model on the training data
lda.fit(X_train, y_train)

# Save the model to a file
with open("model_pkl", "wb") as f:
    pickle.dump(lda, f)

# Make predictions on the testing data
predictions = lda.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
