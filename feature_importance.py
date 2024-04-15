# Import necessary libraries
import pandas as pd
from model_selection.classification import *

# Load the dataset
data = pd.read_csv("match_data_v5.csv")

# Drop the first column
data = data.drop(columns=["matchId"])

# Initialize PyCaret setup
clf = setup(data, target="blueWin", session_id=123)

# Compare different models
best_model = compare_models()

# Plot feature importance
plot_model(best_model, plot='feature')
