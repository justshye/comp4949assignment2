import pandas as pd
from pycaret.classification import *

# Load the dataset
data = pd.read_csv("match_data_v5.csv")

# Drop the first column
data = data.drop(columns=["matchId"])

# Separate features and target variable
X = data.drop(columns=["blueWin"])
y = data["blueWin"]

# Selecting features and target variable
features = ["blueTeamTotalKills", "blueTeamDragonKills", "redTeamDragonKills", "redTeamTotalKills"]
target = "blueWin"

# Initialize PyCaret setup
exp_clf101 = setup(data, target=target, session_id=123, log_experiment=True, experiment_name='lol_prediction')

# Compare Models
best_model = compare_models()

# Tune the best model
tuned_best_model = tune_model(best_model)

# Evaluate the model
evaluate_model(tuned_best_model)

# Deploy the model (if needed)
# deploy_model(tuned_best_model, model_name='best-lol-prediction', authentication={'bucket': 's3-bucket-name'})
