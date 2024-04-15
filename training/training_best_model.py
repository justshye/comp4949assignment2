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

# Save the best model
# save_model(best_model, 'model_pkl')

# # Plot Model
# plot_model(best_model, plot='auc')
#
# # Plot Feature Importance
# plot_model(best_model, plot='feature')
#
# # Plot Confusion Matrix
# plot_model(best_model, plot='confusion_matrix')
#
# # Plot Class Prediction Error
# plot_model(best_model, plot='error')
#
# # Plot Learning Curve
# plot_model(best_model, plot='learning')
#
# # Plot Validation Curve
# plot_model(best_model, plot='vc')
#
# # Evaluate Model
# evaluate_model(best_model)
