import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# Load the dataset
data = pd.read_csv("match_data_v5.csv")

# Drop the first column
data = data.drop(columns=["matchId"])

# Calculate the correlation matrix
correlation_matrix = data.corr()

# Extract correlations with the target variable
target_correlations = correlation_matrix["blueWin"].drop("blueWin")

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(target_correlations.to_frame(), cmap="coolwarm", annot=True, fmt=".2f", cbar=True, linewidths=0.5)
plt.title("Correlation with 'blueWin'")
plt.xlabel("Features")
plt.ylabel("Correlation")
plt.tight_layout()
plt.show()
