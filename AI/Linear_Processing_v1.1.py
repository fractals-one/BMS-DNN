# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("/home/ramanujan/git_repos/BMS-DNN/BMS/dataset/soc_processed_data.csv")

# Split the dataset into features and labels
X = data.drop(["SoC Percentage","SoH Percentage"], axis=1).values
y = data[["SoC Percentage","SoH Percentage"]].values

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the Linear Regression model
linear_model = LinearRegression()

# Train the model
linear_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred_soc_linear, y_pred_soh_linear = linear_model.predict(X_test_scaled).T

# Calculate the metrics
mae_linear = mean_absolute_error(y_test.T, [y_pred_soc_linear,y_pred_soh_linear])
mse_linear = mean_squared_error(y_test.T, [y_pred_soc_linear,y_pred_soh_linear])
rmse_linear = np.sqrt(mse_linear)
r2_linear = r2_score(y_test.T, [y_pred_soc_linear,y_pred_soh_linear])

# Print the metrics
print(f"Linear Regression MAE: {mae_linear:.4f}")
print(f"Linear Regression MSE: {mse_linear:.4f}")
print(f"Linear Regression RMSE: {rmse_linear:.4f}")
print(f"Linear Regression R-squared: {r2_linear:.4f}")


# Create a scatter plot of true SOC values vs predicted SOC values
fig, ax = plt.subplots(2,2)
ax[0,0].scatter(y_test[:,0], y_pred_soc_linear)
ax[0,0].set(xlabel="True SOC Values",ylabel="Predicted SOC Values")
ax[0,0].set_title("True SOC vs Predicted SOC",fontweight='bold')

# Add a diagonal line to the plot to indicate perfect predictions
diagonal_line = np.linspace(0, 1, 100)
ax[0,0].plot(diagonal_line, diagonal_line, 'r', linestyle='--')
# Annotate the plot with metrics
ax[0,0].text(0.02, 0.9, f"MAE: {mae_linear:.4f}", style='italic', fontweight='bold',
           bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 4}, transform=ax[0,0].transAxes)
ax[0,0].text(0.02, 0.8, f"MSE: {mse_linear:.4f}", style='italic', fontweight='bold',
           bbox={'facecolor': 'cyan', 'alpha': 0.5, 'pad': 4}, transform=ax[0,0].transAxes)
ax[0,0].text(0.02, 0.7, f"RMSE: {rmse_linear:.4f}", style='italic', fontweight='bold',
           bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 4}, transform=ax[0,0].transAxes)
ax[0,0].text(0.02, 0.6, f"R-squared: {r2_linear:.4f}", style='italic', fontweight='bold',
           bbox={'facecolor': 'orange', 'alpha': 0.5, 'pad': 4}, transform=ax[0,0].transAxes)

test_soc_length = np.array(list(range(0,len(y_test[:,0]))))
ax[1,0].plot(test_soc_length,y_test[:,0],'tab:red',linestyle='--',label='Dataset')
ax[1,0].plot(test_soc_length,y_pred_soc_linear,'tab:green',linestyle='--',label="Predicted")
ax[1,0].set(xlabel="Samples",ylabel="SOC")
ax[1,0].legend(loc='upper right')

ax[0,1].scatter(y_test[:,1], y_pred_soh_linear)
ax[0,1].set(xlabel="True SOH Values",ylabel="Predicted SOH Values")
ax[0,1].set_title("True SOH vs Predicted SOH",fontweight='bold')

# Add a diagonal line to the plot to indicate perfect predictions
diagonal_line = np.linspace(0, 1, 100)
ax[0,1].plot(diagonal_line, diagonal_line, 'r', linestyle='--')
# Annotate the plot with metrics
ax[0,1].text(0.02, 0.9, f"MAE: {mae_linear:.4f}", style='italic', fontweight='bold',
           bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 4}, transform=ax[0,1].transAxes)
ax[0,1].text(0.02, 0.8, f"MSE: {mse_linear:.4f}", style='italic', fontweight='bold',
           bbox={'facecolor': 'cyan', 'alpha': 0.5, 'pad': 4}, transform=ax[0,1].transAxes)
ax[0,1].text(0.02, 0.7, f"RMSE: {rmse_linear:.4f}", style='italic', fontweight='bold',
           bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 4}, transform=ax[0,1].transAxes)
ax[0,1].text(0.02, 0.6, f"R-squared: {r2_linear:.4f}", style='italic', fontweight='bold',
           bbox={'facecolor': 'orange', 'alpha': 0.5, 'pad': 4}, transform=ax[0,1].transAxes)

test_soh_length = np.array(list(range(0,len(y_test[:,1]))))
ax[1,1].plot(test_soh_length,y_test[:,1],'tab:purple',linestyle='--',label='Dataset')
ax[1,1].plot(test_soh_length,y_pred_soh_linear,'tab:green',linestyle='--',label="Predicted")
ax[1,1].set(xlabel="Samples",ylabel="SOH")
ax[1,1].legend(loc='upper right')

plt.tight_layout()
plt.show()


