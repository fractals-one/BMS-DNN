import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from sklearn.metrics import mean_absolute_error

# Load the data
data = pd.read_csv('/home/ramanujan/git_repos/BMS-DNN/BMS/dataset/soc_processed_data.csv')
df = pd.DataFrame(data)

# Normalize features
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Current', 'Voltage', 'Temperature', 'SoC Capacity']])
df[['Current', 'Voltage', 'Temperature', 'SoC Capacity']] = scaled_data

# Prepare target variables
X = df[['Current', 'Voltage', 'Temperature', 'SoC Capacity']].values
y = df[['SoC Percentage',"SoH Percentage"]].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define CNN model
cnn_model = Sequential([
    Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(4, 1)),  # 4 timesteps, 1 feature
    Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),      # Same padding to maintain input shape
    Flatten(),
    Dense(50, activation='relu'),
    Dense(2)  # Output layer for predicting SoC percentage
])

cnn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Reshape the data for Conv1D input
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Train the CNN model
cnn_model.fit(X_train_cnn, y_train, epochs=10, batch_size=4, verbose=2)

# Save the model to a file
cnn_model.save("/home/ramanujan/git_repos/BMS-DNN/BMS/BMS_DNN/model/dnn_cnn_model.keras")

# Retrive the saved model, useful during debug of the plotting section
cnn_model = load_model("/home/ramanujan/git_repos/BMS-DNN/BMS/BMS_DNN/model/dnn_cnn_model.keras")

# Evaluate the CNN model
cnn_loss, cnn_mae = cnn_model.evaluate(X_test_cnn, y_test)
print(f'CNN MAE: {cnn_mae}')

# Predict with the CNN model
y_pred_soc_cnn ,y_pred_soh_cnn = cnn_model.predict(X_test_cnn).T

# Calculate metrics
mae_cnn = mean_absolute_error(y_test.T, [y_pred_soc_cnn,y_pred_soh_cnn])
print(f'CNN MAE: {mae_cnn}')
"""
# Plot Actual vs Predicted values and scatter plot
fig, ax = plt.subplots(2, figsize=(10, 10))

# Scatter plot (True SOC vs Predicted SOC)
ax[0].scatter(y_soc_test, y_pred_cnn)
ax[0].set(xlabel="True SoC Values", ylabel="Predicted SoC Values")
ax[0].set_title("True SoC vs Predicted SoC", fontweight='bold')

# Diagonal line for perfect prediction
diagonal_line = np.linspace(0, 1, 100)
ax[0].plot(diagonal_line, diagonal_line, 'r', linestyle='--')

# Annotating the plot with the CNN's loss and MAE values
ax[0].text(-0.02, 0.9, f"Mean Absolute Error: {mae_cnn:}", style='italic', fontweight='bold', bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 4})
ax[0].text(-0.02, 0.8, f"Loss: {cnn_loss:}", style='italic', fontweight='bold', bbox={'facecolor': 'cyan', 'alpha': 0.5, 'pad': 4})

# Line plot (Actual vs Predicted SoC over samples)
test_soc_length = np.arange(len(y_soc_test))
ax[1].plot(test_soc_length, y_soc_test, 'tab:red', label='Actual SoC Percentage')
ax[1].plot(test_soc_length, y_pred_cnn, 'tab:green', linestyle='--', label="Predicted SoC Percentage")
ax[1].set(xlabel="Sample Index", ylabel="SoC Percentage")
ax[1].legend(loc='upper right')
"""

# Create a scatter plot of true SOC values vs predicted SOC values
fig, ax = plt.subplots(2,2)
ax[0,0].scatter(y_test[:,0], y_pred_soc_cnn)
ax[0,0].set(xlabel="True SOC Values",ylabel="Predicted SOC Values")
ax[0,0].set_title("True SOC vs Predicted SOC",fontweight='bold')

# Add a diagonal line to the plot to indicate perfect predictions
diagonal_line = np.linspace(0, 1, 100)
ax[0,0].plot(diagonal_line, diagonal_line, 'r', linestyle='--')
ax[0,0].text(-0.02, 0.9, f"Mean Absolute Error on test set: {mae_cnn:.4f}", style='italic',fontweight='bold',bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 4})
ax[0,0].text(-0.02, 0.8, f"Loss on test set: {cnn_loss:.4f}", style='italic',fontweight='bold',bbox={'facecolor': 'cyan', 'alpha': 0.5, 'pad': 4})

test_soc_length = np.array(list(range(0,len(y_test[:,0]))))
ax[1,0].plot(test_soc_length,y_test[:,0],'tab:red',linestyle='--',label='Dataset')
ax[1,0].plot(test_soc_length,y_pred_soc_cnn,'tab:green',linestyle='--',label="Predicted")
ax[1,0].set(xlabel="Samples",ylabel="SOC")
ax[1,0].legend(loc='upper right')

ax[0,1].scatter(y_test[:,1], y_pred_soh_cnn)
ax[0,1].set(xlabel="True SOH Values",ylabel="Predicted SOH Values")
ax[0,1].set_title("True SOH vs Predicted SOH",fontweight='bold')

# Add a diagonal line to the plot to indicate perfect predictions
diagonal_line = np.linspace(0, 1, 100)
ax[0,1].plot(diagonal_line, diagonal_line, 'r', linestyle='--')
ax[0,1].text(-0.02, 0.9, f"Mean Absolute Error on test set: {mae_cnn:.4f}", style='italic',fontweight='bold',bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 4})
ax[0,1].text(-0.02, 0.8, f"Loss on test set: {cnn_loss:.4f}", style='italic',fontweight='bold',bbox={'facecolor': 'cyan', 'alpha': 0.5, 'pad': 4})

test_soh_length = np.array(list(range(0,len(y_test[:,1]))))
ax[1,1].plot(test_soh_length,y_test[:,1],'tab:purple',linestyle='--',label='Dataset')
ax[1,1].plot(test_soh_length,y_pred_soh_cnn,'tab:green',linestyle='--',label="Predicted")
ax[1,1].set(xlabel="Samples",ylabel="SOH")
ax[1,1].legend(loc='upper right')

# Display the plot
plt.tight_layout()
plt.show()
