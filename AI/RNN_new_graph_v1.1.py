import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, GRU
from sklearn.metrics import mean_absolute_error

# Load and prepare the data
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

# Reshape the data for GRU input (3D input: [samples, timesteps, features])
X_train_rnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_rnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Define GRU model
gru_model = Sequential([
    GRU(50, input_shape=(4, 1), return_sequences=True),  # GRU layer with 50 units
    GRU(50),  # Another GRU layer (return_sequences=False by default for the final output)
    Dense(2)  # Output layer for predicting SoC percentage
])

gru_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the GRU model
gru_model.fit(X_train_rnn, y_train, epochs=50, batch_size=4, verbose=2)

# Save the model to a file
gru_model.save("/home/ramanujan/git_repos/BMS-DNN/BMS/BMS_DNN/model/dnn_gru_model.keras")

# Retrive the saved model, useful during debug of the plotting section
#gru_model = load_model("/home/ramanujan/git_repos/BMS-DNN/BMS/BMS_DNN/model/dnn_gru_model.keras")

# Evaluate the GRU model
gru_loss, gru_mae = gru_model.evaluate(X_test_rnn, y_test)
print(f'GRU MAE: {gru_mae}')

# Predict with the GRU model
y_pred_soc_gru, y_pred_soh_gru = gru_model.predict(X_test_rnn).T

# Calculate MAE for predictions
mae_gru = mean_absolute_error(y_test.T, [y_pred_soc_gru,y_pred_soh_gru])
print(f'GRU MAE: {mae_gru}')

# Create a scatter plot of true SOC values vs predicted SOC values
fig, ax = plt.subplots(2,2)
ax[0,0].scatter(y_test[:,0], y_pred_soc_gru)
ax[0,0].set(xlabel="True SOC Values",ylabel="Predicted SOC Values")
ax[0,0].set_title("True SOC vs Predicted SOC",fontweight='bold')

# Add a diagonal line to the plot to indicate perfect predictions
diagonal_line = np.linspace(0, 1, 100)
ax[0,0].plot(diagonal_line, diagonal_line, 'r', linestyle='--')
ax[0,0].text(-0.02, 0.9, f"Mean Absolute Error on test set: {mae_gru:.4f}", style='italic',fontweight='bold',bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 4})
ax[0,0].text(-0.02, 0.8, f"Loss on test set: {gru_loss:.4f}", style='italic',fontweight='bold',bbox={'facecolor': 'cyan', 'alpha': 0.5, 'pad': 4})

test_soc_length = np.array(list(range(0,len(y_test[:,0]))))
ax[1,0].plot(test_soc_length,y_test[:,0],'tab:red',linestyle='--',label='Dataset')
ax[1,0].plot(test_soc_length,y_pred_soc_gru,'tab:green',linestyle='--',label="Predicted")
ax[1,0].set(xlabel="Samples",ylabel="SOC")
ax[1,0].legend(loc='upper right')

ax[0,1].scatter(y_test[:,1], y_pred_soh_gru)
ax[0,1].set(xlabel="True SOH Values",ylabel="Predicted SOH Values")
ax[0,1].set_title("True SOH vs Predicted SOH",fontweight='bold')

# Add a diagonal line to the plot to indicate perfect predictions
diagonal_line = np.linspace(0, 1, 100)
ax[0,1].plot(diagonal_line, diagonal_line, 'r', linestyle='--')
ax[0,1].text(-0.02, 0.9, f"Mean Absolute Error on test set: {mae_gru:.4f}", style='italic',fontweight='bold',bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 4})
ax[0,1].text(-0.02, 0.8, f"Loss on test set: {gru_loss:.4f}", style='italic',fontweight='bold',bbox={'facecolor': 'cyan', 'alpha': 0.5, 'pad': 4})

test_soh_length = np.array(list(range(0,len(y_test[:,1]))))
ax[1,1].plot(test_soh_length,y_test[:,1],'tab:purple',linestyle='--',label='Dataset')
ax[1,1].plot(test_soh_length,y_pred_soh_gru,'tab:green',linestyle='--',label="Predicted")
ax[1,1].set(xlabel="Samples",ylabel="SOH")
ax[1,1].legend(loc='upper right')

# Display the plot
plt.tight_layout()
plt.show()
