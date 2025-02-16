import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
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
y_soc = df['SoC Percentage'].values

# Train-test split
X_train, X_test, y_soc_train, y_soc_test = train_test_split(
    X, y_soc, test_size=0.2, random_state=42
)

# Reshape the data for GRU input (3D input: [samples, timesteps, features])
X_train_rnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_rnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Define GRU model
gru_model = Sequential([
    GRU(50, input_shape=(4, 1), return_sequences=True),  # GRU layer with 50 units
    GRU(50),  # Another GRU layer (return_sequences=False by default for the final output)
    Dense(1)  # Output layer for predicting SoC percentage
])

gru_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the GRU model
gru_model.fit(X_train_rnn, y_soc_train, epochs=50, batch_size=4, verbose=2)

# Evaluate the GRU model
gru_loss, gru_mae = gru_model.evaluate(X_test_rnn, y_soc_test)
print(f'GRU MAE: {gru_mae}')

# Predict with the GRU model
y_pred_gru = gru_model.predict(X_test_rnn).flatten()

# Calculate MAE for predictions
mae_gru = mean_absolute_error(y_soc_test, y_pred_gru)
print(f'GRU MAE: {mae_gru}')

# Create a scatter plot of true SOC values vs predicted SOC values
fig, ax = plt.subplots(2, figsize=(10, 8))

# Scatter plot of True vs Predicted SOC
ax[0].scatter(y_soc_test, y_pred_gru)
ax[0].set(xlabel="True SOC Values", ylabel="Predicted SOC Values")
ax[0].set_title("True SOC vs Predicted SOC (GRU Model)", fontweight='bold')

# Add a diagonal line to indicate perfect predictions
diagonal_line = np.linspace(0, 1, 100)
ax[0].plot(diagonal_line, diagonal_line, 'r', linestyle='--')

# Add text box for MAE and Loss
ax[0].text(0.02, 0.9, f"Mean Absolute Error: {mae_gru:}", style='italic', fontweight='bold', 
           bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 4}, transform=ax[0].transAxes)
ax[0].text(0.02, 0.8, f"Loss: {gru_loss:}", style='italic', fontweight='bold', 
           bbox={'facecolor': 'cyan', 'alpha': 0.5, 'pad': 4}, transform=ax[0].transAxes)

# Line plot for test SOC and predicted SOC
test_soc_length = np.array(list(range(0, len(y_soc_test))))
ax[1].plot(test_soc_length, y_soc_test, 'tab:red', label='Actual SoC Percentage')
ax[1].plot(test_soc_length, y_pred_gru, 'tab:green', label='Predicted SoC Percentage (GRU)')
ax[1].set(xlabel="Samples", ylabel="SOC")
ax[1].legend(loc='upper right')

# Display the plot
plt.tight_layout()
plt.show()
