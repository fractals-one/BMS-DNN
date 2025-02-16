
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Load the dataset
file_path = '/mnt/data/soc_processed_data.csv'
df = pd.read_csv(file_path)

# Display the dataset structure
print(df.head())

# Drop the unnamed column (if necessary)
df.drop(columns=['Unnamed: 0'], inplace=True)

# Define the nominal capacity of the battery (Q_nom)
Q_nom = 2.2  # Example value; you should update this with the correct value

# SOC calculation using equation: SOC% = (Qn / Qm) * 100
df['SOC_Percent'] = (df['SoC Capacity'] / df['SoC Capacity'].max()) * 100

# SOH calculation using equation: SOH% = (Qm / Q_nom) * 100
df['SOH_Percent'] = (df['SoC Capacity'] / Q_nom) * 100

# Display the updated dataset
print(df[['SoC Capacity', 'SOC_Percent', 'SOH_Percent']].head())

# Prepare input features (Current, Voltage, Temperature) and target (SOC, SOH)
X = df[['Current', 'Voltage', 'Temperature']].values
y_soc = df['SOC_Percent'].values
y_soh = df['SOH_Percent'].values

# Scale the input features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train_soc, y_test_soc, y_train_soh, y_test_soh = train_test_split(
    X_scaled, y_soc, y_soh, test_size=0.2, random_state=42
)

# Reshape input for LSTM: [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Build the LSTM model for SOC prediction
model_soc = Sequential()
model_soc.add(LSTM(50, return_sequences=True, input_shape=(1, X_train.shape[2])))
model_soc.add(LSTM(50))
model_soc.add(Dense(25, activation='relu'))
model_soc.add(Dense(1))

# Compile the model
model_soc.compile(optimizer='adam', loss='mean_squared_error')

# Train the SOC model
model_soc.fit(X_train, y_train_soc, epochs=20, batch_size=32)

# Predict on test data
y_pred_soc = model_soc.predict(X_test)

# Build the LSTM model for SOH prediction
model_soh = Sequential()
model_soh.add(LSTM(50, return_sequences=True, input_shape=(1, X_train.shape[2])))
model_soh.add(LSTM(50))
model_soh.add(Dense(25, activation='relu'))
model_soh.add(Dense(1))

# Compile the model
model_soh.compile(optimizer='adam', loss='mean_squared_error')

# Train the SOH model
model_soh.fit(X_train, y_train_soh, epochs=20, batch_size=32)

# Predict on test data
y_pred_soh = model_soh.predict(X_test)

# Plot the results for SOC
plt.figure(figsize=(10,6))
plt.plot(y_test_soc, label='True SOC')
plt.plot(y_pred_soc, label='Predicted SOC')
plt.title('SOC Prediction')
plt.xlabel('Samples')
plt.ylabel('SOC Percentage')
plt.legend()
plt.show()

# Plot the results for SOH
plt.figure(figsize=(10,6))
plt.plot(y_test_soh, label='True SOH')
plt.plot(y_pred_soh, label='Predicted SOH')
plt.title('SOH Prediction')
plt.xlabel('Samples')
plt.ylabel('SOH Percentage')
plt.legend()
plt.show()
