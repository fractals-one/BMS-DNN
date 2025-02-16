import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, RepeatVector, TimeDistributed, Masking
from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow import keras
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Load the dataset
data = pd.read_csv("/home/ramanujan/git_repos/BMS-DNN/BMS/dataset/soc_processed_data.csv")
data = data.dropna()

# Split the dataset into features and labels
X = data.drop(["SoH Percentage","SoC Percentage"], axis=1)
y_soc = data["SoC Percentage"]
y_soh = data["SoH Percentage"]

# Split the data into train and test sets
X_train, X_test, y_train_soc, y_test_soc, y_train_soh, y_test_soh = train_test_split(X, y_soc,y_soh, test_size=0.2, random_state=42)
# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Define the DNN model
model = Sequential([
    Dense(128, activation="relu", input_dim=X_train.shape[1]),
    Dropout(0.2),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dropout(0.2),
    Dense(8, activation="relu"),
    Dense(2, activation="sigmoid")
])

# Compile the model
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

# Train the model
history = model.fit(X_train_scaled, [y_train_soc, y_train_soh], epochs=150, batch_size=25, verbose=1, validation_split=0.2)

# Evaluate the model on the test set
loss, mae = model.evaluate(X_test_scaled, [y_test_soc, y_test_soh])
print(f"Mean Absolute Error on test set: {mae:.4f}")
print(f"Loss on test set: {loss:.4f}")

# Make predictions on the test set
y_pred_soc, y_pred_soh = model.predict(X_test_scaled).T
model.summary()

# Create a scatter plot of true SOC values vs predicted SOC values
fig, ax = plt.subplots(2,2)
ax[0,0].scatter(y_test_soc, y_pred_soc)
ax[0,0].set(xlabel="True SOC Values",ylabel="Predicted SOC Values")
ax[0,0].set_title("True SOC vs Predicted SOC",fontweight='bold')

# Add a diagonal line to the plot to indicate perfect predictions
diagonal_line = np.linspace(0, 1, 100)
ax[0,0].plot(diagonal_line, diagonal_line, 'r', linestyle='--')
ax[0,0].text(-0.02, 0.9, f"Mean Absolute Error on test set: {mae:.4f}", style='italic',fontweight='bold',bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 4})
ax[0,0].text(-0.02, 0.8, f"Loss on test set: {loss:.4f}", style='italic',fontweight='bold',bbox={'facecolor': 'cyan', 'alpha': 0.5, 'pad': 4})

test_soc_length = np.array(list(range(0,len(y_test_soc))))
ax[1,0].plot(test_soc_length,y_test_soc,'tab:red',linestyle='--',label='Dataset')
ax[1,0].plot(test_soc_length,y_pred_soc,'tab:green',linestyle='--',label="Predicted")
ax[1,0].set(xlabel="Samples",ylabel="SOC")
ax[1,0].legend(loc='upper right')

ax[0,1].scatter(y_test_soh, y_pred_soh)
ax[0,1].set(xlabel="True SOH Values",ylabel="Predicted SOH Values")
ax[0,1].set_title("True SOH vs Predicted SOH",fontweight='bold')

# Add a diagonal line to the plot to indicate perfect predictions
diagonal_line = np.linspace(0, 1, 100)
ax[0,1].plot(diagonal_line, diagonal_line, 'r', linestyle='--')
ax[0,1].text(-0.02, 0.9, f"Mean Absolute Error on test set: {mae:.4f}", style='italic',fontweight='bold',bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 4})
ax[0,1].text(-0.02, 0.8, f"Loss on test set: {loss:.4f}", style='italic',fontweight='bold',bbox={'facecolor': 'cyan', 'alpha': 0.5, 'pad': 4})

test_soh_length = np.array(list(range(0,len(y_test_soh))))
ax[1,1].plot(test_soh_length,y_test_soh,'tab:purple',linestyle='--',label='Dataset')
ax[1,1].plot(test_soh_length,y_pred_soh,'tab:green',linestyle='--',label="Predicted")
ax[1,1].set(xlabel="Samples",ylabel="SOH")
ax[1,1].legend(loc='upper right')


# Display the plot
plt.show()

# Save the model to a file
model.save("/home/ramanujan/git_repos/BMS-DNN/BMS/BMS_DNN/model/dnn_model.keras")
