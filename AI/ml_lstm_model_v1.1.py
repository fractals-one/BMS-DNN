import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from numpy import hstack
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, RepeatVector, TimeDistributed, Masking
from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
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
X_train_lstm = X_train_scaled.reshape(1,X_train_scaled.shape[0],X_train_scaled.shape[1])
y_train_soc_lstm = y_train_soc.to_numpy().reshape(1,y_train_soc.shape[0])
y_train_soh_lstm = y_train_soh.to_numpy().reshape(1,y_train_soh.shape[0])
X_test_lstm = X_test_scaled.reshape(1,X_test_scaled.shape[0],X_test_scaled.shape[1])
y_test_soc_lstm = y_test_soc.to_numpy().reshape(y_test_soc.shape[0])
y_test_soh_lstm = y_test_soh.to_numpy().reshape(y_test_soh.shape[0])
y_train_lstm = hstack((y_train_soc,y_train_soh))
y_test_lstm = hstack((y_test_soc_lstm,y_test_soh_lstm))

# Define the LSTM DNN model
optimizer_lstm = Adam(0.003)#Change to 0.001 from 0.00003

model_lstm = Sequential()
model_lstm.add(LSTM(256, activation='sigmoid',
                return_sequences=True,
                input_shape=(X_train.shape[0],X_train.shape[1])
                ))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(256, activation='sigmoid', return_sequences=True))
model_lstm.add(Dropout(0.2))
#model_lstm.add(LSTM(128, activation='relu',return_sequences=True))
#model_lstm.add(Dropout(0.2))
#model_lstm.add(Dense(256, activation='relu'))
#model_lstm.add(LSTM(128, activation='relu',return_sequences=True))
#model_lstm.add(Dense(64, activation='relu'))
#model_lstm.add(LSTM(8, activation='relu',return_sequences=True))
model_lstm.add(Dense(2, activation='linear'))
model_lstm.summary()

model_lstm.compile(optimizer=optimizer_lstm, loss='mse', metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])

es = EarlyStopping(monitor='val_loss', patience=50)
#mc = ModelCheckpoint(data_path + 'results/trained_model/%s_best.h5' % experiment_name, 
#                             save_best_only=True, 
#                             monitor='val_loss')


history_lstm = model_lstm.fit(X_train_lstm, y_train_lstm ,
                                epochs=250, #change to 100 from 1000
                                batch_size=32, 
                                verbose=2,
                                #validation_split=0.2,
                                callbacks = [es]
                               )
model_lstm.summary()
# Evaluate the model on the test set
mae_lstm = model_lstm.evaluate(X_test_lstm, y_test_lstm)
print(mae_lstm)
print(f"Mean Absolute Error on test set: {mae_lstm[1]:.4f}")
print(f"Loss on test set: {mae_lstm[0]:.4f}")

# Save the model to a file
model_lstm.save("/home/ramanujan/git_repos/BMS-DNN/BMS/BMS_DNN/model/dnn_lstm_model.keras")

#model_lstm = load_model("/home/ramanujan/git_repos/BMS-DNN/BMS/BMS_DNN/model/dnn_lstm_model.keras")
# Make predictions on the test set
y_pred_soc_lstm, y_pred_soh_lstm = model_lstm.predict(X_test_lstm)

"""
# Create a scatter plot of true SOC values vs predicted SOC values
fig, ax = plt.subplots(2)
ax[0].scatter(y_test_lstm[0], y_pred_lstm[0,:,0])
ax[0].set(xlabel="True SOC Values",ylabel="Predicted SOC Values")
ax[0].set_title("True SOC vs Predicted SOC",fontweight='bold')

# Add a diagonal line to the plot to indicate perfect predictions
diagonal_line = np.linspace(0, 1, 100)
ax[0].plot(diagonal_line, diagonal_line, 'r', linestyle='--')
ax[0].text(-0.02, 0.9, f"Mean Absolute Error on test set: {mae_lstm[1]:.4f}", style='italic',fontweight='bold',bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 4})
ax[0].text(-0.02, 0.8, f"Loss on test set: {mae_lstm[0]:.4f}", style='italic',fontweight='bold',bbox={'facecolor': 'cyan', 'alpha': 0.5, 'pad': 4})

test_soc_length = np.array(list(range(0,len(y_test_lstm[0]))))
ax[1].plot(test_soc_length,y_test_lstm[0],'tab:red',label='Dataset')
ax[1].plot(test_soc_length,y_pred_lstm[0,:,0],'tab:green',label="Predicted")
ax[1].set(xlabel="Samples",ylabel="SOC")
ax[1].legend(loc='upper right')
"""
# Create a scatter plot of true SOC values vs predicted SOC values
fig, ax = plt.subplots(2,2)
ax[0,0].scatter(y_test_soc_lstm, y_pred_soc_lstm)
ax[0,0].set(xlabel="True SOC Values",ylabel="Predicted SOC Values")
ax[0,0].set_title("True SOC vs Predicted SOC",fontweight='bold')

# Add a diagonal line to the plot to indicate perfect predictions
diagonal_line = np.linspace(0, 1, 100)
ax[0,0].plot(diagonal_line, diagonal_line, 'r', linestyle='--')
ax[0,0].text(-0.02, 0.9, f"Mean Absolute Error on test set: {mae:.4f}", style='italic',fontweight='bold',bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 4})
ax[0,0].text(-0.02, 0.8, f"Loss on test set: {loss:.4f}", style='italic',fontweight='bold',bbox={'facecolor': 'cyan', 'alpha': 0.5, 'pad': 4})

test_soc_length = np.array(list(range(0,len(y_test_soc_lstm))))
ax[1,0].plot(test_soc_length,y_test_soc_lstm,'tab:red',linestyle='--',label='Dataset')
ax[1,0].plot(test_soc_length,y_pred_soc_lstm,'tab:green',linestyle='--',label="Predicted")
ax[1,0].set(xlabel="Samples",ylabel="SOC")
ax[1,0].legend(loc='upper right')

ax[0,1].scatter(y_test_soh_lstm, y_pred_soh_lstm)
ax[0,1].set(xlabel="True SOH Values",ylabel="Predicted SOH Values")
ax[0,1].set_title("True SOH vs Predicted SOH",fontweight='bold')

# Add a diagonal line to the plot to indicate perfect predictions
diagonal_line = np.linspace(0, 1, 100)
ax[0,1].plot(diagonal_line, diagonal_line, 'r', linestyle='--')
ax[0,1].text(-0.02, 0.9, f"Mean Absolute Error on test set: {mae:.4f}", style='italic',fontweight='bold',bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 4})
ax[0,1].text(-0.02, 0.8, f"Loss on test set: {loss:.4f}", style='italic',fontweight='bold',bbox={'facecolor': 'cyan', 'alpha': 0.5, 'pad': 4})

test_soh_length = np.array(list(range(0,len(y_test_soh_lstm))))
ax[1,1].plot(test_soh_length,y_test_soh_lstm,'tab:purple',linestyle='--',label='Dataset')
ax[1,1].plot(test_soh_length,y_pred_soh_lstm,'tab:green',linestyle='--',label="Predicted")
ax[1,1].set(xlabel="Samples",ylabel="SOH")
ax[1,1].legend(loc='upper right')

# Display the plot
plt.show()

