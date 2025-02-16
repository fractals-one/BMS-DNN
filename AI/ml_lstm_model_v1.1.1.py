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
y = data[["SoC Percentage","SoH Percentage"]]

scaler = StandardScaler()
features_scaled = scaler.fit_transform(X)

def create_dataset(X, y, n_steps):
    Xs, ys = [], []
    for i in range(len(X) - n_steps):
        Xs.append(X[i:(i + n_steps)])
        ys.append(y[i + n_steps])
    return np.array(Xs), np.array(ys)

n_steps = 10  # Number of timesteps
X, y = create_dataset(features_scaled, y.values, n_steps)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Reshape X for LSTM [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

# Define the LSTM DNN model
optimizer_lstm = Adam(0.003)#Change to 0.001 from 0.00003

model_lstm = Sequential()
model_lstm.add(LSTM(256, activation='sigmoid',
                return_sequences=True,
                input_shape=(n_steps,X_train.shape[2])
                ))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(256, activation='sigmoid'))
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


history_lstm = model_lstm.fit(X_train ,y_train ,
                                epochs=250, #change to 100 from 1000
                                batch_size=32, 
                                verbose=1,
                                #validation_split=0.2,
                                callbacks = [es]
                               )
model_lstm.summary()

# Save the model to a file
model_lstm.save("/home/ramanujan/git_repos/BMS-DNN/BMS/BMS_DNN/model/dnn_lstm_model.keras")

# Retrive the saved model, useful during debug of the plotting section
#model_lstm = load_model("/home/ramanujan/git_repos/BMS-DNN/BMS/BMS_DNN/model/dnn_lstm_model.keras")

# Evaluate the model on the test set
mae_lstm = model_lstm.evaluate(X_test, y_test)
print(mae_lstm)
print(f"Mean Absolute Error on test set: {mae_lstm[1]:.4f}")
print(f"Loss on test set: {mae_lstm[0]:.4f}")

# Make predictions on the test set
y_pred_soc_lstm,y_pred_soh_lstm = model_lstm.predict(X_test).T
z
# Create a scatter plot of true SOC values vs predicted SOC values
fig, ax = plt.subplots(2,2)
ax[0,0].scatter(y_test[:,0], y_pred_soc_lstm)
ax[0,0].set(xlabel="True SOC Values",ylabel="Predicted SOC Values")
ax[0,0].set_title("True SOC vs Predicted SOC",fontweight='bold')

# Add a diagonal line to the plot to indicate perfect predictions
diagonal_line = np.linspace(0, 1, 100)
ax[0,0].plot(diagonal_line, diagonal_line, 'r', linestyle='--')
ax[0,0].text(-0.02, 0.9, f"Mean Absolute Error on test set: {mae_lstm[1]:.4f}", style='italic',fontweight='bold',bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 4})
ax[0,0].text(-0.02, 0.8, f"Loss on test set: {mae_lstm[0]:.4f}", style='italic',fontweight='bold',bbox={'facecolor': 'cyan', 'alpha': 0.5, 'pad': 4})

test_soc_length = np.array(list(range(0,len(y_test[:,0]))))
ax[1,0].plot(test_soc_length,y_test[:,0],'tab:red',linestyle='--',label='Dataset')
ax[1,0].plot(test_soc_length,y_pred_soc_lstm,'tab:green',linestyle='--',label="Predicted")
ax[1,0].set(xlabel="Samples",ylabel="SOC")
ax[1,0].legend(loc='upper right')

ax[0,1].scatter(y_test[:,1], y_pred_soh_lstm)
ax[0,1].set(xlabel="True SOH Values",ylabel="Predicted SOH Values")
ax[0,1].set_title("True SOH vs Predicted SOH",fontweight='bold')

# Add a diagonal line to the plot to indicate perfect predictions
diagonal_line = np.linspace(0, 1, 100)
ax[0,1].plot(diagonal_line, diagonal_line, 'r', linestyle='--')
ax[0,1].text(-0.02, 0.9, f"Mean Absolute Error on test set: {mae_lstm[1]:.4f}", style='italic',fontweight='bold',bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 4})
ax[0,1].text(-0.02, 0.8, f"Loss on test set: {mae_lstm[0]:.4f}", style='italic',fontweight='bold',bbox={'facecolor': 'cyan', 'alpha': 0.5, 'pad': 4})

test_soh_length = np.array(list(range(0,len(y_test[:,1]))))
ax[1,1].plot(test_soh_length,y_test[:,1],'tab:purple',linestyle='--',label='Dataset')
ax[1,1].plot(test_soh_length,y_pred_soh_lstm,'tab:green',linestyle='--',label="Predicted")
ax[1,1].set(xlabel="Samples",ylabel="SOH")
ax[1,1].legend(loc='upper right')

# Display the plot
plt.show()

