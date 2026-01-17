import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Parameters
SEQ_LEN = 24   # use last 24 timesteps (2 hours)
TEST_SPLIT = 0.2
FILE = "Combined-Dataset-with-timestamps.xlsx"

# Load data
df = pd.read_excel(FILE)
df = df.sort_values("timestamp")  # ensure chronological order

# Prepare features and labels
features = df.drop(columns=["timestamp", "isAttacked"]).values  # numeric inputs
labels = df["isAttacked"].values  # target

# Scale features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Create sequences for RNN input
def create_sequences(data, targets, seq_len=SEQ_LEN):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])  # sequence of previous records
        y.append(targets[i])           # label at current time
    return np.array(X), np.array(y)

X, y = create_sequences(features, labels, SEQ_LEN)
print("Sequence dataset shape:", X.shape, y.shape)

# Chronological train/test split
split = int(len(X) * (1 - TEST_SPLIT))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
print("Train:", X_train.shape, " Test:", X_test.shape)

# LSTM model builder
def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))  # binary output

    model.compile(loss="binary_crossentropy",
        optimizer=Adam(0.001),
        metrics=["accuracy"])
    return model

# GRU model builder
def build_gru(input_shape):
    model = Sequential()
    model.add(GRU(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(GRU(32))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy",
        optimizer=Adam(0.001),
        metrics=["accuracy"])
    return model

# Training settings
callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]

print("\nTraining LSTM Model...")
lstm = build_lstm((SEQ_LEN, X.shape[2]))
history_lstm = lstm.fit(
    X_train, y_train,
    validation_split=0.1,  # small holdout from training set
    epochs=30,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

print("\nTraining GRU Model...")
gru = build_gru((SEQ_LEN, X.shape[2]))
history_gru = gru.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=30,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# Evaluate models on test set
print("\nLSTM Evaluation")
y_pred_lstm = (lstm.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, y_pred_lstm))
print(confusion_matrix(y_test, y_pred_lstm))

print("\nGRU Evaluation")
y_pred_gru = (gru.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, y_pred_gru))
print(confusion_matrix(y_test, y_pred_gru))

print("\nDone!")
