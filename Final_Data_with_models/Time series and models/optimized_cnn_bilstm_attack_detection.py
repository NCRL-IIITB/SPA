import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten
from tensorflow.keras.layers import BatchNormalization, Bidirectional, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# PARAMETERS
SEQ_LEN = 36
BATCH_SIZE = 32
EPOCHS = 40
LR = 0.0008
TEST_SPLIT = 0.2
FILE = "Combined-Dataset-with-timestamps.xlsx"

# LOAD DATA
df = pd.read_excel(FILE)
df = df.sort_values("timestamp")
features = df.drop(columns=["timestamp", "isAttacked"]).values
labels = df["isAttacked"].values
scaler = StandardScaler()
features = scaler.fit_transform(features)

# CREATE SEQUENCES
def create_sequences(data, labels, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(labels[i])
    return np.array(X), np.array(y)

X, y = create_sequences(features, labels, SEQ_LEN)
print("X shape:", X.shape, " y shape:", y.shape)

# SPLIT
split = int(len(X) * (1 - TEST_SPLIT))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
print("Train:", X_train.shape, "Test:", X_test.shape)

# MODEL
def build_optimized_model(input_shape):
    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, padding="same", activation="relu", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv1D(64, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Bidirectional(LSTM(48, return_sequences=True)))
    model.add(Dropout(0.25))
    model.add(Bidirectional(LSTM(24)))
    model.add(Dropout(0.3))

    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer=Adam(LR), metrics=["accuracy"])
    return model

model = build_optimized_model((SEQ_LEN, X.shape[2]))

# TRAIN
callbacks = [EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)]
print("\nTraining Optimized CNN + BiLSTM Model...")
model.fit(X_train, y_train, validation_split=0.1, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=1)

# EVAL
print("\n=== OPTIMIZED MODEL EVALUATION ===")
y_pred = (model.predict(X_test) > 0.5).astype(int)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nDone!")
