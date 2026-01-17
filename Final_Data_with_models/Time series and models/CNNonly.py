import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# -----------------------------
# PARAMETERS
# -----------------------------
SEQ_LEN = 36
BATCH_SIZE = 32
EPOCHS = 40
LR = 0.0008
TEST_SPLIT = 0.2
FILE = "Combined-Dataset-with-timestamps.xlsx"

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_excel(FILE)
df = df.sort_values("timestamp")

features = df.drop(columns=["timestamp", "isAttacked"]).values
labels = df["isAttacked"].values

# Scale features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# -----------------------------
# CREATE SEQUENCES
# -----------------------------
def create_sequences(data, labels, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(labels[i])
    return np.array(X), np.array(y)

X, y = create_sequences(features, labels, SEQ_LEN)

# Chronological split
split = int(len(X) * (1 - TEST_SPLIT))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -----------------------------
# PURE CNN MODEL (NO LSTM)
# -----------------------------
def build_cnn_model(input_shape):

    model = Sequential()

    # CNN block 1
    model.add(Conv1D(64, kernel_size=3, activation="relu", padding="same", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))

    # CNN block 2
    model.add(Conv1D(128, kernel_size=3, activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))

    # CNN block 3
    model.add(Conv1D(256, kernel_size=3, activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    # Flatten and classify
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(LR),
        metrics=["accuracy"]
    )

    return model

model = build_cnn_model((SEQ_LEN, X.shape[2]))

# -----------------------------
# TRAIN MODEL
# -----------------------------
callbacks = [
    EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
]

print("\nTraining Pure CNN Model...")
model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# -----------------------------
# EVALUATION
# -----------------------------
print("\n=== PURE CNN MODEL EVALUATION ===")
y_pred = (model.predict(X_test) > 0.5).astype(int).ravel()

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nDone!")
