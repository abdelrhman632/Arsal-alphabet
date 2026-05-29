import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
import pickle

MAX_FRAMES = 60

df = pd.read_csv("sign_data.csv")

print(df.head())
print(df.shape)
print(df.columns)

feature_columns = [
    col for col in df.columns 
    if col not in ["label", "video_id", "frame_id"]
]

X_sequences = []
y_labels = []

for video_id, group in df.groupby("video_id"):
    group = group.sort_values("frame_id")

    X_video = group[feature_columns].values
    y_video = group["label"].iloc[0]

    X_sequences.append(X_video)
    y_labels.append(y_video)

print(len(X_sequences))
print(len(y_labels))
print(X_sequences[0].shape)
print(y_labels[0])

X = pad_sequences(
    X_sequences,
    maxlen=MAX_FRAMES,
    dtype="float32",
    padding="post",
    truncating="post"
)

print(X.shape)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_labels)

y = to_categorical(y_encoded)

print(y_encoded[:10])
print(y.shape)
print(label_encoder.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

model = Sequential()

model.add(Masking(mask_value=0.0, input_shape=(60, 134)))

model.add(LSTM(128))

model.add(Dropout(0.3))

model.add(Dense(12, activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=16
)

loss, accuracy = model.evaluate(X_test, y_test)

print("Test loss:", loss)
print("Test accuracy:", accuracy)

model.save("arabic_sign_lstm_model.h5")

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)