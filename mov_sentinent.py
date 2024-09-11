import pandas as pd
import string
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load and preprocess the dataset
df = pd.read_csv('IMDB_Dataset.csv')

# Clean text function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text

df['review'] = df['review'].apply(clean_text)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Shuffle and split the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
split_index = int(len(df) * 0.8)
X_train, y_train = df['review'][:split_index], df['sentiment'][:split_index]
X_test, y_test = df['review'][split_index:], df['sentiment'][split_index:]

# Tokenization and padding
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=200)
X_test_pad = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=200)

# Build and compile the model
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=200),
    LSTM(128),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train and evaluate the model
history = model.fit(X_train_pad, y_train, epochs=2, validation_data=(X_test_pad, y_test), batch_size=64)
loss, accuracy = model.evaluate(X_test_pad, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Make and display predictions
y_pred = (model.predict(X_test_pad) > 0.5).astype("int32")
for i in range(5):
    print(f"Review: {X_test.iloc[i]}")
    print(f"True sentiment: {y_test.iloc[i]}")
    print(f"Predicted sentiment: {y_pred[i][0]}")
    print()
