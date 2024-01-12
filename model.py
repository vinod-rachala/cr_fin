import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load dataset
df = pd.read_csv('synthetic_pci_phi_dataset.csv')
sentences = df['sentence'].values
labels = df['label'].values

# Tokenize sentences
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
data = pad_sequences(sequences, maxlen=100)

# Split data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

# Build LSTM model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=50, input_length=100))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print(f"Test Accuracy: {accuracy}")
