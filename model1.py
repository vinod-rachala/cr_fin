from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['sentence'])

sequences = tokenizer.texts_to_sequences(df['sentence'])
padded_sequences = pad_sequences(sequences, maxlen=100, padding='post')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional

model = Sequential([
    Embedding(10000, 64, input_length=100),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
from sklearn.model_selection import train_test_split

labels = df['contains_sensitive_data'].values
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2)

model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))



def predict_sensitive_data(sentence, tokenizer, model):
    sequence = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(sequence, maxlen=100, padding='post')
    prediction = model.predict(padded)
    return prediction[0][0]

# Example usage
sentence = "My credit card number is 1234-5678-9012-3456"
probability = predict_sensitive_data(sentence, tokenizer, model)
print(f"Probability of containing sensitive data: {probability}")
