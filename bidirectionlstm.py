from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['sentence'])

sequences = tokenizer.texts_to_sequences(df['sentence'])
padded_sequences = pad_sequences(sequences, maxlen=100, padding='post')
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Permute, Multiply, Lambda, Flatten
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

def attention_mechanism(W_size):
    def attention(inputs):
        hidden_states, W = inputs
        hidden_states_permuted = Permute((2, 1))(hidden_states)
        attention = Multiply()([W, hidden_states_permuted])
        attention = Lambda(lambda x: K.sum(x, axis=2))(attention)
        attention = Dense(W_size, activation='softmax')(attention)
        return attention
    return attention

input_layer = Input(shape=(100,))
embedding_layer = Embedding(10000, 128, input_length=100)(input_layer)
bi_lstm = Bidirectional(LSTM(64, return_sequences=True))(embedding_layer)

# Attention mechanism
W = Dense(128, activation='tanh')(bi_lstm)
attention_layer = attention_mechanism(100)([bi_lstm, W])
merged = Multiply()([bi_lstm, Permute((2, 1))(attention_layer)])
flattened = Flatten()(merged)

output_layer = Dense(1, activation='sigmoid')(flattened)

model = Model(inputs=input_layer, outputs=[output_layer, attention_layer])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



from sklearn.model_selection import train_test_split

labels = df['contains_sensitive_data'].values
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2)

model.fit(X_train, [y_train, np.zeros(y_train.shape)], epochs=5, validation_data=(X_test, [y_test, np.zeros(y_test.shape)]))
from sklearn.model_selection import train_test_split

labels = df['contains_sensitive_data'].values
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2)

model.fit(X_train, [y_train, np.zeros(y_train.shape)], epochs=5, validation_data=(X_test, [y_test, np.zeros(y_test.shape)]))
def predict_and_score_words(sentence):
    sequence = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(sequence, maxlen=100, padding='post')
    prediction, attention_scores = model.predict(padded)

    word_scores = []
    for i, score in enumerate(attention_scores[0]):
        if i < len(sequence[0]):
            word = tokenizer.index_word[sequence[0][i]]
            word_scores.append((word, score))

    return prediction[0][0], word_scores

# Example usage
sentence = "My credit card number is 1234-5678-9012-3456"
sentence_prediction, word_scores = predict_and_score_words(sentence)
print(f"Sentence Prediction: {sentence_prediction}")
print("Word Scores:")
for word, score in word_scores:
    print(f"{word}: {score}")



