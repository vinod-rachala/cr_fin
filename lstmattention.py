from faker import Faker
import pandas as pd
import numpy as np

fake = Faker()

def create_dataset(size):
    data = []
    for _ in range(size):
        record = {
            "credit_card_number": fake.credit_card_number(),
            "name": fake.name(),
            "address": fake.address(),
            "email": fake.email(),
            "ssn": fake.ssn(),
            "phone_number": fake.phone_number(),
            "birthdate": fake.date_of_birth(),
            "medical_info": fake.text(),
            "tax_id": fake.itin(),
            # ... other fields ...
            "sentence": fake.sentence(),
            "contains_sensitive_data": np.random.randint(0, 2)
        }
        data.append(record)

    return pd.DataFrame(data)

df = create_dataset(1000000)


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df['sentence'])

sequences = tokenizer.texts_to_sequences(df['sentence'])
padded_sequences = pad_sequences(sequences, maxlen=100)


from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Permute, Multiply, Lambda, Flatten, concatenate
from tensorflow.keras.models import Model
import tensorflow as tf

def attention_mechanism(inputs, units):
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Dense(units, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

input_layer = Input(shape=(100,))
embedding_layer = Embedding(10000, 128, input_length=100)(input_layer)
lstm_layer = LSTM(64, return_sequences=True)(embedding_layer)
attention_layer = attention_mechanism(lstm_layer, 100)
attention_flatten = Flatten()(attention_layer)
output_layer = Dense(1, activation='sigmoid')(attention_flatten)

model = Model(inputs=[input_layer], outputs=output_layer)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



from sklearn.model_selection import train_test_split

X = padded_sequences
y = df['contains_sensitive_data'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))


def predict_and_score(sentence):
    sequence = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(sequence, maxlen=100)
    prediction = model.predict(padded)[0][0]

    # Extract attention vector to get the scores for each word
    attention_model = Model(inputs=model.input, outputs=model.get_layer('attention_vec').output)
    attention_output = attention_model.predict(padded)

    word_attention = attention_output[0].flatten()[:len(sequence[0])]
    words = [tokenizer.index_word.get(i, 'UNK') for i in sequence[0]]

    return prediction, list(zip(words, word_attention))

# Example usage
test_sentence = "My credit card number is 1234-5678-9012-3456


sentence_prediction, word_scores = predict_and_score(test_sentence)
print(f"Sentence Prediction (Probability of containing sensitive data): {sentence_prediction}")
for word, score in word_scores:
print(f"{word}: {score}")






