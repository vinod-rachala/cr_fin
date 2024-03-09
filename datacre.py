import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np

def build_generator(seq_length, latent_dim):
    inputs = Input(shape=(seq_length, latent_dim))
    x = LSTM(128, return_sequences=True)(inputs)
    x = Dense(1, activation='tanh')(x)
    model = Model(inputs, x)
    return model

def build_discriminator(seq_length):
    inputs = Input(shape=(seq_length, 1))
    x = LSTM(64, return_sequences=True)(inputs)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, x)
    model.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])
    return model

latent_dim = 100
seq_length = 10  # Example sequence length

# Build and compile the discriminator
discriminator = build_discriminator(seq_length)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

# Build the generator
generator = build_generator(seq_length, latent_dim)

# The generator takes noise as input and generates sequences
z = Input(shape=(seq_length, latent_dim))
generated_seq = generator(z)

# For the combined model we will only train the generator
discriminator.trainable = False

# The discriminator takes generated sequences as input and determines validity
valid = discriminator(generated_seq)

# The combined model (stacked generator and discriminator)
combined = Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
