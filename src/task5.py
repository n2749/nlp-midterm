from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Embedding
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt


def get_dataset(vocab_size=5000, maxlen=400):
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    return x_train, y_train, x_test, y_test


def get_subdataset(x_train, y_train):
    x_valid, y_valid = x_train[:64], y_train[:64]
    x_train_, y_train_ = x_train[64:], y_train[64:]

    return x_train_, y_train_, x_valid, y_valid


def train_bilstm(x_train, y_train, x_valid, y_valid, vocab_size=5000,
                 embed_len=32, input_length=400, batch_size=64, epochs=5,
                 verbose=1):
    MODEL_TITLE = "Bidirectional_LSTM"
    # Defining GRU model
    model = Sequential(name=MODEL_TITLE)
    model.add(Embedding(vocab_size, embed_len))
    model.add(Bidirectional(LSTM(128, activation='tanh',
                                         return_sequences=False)))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(clipnorm=1.0),
        metrics=['accuracy']
    )

    # Training the model
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=verbose,
                        validation_data=(x_valid, y_valid))
    return model, history


def plot_loss(history, model_name=''):
    loss = history.history['loss']
    epochs = range(1, len(loss) + 1)
    plt.figure(figsize=(10, 7))
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.title(f'{model_name} Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    file_name = f'task5{model_name}.png' if model_name != '' else 'task5loss.png'
    plt.savefig(file_name)


def main():
    x_train, y_train, x_test, y_test = get_dataset()
    x_train_, y_train_, x_valid, y_valid = get_subdataset(x_train, y_train)

    model, history = train_bilstm(x_train_, y_train_, x_valid, y_valid)
    print("Bidirectional LSTM Score---> ", model.evaluate(x_test, y_test, verbose=0))
    plot_loss(history, 'bilstm')


if __name__ == "__main__":
    main()

