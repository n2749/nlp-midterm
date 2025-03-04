from tensorflow.keras.layers import SimpleRNN, Dense, Embedding
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import sequence
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


def train_RNN(x_train, y_train, x_valid, y_valid, vocab_size=5000,
              embed_len=32, input_length=400, batch_size=64, epochs=5,
              verbose=1):
    MODEL_TITLE = "Simple_RNN"
    # Creating a RNN model
    model = Sequential(name=MODEL_TITLE)
    model.add(Embedding(vocab_size, embed_len, input_length=input_length))
    model.add(SimpleRNN(128, activation='tanh', return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))

    # Compiling model
    model.compile(
        loss="binary_crossentropy",
        optimizer='adam',
        metrics=['accuracy']
    )

    # Training the model
    history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=verbose,
                            validation_data=(x_valid, y_valid))
    
    return model, history


def plot_loss(history):
    loss = history.history['loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('task2_loss.png')


def main():
    x_train, y_train, x_test, y_test = get_dataset()
    x_train_, y_train_, x_valid, y_valid = get_subdataset(x_train, y_train)
    model, history = train_RNN(x_train_, y_train_, x_valid, y_valid)

    print("Simple_RNN Score---> ", model.evaluate(x_test, y_test, verbose=0))
    plot_loss(history)


if __name__ == "__main__":
    main()
