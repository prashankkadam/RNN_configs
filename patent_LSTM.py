# Setup IPython to show all the outputs from a single cell
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import sys
import gc
import random
from keras.preprocessing.text import Tokenizer
from sklearn.utils import shuffle
from tensorflow.python.client import device_lib
from IPython.core.interactiveshell import InteractiveShell
from keras.utils import get_file

InteractiveShell.ast_node_inteactivity = 'all'

warnings.filterwarnings('ignore', category=RuntimeWarning)

RANDOM_STATE = 50
EPOCHS = 150
BATCH_SIZE = 2048
TRAINING_LENGTH = 50
TRAIN_FRACTION = 0.7
LSTM_CELLS = 64
VERBOSE = 0
SAVE_MODEL = True

# print(device_lib.list_local_devices())

# Reading the data:
data = pd.read_csv('data/neural_network_patent_extract.csv', parse_dates=['patent_date'])

# Extract abstracts
original_abstracts = list(data['patent_abstract'])
# print(len(original_abstracts))
# print(data.head())

# Breif data exploration:
# print(data['patent_abstract'][100])

plt.style.use('fivethirtyeight')

data['year-month'] = [
    pd.datetime(year, month, 1) for year, month in zip(
        data['patent_date'].dt.year, data['patent_date'].dt.month)
]

monthly = data.groupby('year-month')['patent_number'].count().reset_index()

# monthly.set_index('year-month')['patent_number'].plot(figsize=(16, 8))
#
# plt.ylabel('Number of Patents')
# plt.xlabel('Date')
# plt.title('Neural Network Patent over Time')
# plt.show()

monthly.groupby(monthly['year-month'].dt.year)['patent_number'].sum().plot.bar(
    color='red', edgecolor='k', figsize=(12, 6))


# plt.xlabel('Year')
# plt.ylabel('Number of Patents')
# plt.title('Neural Network Patent by Year')
# plt.show()


def format_patent(patent):
    # Add spaces around punctuation
    patent = re.sub(r'(?<=[^\s0-9])(?=[.,;?])', r' ', patent)

    # Remove references to figures
    patent = re.sub(r'\((\d+)\)', r'', patent)

    # Remove double spaces
    patent = re.sub(r'\s\s', ' ', patent)
    return patent


def remove_spaces(patent):
    patent = re.sub(r'\s+([.,;?])', r'\1', patent)
    return patent


formatted = []

# Iterate through all the original abstracts
for a in original_abstracts:
    formatted.append(format_patent(a))

print(len(formatted))


# Convert text to sequences
def make_sequences(texts,
                   training_length=50,
                   lower=True,
                   filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'):
    # Turn a set of texts into sequence of intergers
    # Create tokenizer object and train on texts:
    tokenizer = Tokenizer(lower=lower, filters=filters)
    tokenizer.fit_on_texts(texts)

    # Create look-up dictionaries and reverse look-ups
    word_idx = tokenizer.word_index
    idx_word = tokenizer.index_word
    num_words = len(word_idx) + 1
    word_counts = tokenizer.word_counts

    print(f'There are {num_words} unique words.')

    # Convert text to sequences of integers
    sequences = tokenizer.texts_to_sequences(texts)

    # Limit the sequences with more than training length tokens
    seq_lengths = [len(x) for x in sequences]

    over_idx = [
        i for i, l in enumerate(seq_lengths) if l > (training_length + 20)
    ]

    new_texts = []
    new_sequences = []

    # Only keep sequences with more than training length tokens
    for i in over_idx:
        new_texts.append(texts[i])
        new_sequences.append(sequences[i])

    training_seq = []
    labels = []

    # Iterate through the sequence of tokens
    for seq in new_sequences:
        # Create multiple training examples from each sequence:
        for i in range(training_length, len(seq)):
            # Extract the features and labels:
            extract = seq[i - training_length:i + 1]

            # Set the features and labels:
            training_seq.append(extract[:-1])
            labels.append(extract[-1])

    print(f'There are {len(training_seq)} training sequences')

    # Return everything needed to set up the model
    return word_idx, idx_word, num_words, word_counts, new_texts, new_sequences, training_seq, labels


TRAINING_LENGTH = 50
filters = '!"#$%&()*+/:<=>@[\\]^_`{|}~\t\n'
word_idx, idx_word, num_words, word_counts, \
abstracts, sequences, features, labels = make_sequences(formatted, TRAINING_LENGTH,
                                                        lower=True, filters=filters)


# n = 3
# print(features[n][:10])


def find_answer(index):
    # Find the data corresponding to the features of index in the training data
    # Find features and label:
    feats = ' '.join(idx_word[i] for i in features[index])
    answer = idx_word[labels[index]]

    print('Features:', feats)
    print('\nLabel:', answer)


# find_answer(n)
# print(original_abstracts[0])

# print(sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:15])


# Encoding the labels:
def create_train_valid(features,
                       labels,
                       num_words,
                       train_fraction=TRAIN_FRACTION):
    # Create training and validation features and label
    # Randomly shuffle features and labels:
    features, labels = shuffle(features, labels, random_state=RANDOM_STATE)

    # Decide on number of samples for training
    train_end = int(train_fraction * len(labels))

    train_features = np.array(features[:train_end])
    valid_features = np.array(features[train_end:])

    train_labels = labels[:train_end]
    valid_labels = labels[train_end:]

    # Convert to arrays
    X_train, X_valid = np.array(train_features), np.array(valid_features)

    # Using int8 for memory savings
    y_train = np.zeros((len(train_labels), num_words), dtype=np.int8)
    y_valid = np.zeros((len(valid_labels), num_words), dtype=np.int8)

    # One hot encoding of states:
    for example_index, word_index in enumerate(train_labels):
        y_train[example_index, word_index] = 1

    for example_index, word_index in enumerate(valid_labels):
        y_valid[example_index, word_index] = 1

    # Memory Management
    import gc
    gc.enable()
    del features, labels, train_features, valid_features, train_labels, valid_labels
    gc.collect()

    return X_train, X_valid, y_train, y_valid


X_train, X_valid, y_train, y_valid = create_train_valid(features, labels, num_words)

print(X_train.shape)
print(y_train.shape)

# One hot encoding creates massive numpy arrays, doing some memory management
print(sys.getsizeof(y_train) / 1e9)


def check_sizes(gb_min=1):
    for x in globals():
        size = sys.getsizeof(eval(x)) / 1e9
        if size > gb_min:
            print(f'Object: {x:10}\tSize: {size} GB')
        else:
            print('Memory OK!')


check_sizes(gb_min=1)

# Predefined embeddings:
glove_vectors = 'data/glove.6B.zip'

# Download word embeddings if they are not present
if not os.path.exists(glove_vectors):
    glove_vectors = get_file('glove.6B.zip', 'http://nlp.stanford.edu/data/glove.6B.zip')
    os.system(f'unzip {glove_vectors}')

# Load the unzipped file
glove_vectors = 'data/glove.6B.100d.txt'
glove = np.loadtxt(glove_vectors, dtype='str', comments=None)
print(glove.shape)

vectors = glove[:, 1:].astype('float')
words = glove[:, 0]

del glove

print(vectors[100], words[100])

# Checking the shape of the vectors:
print(vectors.shape)

word_lookup = {word: vector for word, vector in zip(words, vectors)}

embedding_matrix = np.zeros((num_words, vectors.shape[1]))

not_found = 0

for i, word in enumerate(word_idx.keys()):
    # Lookup for word embeddings
    vector = word_lookup.get(word, None)

    # Record in matrix
    if vector is not None:
        embedding_matrix[1 + 1, :] = vector
    else:
        not_found += 1

print("There are {not_found} words without pre-trained embeddings")

gc.enable()
del vectors
gc.collect()

# Normalize and convert the nan to 0
embedding_matrix = embedding_matrix / \
                   np.linalg.norm(embedding_matrix, axis=1).reshape((-1, 1))

embedding_matrix = np.nan_to_num(embedding_matrix)


def find_closest(query, embedding_matrix, word_idx, idx_word, n=10):
    # Find the closest word to the query word on the embedding
    idx = word_idx.get(query, None)

    # Handle the case where the query is not in the vocab:
    if idx is None:
        print(f'{query} not found in vocab')
        return
    else:
        vec = embedding_matrix[idx]

        # Handle the case where the word doesn't have an embedding:
        if np.all(vec == 0):
            print(f'{query} has no pre-trained embeddings.')
            return
        else:
            # Calculate the distance between vectors and all others
            dists = np.dot(embedding_matrix, vec)

            # Sort indexes in reverse order
            idxs = np.argsort(dists)[::-1][:n]
            sorted_dists = dists[idxs]
            closest = [idx_word[i] for i in idxs]

    print(f'Query: {query}\n')
    max_len = max([len(i) for i in closest])

    # Print out the word and cosine distances:
    for word, dist in zip(closest, sorted_dists):
        print(f'Word: {word:15} Cosine Similarity: {round(dist, 4)}')


# Building the model:
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Embedding, Masking, Bidirectional
from keras.optimizers import Adam
from keras.utils import plot_model


# Make a word level RNN with options for pre-trained embeddings:
def make_word_level_model(num_words, embedding_matrix, lstm_cells=64,
                          trainable=False, lstm_layers=1, bi_direc=False):
    model = Sequential()

    # Map words to an embedding:
    if not trainable:
        model.add(
            Embedding(
                input_dim=num_words,
                output_dim=embedding_matrix.shape[1],
                weights=[embedding_matrix],
                trainable=False,
                mask_zero=True))
        model.add(Masking())
    else:
        model.add(
            Embedding(
                input_dim=num_words,
                output_dim=embedding_matrix.shape[1],
                weights=[embedding_matrix],
                trainable=True))

    # If multiple LSTM layers are to be added:
    if lstm_layers > 1:
        for i in range(lstm_layers - 1):
            model.add(
                LSTM(
                    lstm_cells,
                    return_sequences=True,
                    dropout=0.1,
                    recurrent_dropout=0.1))

    # Add final LSTM layer:
    if bi_direc:
        model.add(
            Bidirectional(
                LSTM(
                    lstm_cells,
                    return_sequences=True,
                    dropout=0.1,
                    recurrent_dropout=0.1)))
    else:
        model.add(
            LSTM(
                lstm_cells,
                return_sequences=True,
                dropout=0.1,
                recurrent_dropout=0.1))

    model.add(Dense(num_words, activation='relu'))

    # Dropout for regularization
    model.add(Dropout(0.5))

    # Output Layer
    model.add(Dense(num_words, activation='softmax'))

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model


model = make_word_level_model(
    num_words,
    embedding_matrix=embedding_matrix,
    lstm_cells=LSTM_CELLS,
    trainable=False,
    lstm_layers=1)

model.summary()

# Plotting the model:
from IPython.display import Image

model_name = 'pre-trained-rnn'
model_dir = '/models/'

plot_model(model, to_file=f'{model_dir}{model_name}.png', show_shapes=True)

Image(f'{model_dir}{model_name}'.png)

# Training the model:
# Callbacks:
# 1 - Early Stopping - Stop training when the validation loss no longer decreases
# 2 - Model Checkpoint - Save the best model on the disk

from keras.callbacks import EarlyStopping, ModelCheckpoint

BATCH_SIZE = 2048


def make_callbacks(model_name, save=SAVE_MODEL):
    # Make a list of callbacks for training:
    callbacks = [EarlyStopping(monitor='val_loss', patience=5)]

    if save:
        callbacks.append(
            ModelCheckpoint(
                f'{model_dir}{model_name}.h5',
                save_best_only=True,
                save_weights_only=False))

    return callbacks


callbacks = make_callbacks(model_name)

history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=VERBOSE,
    callbacks=callbacks,
    validation_data=(X_valid, y_valid))


# Once the training is done, we can load back the model so that we do not need to train
# it again
def load_and_evaluate(model_name, return_model=False):
    # Load the trained model and evaluate the log loss and accuracy
    model = load_model(f'{model_dir}{model_name}.h5')
    r = model.evaluate(X_valid, y_valid, batch_size=2048, verbose=1)

    valid_crossentropy = r[0]
    valid_accuracy = r[1]

    print(f'Cross Entropy: {round(valid_crossentropy, 4)}')
    print(f'Accuracu: {round(100 * valid_accuracy, 2)}%')

    if return_model:
        return model


# This is print the cross entropy loss and accuracy of the model to the console:
model = load_and_evaluate(model_name, return_model=True)

# Generating output
from IPython.display import HTML


def header(text, color='black'):
    raw_html = f'<h1 style="color: {color};"><center>' + \
               str(text) + '</center></h1>'
    return raw_html


def box(text):
    raw_html = '<div style="border:1px inset black; padding:1em; font-size: 20px;">' + \
               str(text) + '</div>'
    return raw_html


def addContent(old_html, raw_html):
    old_html += raw_html
    return old_html


def generate_output(model,
                    sequences,
                    training_length=50,
                    new_words=50,
                    diversity=1,
                    return_output=False,
                    n_gen=1):
    # Generate 'new word' words of output from a trained model and format into HTML:
    # Choose a random sequence
    seq = random.choice(sequences)

    # Choose a random starting point
    seed_idx = random.randint(0, len(seq) - training_length - 10)

    # Ending the index for seed
    end_idx = seed_idx + training_length

    gen_list = []

    for n in range(n_gen):
        # Extract the seed sequence
        seed = seq[seed_idx:end_idx]
        original_sequence = [idx_word[i] for i in seed]
        generated = seed[:] + ['#']

        # Find the actual entire entire sequence:
        actual = generated[:] + seq[end_idx:end_idx + new_words]

        # Keep adding new words
        for i in range(new_words):
            # Make prediction from the seed
            preds = model.predict(np.array(seed).reshape(1, -1))[0].astype(np.float64)

            # Diversify
            preds = np.log(preds) / diversity
            exp_preds = np.exp(preds)

            # Softmax
            preds = exp_preds / sum(exp_preds)

            # Choose the next word
            probas = np.random.multinomial(1, preds, 1)[0]

            next_idx = np.argmax(probas)

            # New seed adds on old word
            seed = seed[1:] + [next_idx]
            generated.append(next_idx)

        # Showing generated and actual abstract
        n = []

        for i in generated:
            n.append(idx_word.get(i, '< --- >'))

        gen_list.append(n)

    a = []

    for i in actual:
        a.append(idx_word.get(i, '< --- >'))

    a = a[training_length]

    gen_list = [gen[training_length;training_length + len(a)] for gen in gen_list]

    if return_output:
        return original_sequence, gen_list, a

    # HTML formatting :
    seed_html = ''
    seed_html = addContent(seed_html, header(
        'Seed Sequence', color='darkblue'))
    seed_html = addContent(seed_html,
                           box(remove_spaces(' '.join(original_sequence))))

    gen_html = ''
    gen_html = addContent(gen_html, header('RNN Generated', color='darkred'))
    gen_html = addContent(gen_html, box(remove_spaces(' '.join(gen_list[0]))))

    a_html = ''
    a_html = addContent(a_html, header('Actual', color='darkgreen'))
    a_html = addContent(a_html, box(remove_spaces(' '.join(a))))

    return seed_html, gen_html, a_html


seed_html, gen_html, a_html = generate_output(model, sequences,TRAINING_LENGTH)


# This will print out the outputs
print(HTML(seed_html))
print(HTML(gen_html))
print(HTML(a_html))
