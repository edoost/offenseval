# directories
train_data_dir = './data/train_data_2.tsv'
valid_data_dir = './data/valid_data.tsv'
model_dir = '/mnt/hdd2/model_b_2'
we_model_dir = './data/we_model.vec'
we_pickled_model_dir = './data/we_model.pickle'
char_indices_dir = './data/char_indices.pickle'
abbreviations_dir = './data/preprocessing_data/abbreviations.tsv'
distorted_black_words_dir = './data/preprocessing_data/distorted_black_words.tsv'
contractions_dir = './data/preprocessing_data/contractions.tsv'
train_emotion_features_dir = './data/valid_emotions.pickle'
valid_emotion_features_dir = './data/train_emotions.pickle'

# general
num_epochs = 200
shuffle_buffer = 3500
batch_size = 32
learning_rate = 1e-3
word_max_len = 150


# embeddings
num_words = 10000  # number of most common words to keep in vocabulary
word_embed_dim = 300
num_chars = 256
char_embed_dim = 32


# lstm variables
lstm_units = 256  # number of hidden units in the RNN
dropout = .5  # keeping probability

