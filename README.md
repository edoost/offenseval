# Ghmerti at SemEval-2019 Task 6: A Deep Word- and Character-based Approach to Offensive Language Identification

- ```char_cnn_word_blstm.py``` is a CNN with character embeddings as input and an bidirectional RNN with LSTM cells and word embeddings as input features.

- ```char_cnn_word_blstm_attn.py``` is the same thing, except that it also uses the attention mechanism on RNN's output.

- ```char_cnn_word_gru.py``` is exactly the same as ```char_cnn_word_blstm.py```, except that the RNN's cell is GRU.

- ```char_cnn_word_lstm.py``` this one uses a unidirectional RNN with LSTM cells.
