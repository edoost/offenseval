# Ghmerti at SemEval-2019 Task 6: A Deep Word- and Character-based Approach to Offensive Language Identification

### Introduction
This is the code for [***Ghmerti at SemEval-2019 Task 6: A Deep Word- and Character-based Approach to Offensive Language Identification***](https://www.aclweb.org/anthology/S19-2110/), published in the Proceedings of The 13th International Workshop on Semantic Evaluation (SemEval 2019).

- ```char_cnn_word_blstm.py``` is a CNN with character embeddings as input and a bidirectional RNN with LSTM cells and word embeddings as input features.

- ```char_cnn_word_blstm_attn.py``` is the same thing, except that it also uses the attention mechanism on RNN's output.

- ```char_cnn_word_gru.py``` is exactly the same as ```char_cnn_word_blstm.py```, except that the RNN's cell is GRU.

- ```char_cnn_word_lstm.py``` this one uses a unidirectional RNN with LSTM cells.

- ```char_cnn_word_lstm_attn.py``` is the same as the above one, plus attention.

- ```char_cnn_word_lstm_emotion.py``` also makes use of emotion dense features as the input to the FCN layer.

- ```data_loader.py``` and ```data_loader_emotion_feature.py``` read the data from file, preprocess it, and convert it to the right format for first 5 and the last one, repectively.

### Citaion
```bibtex
@inproceedings{doostmohammadi-etal-2019-ghmerti,
    title = "Ghmerti at {S}em{E}val-2019 Task 6: A Deep Word- and Character-based Approach to Offensive Language Identification",
    author = "Doostmohammadi, Ehsan  and
      Sameti, Hossein  and
      Saffar, Ali",
    booktitle = "Proceedings of the 13th International Workshop on Semantic Evaluation",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/S19-2110",
    doi = "10.18653/v1/S19-2110",
    pages = "617--621",
}
```
