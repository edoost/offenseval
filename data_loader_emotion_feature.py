import re
import pickle
from collections import Counter
# import functools
# import operator

import emoji
import numpy as np
from gensim.models import KeyedVectors
from nltk.tokenize import TweetTokenizer

from common import config as cfg


class DataLoader:
    def __init__(self):
        # loading word embedding model
        try:
            with open(cfg.we_pickled_model_dir, 'rb') as handle:
                self.word_embedding_model = pickle.load(handle)
        except FileNotFoundError:
            self.word_embedding_model = KeyedVectors.load_word2vec_format(cfg.we_model_dir, binary=False)
            #self.word_embedding_model = KeyedVectors.load_word2vec_format('/mnt/hdd2/ehsan/word2vec.txt', binary=False)
            with open(cfg.we_pickled_model_dir, 'wb') as handle:
                pickle.dump(self.word_embedding_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        self.tokenizer = TweetTokenizer()
        
        # loading distorted black words
        self.distorted_black_words_dict = {}
        with open(cfg.distorted_black_words_dir) as file:
            for line in file:
                source, target = line.strip().split('\t')
                self.distorted_black_words_dict[source] = target

        # loading abbreviations
        self.abbr_dict = {}
        with open(cfg.abbreviations_dir) as file:
            for line in file:
                source, target = line.strip().split('\t')
                self.abbr_dict[source] = target
        
        # loading contractions
        self.contractions_dict = {}
        with open(cfg.contractions_dir) as file:
            for line in file:
                source, target = line.strip().split('\t')
                self.contractions_dict[source] = target 
        
        # loading/builing chracter indices
        try:
            with open(cfg.char_indices_dir, 'rb') as handle:
                self.char_to_index = pickle.load(handle)
        except FileNotFoundError:
            tweets, _, _, _ = self._data_reader(cfg.train_data_dir)
            chars_list = []
            chars_list.extend([char for tweet in tweets for char in tweet.replace(' ', ' ')])
            char_vocab_list = [x[0] for x in [('<PAD>', 0), ('<UNK>', 1)] + Counter(chars_list).most_common(cfg.num_chars)]
            
            self.char_to_index = {}
            for i, char in enumerate(char_vocab_list):
                self.char_to_index[char] = i

            with open(cfg.char_indices_dir, 'wb') as handle:
                pickle.dump(self.char_to_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        self.a_tag_to_index = {'NOT': 0, 'OFF': 1}
    
    def _data_reader(self, file_path):
        tweets, a_tags, b_tags, c_tags = [], [], [], []
        with open(file_path) as file:
            for line in file:
                try:
                    _, tweet, a_tag, b_tag, c_tag = line.strip().split('\t')
                except ValueError:
                    #continue
                    tweet, a_tag = line.strip().split('\t')
                    b_tag, c_tag = 'NA', 'NA'

                tweets.append(tweet)
                a_tags.append(a_tag)
                b_tags.append(b_tag)
                c_tags.append(c_tag)

        return tweets, a_tags, b_tags, c_tags
    
    def _preprocessor(self, tweet):
        for distorted_black_word in self.distorted_black_words_dict:
            tweet = tweet.replace(distorted_black_word, self.distorted_black_words_dict[distorted_black_word])
        
        # for word in self.abbr_dict:
        #     tweet = re.sub(' ' + word.lower() + ' ', self.abbr_dict.get(word.lower()), tweet)

        # for word in self.contractions_dict:
        #     tweet = re.sub(word, self.contractions_dict[word], tweet)
        
        # tweet = tweet.replace('@user', '').replace('@USER', '')
        tweet = re.sub(r' +', ' ', tweet)

        return tweet.strip().replace('&amp;', 'and')

    # def tokenizer(self, tweet):
    #     split_emoji = emoji.get_emoji_regexp().split(tweet)
    #     split_whitespace = [re.findall(r"[\w'@$/*]+|[.,!?;\"%()]", substr) if substr not in emoji.UNICODE_EMOJI else substr for substr in split_emoji] 
    #     tokenized_tweet = functools.reduce(operator.concat, [[x] if type(x) is str else x for x in split_whitespace])
    #
    #     return tokenized_tweet
    
    def _tweet_to_embeddings(self, tweet):
        embedded_tweet = []
        for word in self.tokenizer.tokenize(tweet):
            try:
                embedded_tweet.append(self.word_embedding_model[word.lower()])
            except KeyError:
                embedded_tweet.append([0 for _ in range(cfg.word_embed_dim)])
        return np.array(embedded_tweet)
   
    def _pad(self, word):
        for _ in range(cfg.word_max_len - len(word)):
            word.append(0)
        return word

    def _tweet_to_indices(self, tweet):
        indexed_tweet = []
        for word in self.tokenizer.tokenize(tweet):
            indexed_word = []
            for char in word:
                indexed_word.append(self.char_to_index.get(char, 1))
            indexed_tweet.append(self._pad(indexed_word))
        
        return np.array(indexed_tweet)

    def _tags_to_one_hot(self, tag, mode=None):
        if mode is 'a':
            return np.eye(2)[self.a_tag_to_index[tag]]
        elif mode is 'b':
            return np.eye(2)[self.b_tag_to_index[tag]]
    
    def data_generator(self, mode=None):
        if mode is 'train':
            with open(cfg.train_emotion_features_dir, 'rb') as handle:
                emotion_features = pickle.load(handle)

            tweets, a_tags, b_tags, c_tags = self._data_reader(cfg.train_data_dir)
        elif mode is 'valid':
            with open(cfg.valid_emotion_features_dir, 'rb') as handle:
                emotion_features = pickle.load(handle)
            
            tweets, a_tags, b_tags, c_tags = self._data_reader(cfg.valid_data_dir) 
        
        for tweet, a_tag, emotion_feature in zip(tweets, a_tags, emotion_features):
            preprocessed_tweet = self._preprocessor(tweet)
            
            embedded_tweet = self._tweet_to_embeddings(preprocessed_tweet)
            indexed_tweet = self._tweet_to_indices(preprocessed_tweet)
 
            indexed_a_tag = self._tags_to_one_hot(a_tag, 'a')
            
            yield (embedded_tweet, indexed_tweet, emotion_feature), indexed_a_tag

    def data_generator_pred(self):
        tweets, a_tags, b_tags, c_tags = self._data_reader(cfg.valid_data_dir)

        for tweet in tweets:
            preprocessed_tweet = self._preprocessor(tweet)
            
            embedded_tweet = self._tweet_to_embeddings(preprocessed_tweet)
            indexed_tweet = self._tweet_to_indices(preprocessed_tweet)
 
            yield embedded_tweet, indexed_tweet
