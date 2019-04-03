import os
import sys
import logging
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from data_loader_emotion_feature import DataLoader
from common import config as cfg
from pathlib import Path
from tf_metrics import precision, recall, f1


tf.enable_eager_execution()

data_loader = DataLoader()

def model_fn(mode, features, labels):
    # Logging
    Path('results').mkdir(exist_ok=True)
    tf.logging.set_verbosity(logging.INFO)
    handlers = [logging.FileHandler('./results/main.log'),
                logging.StreamHandler(sys.stdout)]
    logging.getLogger('tensorflow').handlers = handlers
    
    word_inputs, char_inputs, emo_features = features
 
    training = (mode == tf.estimator.ModeKeys.TRAIN)
     
    # Embeddings
    embeddings = tf.get_variable('embeddings', [cfg.num_chars + 2, cfg.char_embed_dim])
    char_input_emb = tf.nn.embedding_lookup(embeddings, char_inputs)
    
    # Reshaping for CNN
    output = tf.reshape(char_input_emb, [-1, tf.shape(char_inputs)[2], cfg.char_embed_dim])

    # CNN
    output = tf.layers.conv1d(output, filters=64, kernel_size=2, strides=1, padding="same", activation=tf.nn.relu)
    output = tf.layers.max_pooling1d(output, pool_size=2, strides=2)
    output = tf.layers.conv1d(output, filters=128, kernel_size=2, strides=1, padding="same", activation=tf.nn.relu)
    output = tf.layers.max_pooling1d(output, pool_size=2, strides=2)
    
    cnn_output = tf.layers.dropout(output, rate=.5, training=training)
    cnn_output = tf.layers.flatten(cnn_output)

    # Reshaping CNN and concatenating for LSTM
    cnn_output = tf.reshape(cnn_output, [-1, tf.shape(char_inputs)[1], 128 * int(cfg.word_max_len / 4)])
    word_inputs = tf.layers.dropout(word_inputs, rate=.5, training=training)
    lstm_inputs = tf.concat([word_inputs, cnn_output], axis=-1)
    
    # LSTM
    #with tf.variable_scope('lstm_1'):
    #    cell = tf.contrib.rnn.LSTMCell(num_units=cfg.lstm_units)
    #    lstm_inputs, _ = tf.nn.dynamic_rnn(cell, lstm_inputs, dtype=tf.float32)
    
    with tf.variable_scope('lstm_2'):
        cell = tf.contrib.rnn.LSTMCell(num_units=cfg.lstm_units)
        _, final_state = tf.nn.dynamic_rnn(cell, lstm_inputs, dtype=tf.float32)
    
    lstm_output = tf.concat(final_state, axis=-1)
    lstm_output = tf.concat([lstm_output, emo_features], axis=-1)
    lstm_output = tf.layers.dropout(lstm_output, rate=.5, training=training)
    
    # Dense
    output = tf.layers.dense(lstm_output, 128)
    output = tf.layers.dropout(output, rate=.5, training=training)
    logits = tf.layers.dense(output, 2)
    
    # Loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    
    # Metrics
    indices = [0, 1]
    labels = tf.argmax(labels, 1)
    pred_ids = tf.argmax(logits, 1)

    metrics = {
               'acc': tf.metrics.accuracy(labels, pred_ids),
               'precision': precision(labels, pred_ids, 2, indices, None, average='macro'),
               'recall': recall(labels, pred_ids, 2, indices, None, average='macro'),
               'f1': f1(labels, pred_ids, 2, indices, None, average='macro')
              }
               
    for metric_name, op in metrics.items():
        tf.summary.scalar(metric_name, op[1])
    
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, 
                                          eval_metric_ops=metrics)

    elif mode == tf.estimator.ModeKeys.TRAIN:
        train_op = tf.train.AdamOptimizer(cfg.learning_rate).minimize(loss, global_step=tf.train.get_or_create_global_step())

        return tf.estimator.EstimatorSpec(mode, loss=loss, 
                                          train_op=train_op)


def input_fn(mode=None):
    data_generator = lambda: data_loader.data_generator(mode=mode)

    dataset = tf.data.Dataset.from_generator(data_generator, 
                                             output_types=((tf.float32, tf.int32, tf.float32), tf.int32),
                                             output_shapes=(([None, cfg.word_embed_dim], [None, None], [128]), [None]))

    if mode is 'train':
        dataset = dataset.shuffle(cfg.shuffle_buffer).repeat(cfg.num_epochs)
     
    dataset = dataset.padded_batch(cfg.batch_size, padded_shapes=(([None, cfg.word_embed_dim], [None, None], [128]), [None]))
       
    return dataset


def train():
    train_input_func = lambda: input_fn(mode='train')
    eval_input_func = lambda: input_fn(mode='valid')
    
    est_conf = tf.estimator.RunConfig(cfg.model_dir, save_checkpoints_secs=30, keep_checkpoint_max=500)
    est = tf.estimator.Estimator(model_fn, cfg.model_dir, est_conf)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_func)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_func, throttle_secs=30)
    
    tf.estimator.train_and_evaluate(est, train_spec, eval_spec)

if __name__ == '__main__':
    train()
