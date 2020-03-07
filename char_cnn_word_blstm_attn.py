import os
import sys
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tf_metrics import precision, recall, f1

from data_loader import DataLoader
from common import config as cfg


tf.enable_eager_execution()

# This module preprocesses and loads the data
data_loader = DataLoader()


def model_fn(mode, features, labels):
    # Logging
    Path('results').mkdir(exist_ok=True)
    tf.logging.set_verbosity(logging.INFO)
    handlers = [logging.FileHandler('./results/main.log'),
                logging.StreamHandler(sys.stdout)]
    logging.getLogger('tensorflow').handlers = handlers
    
    word_inputs, char_inputs = features
 
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
    fw_cell = tf.contrib.rnn.LSTMCell(num_units=cfg.lstm_units)
    bw_cell = tf.contrib.rnn.LSTMCell(num_units=cfg.lstm_units)
    (fw_outputs, bw_outputs), (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, lstm_inputs, dtype=tf.float32)
    
    # Attention
    W = tf.Variable(tf.random_normal([cfg.lstm_units], stddev=0.1))
    H = fw_outputs + bw_outputs
    M = tf.tanh(H)
    alpha = tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(M, [-1, cfg.lstm_units]), tf.reshape(W, [-1, 1])), (-1, tf.shape(word_inputs)[1])))
    r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(alpha, [-1, tf.shape(word_inputs)[1], 1]))
    r = tf.squeeze(r)
    h_star = tf.tanh(r)
    h_drop = tf.nn.dropout(h_star, .5)
    
    # Dense
    FC_W = tf.Variable(tf.truncated_normal([cfg.lstm_units, 2], stddev=0.1))
    FC_b = tf.Variable(tf.constant(0., shape=[2]))
    logits = tf.nn.xw_plus_b(h_drop, FC_W, FC_b)
    
    # Loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    
    # Gradient clipping
    # optimizer = tf.train.AdamOptimizer(1e-4)
    # gradients, variables = zip(*optimizer.compute_gradients(loss))
    # gradients, _ = tf.clip_by_global_norm(gradients, .1)
    # train_op = optimizer.apply_gradients(zip(gradients, variables), tf.train.get_global_step())  
    
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
                                             output_types=((tf.float32, tf.int32), tf.int32),
                                             output_shapes=(([None, cfg.word_embed_dim], [None, None]), [None]))

    if mode is 'train':
        dataset = dataset.shuffle(cfg.shuffle_buffer).repeat(cfg.num_epochs)
        
    dataset = dataset.padded_batch(cfg.batch_size, padded_shapes=(([None, cfg.word_embed_dim], [None, None]), [None]))
       
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
