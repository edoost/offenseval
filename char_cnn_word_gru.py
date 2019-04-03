import os
import sys
import logging
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from data_loader import DataLoader
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
    
    cnn_output = tf.layers.dropout(output, rate=cfg.dropout, training=training)
    cnn_output = tf.layers.flatten(cnn_output)

    #cell = tf.contrib.rnn.GRUCell(num_units=64)
    #encoder_outputs, cnn_output = tf.nn.dynamic_rnn(cell, word_inputs, dtype=tf.float32)

    # Reshaping CNN and concatenating for LSTM
    cnn_output = tf.reshape(cnn_output, [-1, tf.shape(char_inputs)[1], 128 * 15])
    word_inputs = tf.concat([word_inputs, cnn_output], axis=-1)
    
    # LSTM
    cell = tf.contrib.rnn.GRUCell(num_units=cfg.lstm_units)
    encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(cell, word_inputs, dtype=tf.float32)

    #output = tf.concat(encoder_final_state, axis=-1)
    lstm_output = tf.layers.dropout(encoder_final_state, rate=cfg.dropout, training=training)

    # Dense
    output = tf.layers.dense(lstm_output, 128)
    output = tf.layers.dropout(output, rate=.5, training=training)
    logits = tf.layers.dense(output, 2)
    
    # Loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    
    # Metrics
    indices = [0, 1]
    metrics = {
               'acc': tf.metrics.accuracy(tf.argmax(labels, 1), tf.argmax(logits, 1)),
               #'precision': precision(labels, pred_ids, 2, indices, weights),
               #'recall': recall(labels, pred_ids, 2, indices, weights),
               #'f1': f1(labels, pred_ids, 2, indices, weights)
               'precision': tf.metrics.precision(tf.argmax(labels, 1), tf.argmax(logits, 1)),
               'recall': tf.metrics.recall(tf.argmax(labels, 1), tf.argmax(logits, 1)),
               'f1': tf.contrib.metrics.f1_score(tf.argmax(labels, 1), tf.argmax(logits, 1))
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
                                                 output_shapes=(([None, 300], [None, None]), [None]))

        if mode is 'train':
            dataset = dataset.shuffle(cfg.shuffle_buffer).repeat(cfg.num_epochs)
        
        dataset = dataset.padded_batch(cfg.batch_size, padded_shapes=(([None, 300], [None, None]), [None]))
       
        return dataset


def train():
    train_input_func = lambda: input_fn(mode='train')
    eval_input_func = lambda: input_fn(mode='valid')
    
    est_conf = tf.estimator.RunConfig(cfg.model_dir, save_checkpoints_secs=60)
    est = tf.estimator.Estimator(model_fn, cfg.model_dir, est_conf)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_func)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_func, throttle_secs=60)
    
    tf.estimator.train_and_evaluate(est, train_spec, eval_spec)

if __name__ == '__main__':
    train()
