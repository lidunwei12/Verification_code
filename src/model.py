# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 16:21:28 2018

@author: bob.lee
"""
import tensorflow as tf
from src.config import Model


class CNN():

    def __init__(self):
        self.name = 'lidunwei'

    def cnn_model(self):
        w_alpha = 0.01
        b_alpha = 0.1
        x = tf.placeholder(tf.float32, [None, Model.image_h * Model.image_w * 3])
        keep_prob = tf.placeholder(tf.float32)  # dropout
        w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 3, 32]))
        b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv1 = tf.nn.dropout(conv1, keep_prob)

        w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
        b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.dropout(conv2, keep_prob)

        w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
        b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv3 = tf.nn.dropout(conv3, keep_prob)

        # Fully connected layer
        w_d = tf.Variable(w_alpha * tf.random_normal([7 * 17 * 64, 1024]))
        b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
        dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
        dense = tf.nn.dropout(dense, keep_prob)

        w_out = tf.Variable(w_alpha * tf.random_normal([1024, Model.max_length * Model.len_char]))
        b_out = tf.Variable(b_alpha * tf.random_normal([Model.max_length * Model.len_char]))
        out = tf.add(tf.matmul(dense, w_out), b_out)
        return out

    def model_loss(self, out):
        Y = tf.placeholder(tf.float32, [None, Model.max_length * Model.len_char])
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out, labels=Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
        return optimizer, loss

    def model_acc(self, out):
        Y = tf.placeholder(tf.float32, [None, Model.max_length * Model.len_char])
        predict = tf.reshape(out, [-1, Model.max_length * Model.len_char])
        max_idx_p = tf.argmax(predict, 2)
        max_idx_l = tf.argmax(tf.reshape(Y, [-1, Model.max_length, Model.len_char]), 2)
        correct_pred = tf.equal(max_idx_p, max_idx_l)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return accuracy
