# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 16:21:28 2018

@author: bob.lee
"""
import tensorflow as tf
import os
from src.model import CNN
from src.config import Model

logDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'model/'))
if not os.path.isdir(logDIR):
    os.mkdir(logDIR)


def image_cnn(image):
    train_x = tf.placeholder(tf.float32, [None, Model.image_h * Model.image_w * 3])
    keep_prob = tf.placeholder(tf.float32)
    model = CNN()
    output = model.cnn_model()
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(logDIR)
    with tf.Session() as sess:
        saver.restore(sess, ckpt.model_checkpoint_path)
        predict = tf.argmax(tf.reshape(output, [-1, Model.max_length, Model.len_char]), 2)
        text_list = sess.run(predict, feed_dict={train_x: [image], keep_prob: 1})
        text = text_list[0].tolist()
        ans = [Model.char_set[char] for i, char in enumerate(text)]
    return ''.join(ans)
