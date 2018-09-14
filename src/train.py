# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 16:21:28 2018

@author: bob.lee
"""
import tensorflow as tf
import os
import random
import cv2
from src.model import CNN
from src.config import Model
import numpy as np
import sys

logDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'model/'))
if not os.path.isdir(logDIR):
    os.mkdir(logDIR)


def text2vec(text):
    text_len = len(text)
    if text_len > Model.max_length:
        raise ValueError('验证码长度不对')

    vector = np.zeros(Model.max_length * Model.len_char)
    for i, c in enumerate(text):
        idx = i * Model.len_char + Model.char_set.index(c)
        vector[idx] = 1
    return vector


def get_data(image_txt, label_txt, image_file):
    image = []
    for line in open(image_txt, encoding='utf8'):
        image.append(image_file + line.strip('\n') + '.jpg')
    label = []
    for line in open(label_txt, encoding='utf8'):
        label.append(line.strip('\n'))
    return image, label


def get_next_batch(batch_size, image, label):
    batch_x = np.zeros([batch_size, Model.image_w * Model.image_h * 3])
    batch_y = np.zeros([batch_size, Model.max_length * Model.len_char])
    for i in range(batch_size):
        number = random.randrange(0, 100)
        image_gray = cv2.imread(image[number])
        batch_x[i, :] = image_gray.flatten() / 255
        batch_y[i, :] = text2vec(label[number])
    return batch_x, batch_y


def train_cnn(image_txt, label_txt, image_file):
    train_x = tf.placeholder(tf.float32, [None, Model.image_h * Model.image_w * 3])
    train_y = tf.placeholder(tf.float32, [None, Model.max_length * Model.len_char])
    image, label = get_data(image_txt, label_txt, image_file)
    keep_prob = tf.placeholder(tf.float32)
    model = CNN()
    output = model.cnn_model()
    optimizer, loss = model.model_loss(output)
    accuracy = model.model_acc(output)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        count = 0
        while True:
            batch_x, batch_y = get_next_batch(32, image, label)
            _, loss_ = sess.run([optimizer, loss], feed_dict={train_x: batch_x, train_y: batch_y, keep_prob: 0.75})
            print(step, loss_)
            # 每100 step计算一次准确率
            if step % 100 == 0:
                batch_x_test, batch_y_test = get_next_batch(64, image, label)
                acc = sess.run(accuracy, feed_dict={train_x: batch_x_test, train_y: batch_y_test, keep_prob: 1.})
                print(step, acc)
                if step % 1000 == 0:
                    checkpoint_path = os.path.join(logDIR, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
                if acc == 1:
                    count = count + 1
                if count == 10:
                    checkpoint_path = os.path.join(logDIR, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
                    break
            step += 1


if __name__ == '__main__':
    train_cnn(sys.argv[1], sys.argv[1], sys.argv[1])
