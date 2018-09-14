# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 16:21:28 2018

@author: bob.lee
"""
from src.model_test import image_cnn
from PIL import Image
import sys


def image_main(image_name):
    image = Image.open(image_name)
    image = image.flatten() / 255
    predict_text = image_cnn(image)
    return predict_text


if __name__ == '__main__':
    result = image_main(sys.argv[1])
    print(result)
