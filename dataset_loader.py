"""module for loading datasets for model"""
import random
import numpy as np
import tensorflow as tf
def add_noise(image_array, random_chance=5):
    """
    Function to take an array of images and return
    the array of images with adding noise to it around
    the random chance of given value.
    """
    images = []
    for image in image_array:
        noisy = []
        for row in image:
            new_row = []
            for pix in row:
                if random.choice(range(100)) <= random_chance:
                    new_val = random.uniform(0, 1)
                    new_row.append(new_val)
                else:
                    new_row.append(pix)
            noisy.append(new_row)
        images.append(np.array(noisy))
    return np.array(images)
def load_dataset():
    """
    Function to load dataset and perform required operations like
    adding noise usingadd_noise function
    """
    print("Loading dataset...")
    (trainx,_), (testx,_) = tf.keras.datasets.mnist.load_data()
    trainx = trainx/255
    testx = testx/255
    x_train_noisy = add_noise(trainx)
    x_test_noisy = add_noise(testx)
    train_dataset = (x_train_noisy,trainx)
    validation_dataset = (x_test_noisy[:-500],testx)
    return (train_dataset,validation_dataset)
