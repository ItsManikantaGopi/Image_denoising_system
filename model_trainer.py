"""
module for training the model
"""
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from dataset_loader import load_dataset
from auto_encoder_model import AutoEncoder

def save_some_pics(noisy_array,non_noisy_array,
                   noisy_dest,non_noisy_dest):
    """
    function to save some images in the
    array of noisy images and respective non
    noisy images
    """
    for index,noisy_image in enumerate(noisy_array):
        non_noisy_image = non_noisy_array[index]
        plt.imsave(noisy_dest+f"{index}.png",
                    noisy_image, cmap="gray")
        plt.imsave(non_noisy_dest+f"{index}.png",
                    non_noisy_image, cmap = "gray")

def train_model(model_name = "",epochs = 1):
    """
    Method for loading data , training model and
    saving it and it will also save some data
    from validation data
    """
    train_data,validation_data = load_dataset()
    test_noisy = validation_data[0][-50:]
    test_normal = validation_data[1][-50:]
    noisy_dest = "dataset/testing_dataset/noise"
    non_noisy_dest = "dataset/testing_dataset/original"
    save_some_pics(test_noisy, test_normal,
                   noisy_dest, non_noisy_dest)
    valid_noisy = validation_data[0][:-50]
    valid_normal = validation_data[1][:-50]
    validation_data = (valid_noisy,valid_normal)
    model = AutoEncoder((28,28,1),3,1).model
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy')
    print("training model...")
    model = model.fit(train_data[0], train_data[1], epochs = 1,
                      batch_size = 128, shuffle = True,
                      validation_data = validation_data)
    if model_name not in os.listdir("models"):
        os.mkdir(f"models/{model_name}")
    for epoch in range(1, epochs):
        print(f"saving epoch{epoch}\n")
        model.save("models/"+model_name+f"epoch_{epoch}.h5")
        model = tf.keras.models.load_model(model)

if __name__ == "__main__":
    train_model(model_name="denoiser", epochs=10)
