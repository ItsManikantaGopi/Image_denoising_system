import gradio as gr
import cv2
from tensorflow.keras.models import load_model
import numpy as np
model = load_model('models\\autoencoder_10_epochs.h5'.encode("utf-8"))
def Predict(image):
    """
    Method to get input image and predicts it's output
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image
    image = image.reshape((-1, 28, 28, 1)),
    prediction = model.predict(image)[0]/255
    return np.squeeze(prediction, axis=2)
image_in = gr.inputs.Image(shape = (28, 28))
image_out = gr.outputs.Image(type = "numpy")
gr.Interface(fn = Predict,inputs = image_in,outputs = "image",
             title="Image Denoising",description= "upload image and denoise it").launch()