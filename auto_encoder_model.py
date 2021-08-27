"""
module for defining the model of the
auto encoder
"""
import tensorflow as tf
import numpy as np
class AutoEncoder:
    """
    class for defining an object of model creator
    """
    def __init__(self, input_size, depth, dilation):
        print("Constructing model...")
        self.model = AutoEncoder.get_model(
            input_size = input_size, depth = depth, dilation = dilation)
    @staticmethod
    def depth_seperabale_block(layer, num_filters = 3,
                            kernel_size = 3,
                            dilation=1, layer_name=""):
        """
        function for depth seperable convolution block
        """
        layer = tf.keras.layers.DepthwiseConv2D(kernel_size = kernel_size,
                                            padding = 'same', kernel_initializer = 'he_normal',
                                            dilation_rate = dilation)(layer)
        if layer_name != "":
            layer = tf.keras.layers.Conv2D(np.floor(num_filters),
                                    kernel_size = (1, 1), kernel_initializer = 'he_normal',
                                    use_bias = False, padding = 'same',
                                    dilation_rate = dilation, activation="relu",
                                    name = layer_name, )(layer)
        else:
            layer = tf.keras.layers.Conv2D(np.floor(num_filters),
                                    kernel_size = (1, 1), kernel_initializer = 'he_normal',
                                    use_bias = False, padding = 'same',
                                    dilation_rate = dilation,  activation = "relu")(layer)
        return layer
    @staticmethod
    def get_model(input_size = (None,None, 3), depth = 3, dilation = 1):
        """
        function for defining the model of autoencoder
        """
        depth = depth - 1
        features = 32*(2**depth)
        encoder_input = tf.keras.layers.Input(input_size)
        encoder_naming_template = "encode_"
        decoder_naming_template = "decode_"
        encode = encoder_input
        for i in range(1, depth+1):
            encode = AutoEncoder.depth_seperabale_block(encode, num_filters = features,
                                kernel_size = 3, dilation = dilation,
                                layer_name = encoder_naming_template+str(i))
            features = features//2
        decoder_input = encode
        decode = decoder_input
        for i in reversed(range(1, depth+1)):
            features = features*2
            decode = AutoEncoder.depth_seperabale_block(decode, num_filters = features,
                                kernel_size = 3, dilation = dilation,
                                layer_name = decoder_naming_template+str(i))
            decode = tf.keras.layers.UpSampling2D(size = (1, 1))(decode)
        decoder_output = AutoEncoder.depth_seperabale_block(decode, num_filters = 3,
                                    kernel_size = 1, dilation = dilation,
                                    layer_name = "output")
        model = tf.keras.Model(encoder_input, decoder_output)
        return model
