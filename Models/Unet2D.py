

from keras.models import Model, Input
from keras.layers.merge import concatenate
from keras.layers.core import Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint


def initialize_inputs():
    model_inputs = {
      "layer_count": 5,
      "starting_filter_count": 16,
      "convolution_dict": {
         "kernel_initializer": "he_normal",
         "activation": "relu",
         "kernel_size": (3, 3),
         "padding": "same",
        },
      "dropout_weight": 0.2,
      "pooling_dims": (2, 2),
      "stride_dims": (2, 2),
      "output_activation": "sigmoid",
      "image_height": 128,
      "image_width": 128,
      "input_channels": 3,
      "categories": 1,
      "model_optimizer": "adam",
      "loss_function": "binary_crossentropy",
      "validation_split": 0.1,
      "epochs": 1,
      "batch_size": 5,
      "filename": 'Unet-trained-model.h5',
      "stop_after": 5
    }
    return model_inputs


def unet_conv_layer(convolution_dict, filtersize, inputlayer, dropoutweight, name):
    """ Add a convolutional layer to the Unet Structure
    Parameters
    ----------
    convolution_dict: Dictionary of params to pass to keras Unet2D method
    filtersize: Number of filters for this layer
    inputlayer: Tensor connected to this layers
    dropoutweight: Weight of dropouts in dropout layer between convolutions
    name: descriptive name for this convolution layer
    """
    convolution_dict["filters"] = filtersize
    convolution_dict["name"] = name+"conv_a"
    c = Conv2D(**convolution_dict)(inputlayer)
    c = Dropout(dropoutweight, name=name+"Drop")(c)
    convolution_dict["name"] = name+"conv_b"
    c = Conv2D(**convolution_dict)(c)
    return c


def unet_pool_layer(inputlayer, pool_dims, name):
    """ Add a pool layer to the Unet Structure
    Parameters
    ----------
    inputlayer: Tensor connected to this layers
    pool_dims: Tuple of dimensionality of pooling ie (height, width)
    name: descriptive name for this pooling layer
    """
    p = MaxPooling2D(pool_dims, name=name)(inputlayer)
    return p


def unet_expand_layer(input_layer, down_layer, filter_size, common_inputs, name):
    """ Add an expansion layer to the Unet Structure which includes one
        Conv2DTranspose layer followed by one concatenate layer.
    Parameters
    ----------
    input_layer: Deeper tensor that is an input to this layer
    down_layer: Layer from the dimensionality reduction side of the U to
                concatenate with
    filter_size: Number of filters for convolution transpose input
    common_inputs: Dictionary of model definition inputs. Must contain
                   stride_dims, pooling_dims, and convolution_dict which
                   contains a 'padding' entry
    pool_dims: Tuple of dimensionality of pooling ie (height, width)
    name: descriptive name for this expansion layer
    """
    stride_dims = common_inputs['stride_dims']
    up_dims = common_inputs['pooling_dims']
    pad_type = common_inputs['convolution_dict']['padding']

    u = Conv2DTranspose(
      filter_size, up_dims, name=name + "_transpose", strides=stride_dims,
      padding=pad_type)(input_layer)

    u = concatenate([u, down_layer])

    return u


def definemodel(model_params):
    """Build a model based on the inputs from model_params
    :param int num_layers: Number of layers in downsampling layers
    :param int starting_filter_count: Count of filters in 1st convolution layer
    :param dict convolution_dict: Parameters common to all convolution layers
    :param double dropout_weight: Value between 0-1 for dropout percentage
    :param tuple pooling_dims: dims for pooling (2,2) default height/img_width
    :param tuple stride_dims: dims for concatenate layers (2,2) default
    :param string output_activation: Output layer activation function
    :param int image_height: Height of images in pixels
    :param int image_width: Width of images in pixels
    :param int image_channels: Number of image channels (RGB=3, Grayscale=1...)
    :param string model_optimizer: Optimizer for Keras model
    :param string loss_function: Loss function to be minimized
    :param double validation_split: Percentage of training sent to validation
    :param int batch: Number of images per batch
    :param int trials: Number of epochs
    """
    #
    num_layers = model_params['layer_count']
    starting_filter_count = model_params['starting_filter_count']
    convolution_dict = model_params['convolution_dict']
    dropout_weight = model_params['dropout_weight']
    pooling_dims = model_params['pooling_dims']
    output_activation = model_params['output_activation']
    image_height = model_params['image_height']
    image_width = model_params['image_width']
    image_channels = model_params['input_channels']
    model_optimizer = model_params['model_optimizer']
    loss_function = model_params['loss_function']

    '''Input  Layer'''
    inputs = Input((image_height, image_width, image_channels))

    next_sequence_layer = inputs
    conv_layers = []

    '''
    Generate down sampling layers. Each layer consists of convolution followed
    by a dropout layer followed by a second dropout layer. All but the deepest
    layer feed into a pooling layer that reduces the dimensionality by 2x
    '''

    for i in range(num_layers):
        # Calculate number of filters in this convolution layer
        filter_size = starting_filter_count*2**i
        # Descriptive name for layer
        name = "down_"+str(i+1)+"_"

        next_sequence_layer = unet_conv_layer(
          convolution_dict, filter_size, next_sequence_layer,
          dropout_weight, name)

        conv_layers.append(next_sequence_layer)

        '''Pool all but the deepest layer'''
        if i < num_layers-1:
            next_sequence_layer = unet_pool_layer(
              next_sequence_layer, pooling_dims, name+"pool")
    '''
    Generate up sampling layers. Each layer consists of an expansion consisting
    of Conv2DTranspose with previous layer followed by an expansion via a
    concatenate layer combining Conv2dTranspose layer with the convolutional
    layer from the down-sampling side of the U located at the same depth as the
    current up sampling layer. Expansion is followed by a Unet_conv_layer which
    is a convolution followed by down sampling followed by second convolution.
    Output should have same dimensionality as input
    '''
    for j in range(num_layers-1):
        name = "up_"+str(num_layers-j) + "_"
        filter_size //= 2

        next_sequence_layer = unet_expand_layer(
          next_sequence_layer, conv_layers[-2-j], filter_size, model_params,
          name)

        if j == num_layers-2:
            next_sequence_layer.axis = 3

        next_sequence_layer = unet_conv_layer(
          convolution_dict, filter_size, next_sequence_layer,
          dropout_weight, name)

    '''
    Generate output layer which is a final convolution with size(1,1) and 1
    filter
    '''
    outputs = Conv2D(
      1, (1, 1), activation=output_activation,
      name="output_convolution")(next_sequence_layer)

    model = Model(inputs=[inputs], outputs=[outputs])

    model.compile(
      optimizer=model_optimizer, loss=loss_function)

    return model


def trainmodel(x_train, y_train, model, model_params):
    """ Returns model and model history. Can be called from hyperas to do
    parameter optimization
    """
    validation_split = model_params['validation_split']
    batch_size = model_params['batch_size']
    epochs = model_params['epochs']
    filename = model_params['filename']
    stop_after = model_params['stop_after']

    earlystopper = EarlyStopping(patience=stop_after, verbose=0)
    checkpointer = ModelCheckpoint(filename, verbose=1, save_best_only=True)

    results = model.fit(
      x_train, y_train, validation_split=validation_split,
      batch_size=batch_size, epochs=epochs, callbacks=[earlystopper, checkpointer])

    return model, results


if __name__ == '__main__':
    model_params = initialize_inputs()
    model = definemodel(model_params)
    # x_train, y_train, x_test, y_test = getData()
    # model, results = getmodel(x_train, y_train, x_test, y_test, **model_params)
