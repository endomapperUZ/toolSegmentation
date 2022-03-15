import tensorflow as tf
from tensorflow.keras import layers, regularizers


def reshape_into(inputs, input_to_copy):
    return tf.image.resize(inputs, (input_to_copy.shape[1], input_to_copy.shape[2]), method=tf.image.ResizeMethod.BILINEAR)


# convolution
def convolution(filters, kernel_size, strides=1, dilation_rate=1, use_bias=True):
    return layers.Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=use_bias,
                         kernel_regularizer=regularizers.l2(l=0.00004), dilation_rate=dilation_rate)


# Traspose convolution
def transposeConv(filters, kernel_size, strides=1, dilation_rate=1, use_bias=True):
    return layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding='same', use_bias=use_bias,
                                  kernel_regularizer=regularizers.l2(l=0.00004), dilation_rate=dilation_rate)



# Depthwise convolution
def depthwiseConv(kernel_size, strides=1, depth_multiplier=1, dilation_rate=1, use_bias=True):
    return layers.DepthwiseConv2D(kernel_size, strides=strides, depth_multiplier=depth_multiplier,
                                  padding='same', use_bias=use_bias, kernel_regularizer=regularizers.l2(l=0.0001),
                                  dilation_rate=dilation_rate)


# Depthwise convolution
def separableConv(filters, kernel_size, strides=1, dilation_rate=1, use_bias=True):
    return layers.SeparableConv2D(filters, kernel_size, strides=strides, padding='same', use_bias=use_bias,  activation=None,
                                  pointwise_regularizer=regularizers.l2(l=0.00004), dilation_rate=dilation_rate)


class DepthwiseConv_BN(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides=1, dilation_rate=1):
        super(DepthwiseConv_BN, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        # separableConv
        self.conv = separableConv(filters=filters, kernel_size=kernel_size, strides=strides,
                                  dilation_rate=dilation_rate)
        self.bn = layers.BatchNormalization()

    def call(self, inputs, activation=True, training=True):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        if activation:
            x = layers.ReLU()(x)

        return x



class Conv_BN(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides=1, dilation_rate=1):
        super(Conv_BN, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv = convolution(filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate)
        self.bn = layers.BatchNormalization()

    def call(self, inputs, activation=True, training=True):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        if activation:
            x = layers.ReLU()(x)

        return x



class Residual_SeparableConv(tf.keras.Model):
    def __init__(self, filters, kernel_size, dilation_rate=1, strides=1, dropout=0):
        super(Residual_SeparableConv, self).__init__()

        self.conv = DepthwiseConv_BN(filters, kernel_size, strides=strides, dilation_rate=dilation_rate)
        self.dropout = layers.Dropout(rate=dropout)
    def call(self, inputs, training=True):

        x = self.conv(inputs, activation=False, training=training)
        x = self.dropout(x, training=training)
        if inputs.shape == x.shape:
            x = x + inputs
        x = layers.ReLU()(x)

        return x

class Residual_SeparableConv_dil(tf.keras.Model):
    def __init__(self, filters, kernel_size, dilation_rate=1, strides=1, dropout=0.25):
        super(Residual_SeparableConv_dil, self).__init__()
        self.conv1 = depthwiseConv(kernel_size, strides=strides, depth_multiplier=1, dilation_rate=1, use_bias=False)
        self.conv2 = depthwiseConv(kernel_size, strides=strides, depth_multiplier=1, dilation_rate=dilation_rate, use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()


        self.conv = convolution(filters=filters, kernel_size=1, strides=1, dilation_rate=1)
        self.bn = layers.BatchNormalization()

        self.dropout = layers.Dropout(rate=dropout)
    def call(self, inputs, training=True):

        x1 = self.conv1(inputs)
        x1 = self.bn1(x1, training=training)
        x2 = self.conv2(inputs)
        x2 = self.bn2(x2, training=training)
        x1 = layers.ReLU()(x1)
        x2 = layers.ReLU()(x2)

        x = self.conv(x1 + x2)
        x = self.bn(x, training=training)
        x = self.dropout(x, training=training)

        if inputs.shape == x.shape:
            x = x + inputs

        x = layers.ReLU()(x)

        return x


class MininetV2Module(tf.keras.Model):
    def __init__(self, filters, kernel_size, dilation_rate=1, strides=1, dropout=0):
        super(MininetV2Module, self).__init__()

        self.conv1 = Residual_SeparableConv(filters, kernel_size, strides=strides, dilation_rate=1, dropout=dropout)
        self.conv2 = Residual_SeparableConv(filters, kernel_size, strides=1, dilation_rate=dilation_rate, dropout=dropout)


    def call(self, inputs, training=True):

        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)

        return x


class MininetV2Module_dil(tf.keras.Model):
    def __init__(self, filters, kernel_size, dilation_rate=1, strides=1, dropout=0.25):
        super(MininetV2Module_dil, self).__init__()

        self.conv1 = Residual_SeparableConv_dil(filters, kernel_size, strides=strides, dilation_rate=1, dropout=dropout)
        self.conv2 = Residual_SeparableConv_dil(filters, kernel_size, strides=1, dilation_rate=dilation_rate, dropout=dropout)


    def call(self, inputs, training=True):

        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)

        return x


class MininetV2Downsample(tf.keras.Model):
    def __init__(self, n_filters_in, n_filters_out):
        super(MininetV2Downsample, self).__init__()

        self.maxpool_use = n_filters_in < n_filters_out
        if not self.maxpool_use:
            filters_conv = n_filters_out
        else:
            filters_conv = n_filters_out - n_filters_in


        self.conv = convolution(filters_conv, 3, strides=2, dilation_rate=1, use_bias=False)
        self.bn = layers.BatchNormalization()


    def call(self, inputs, training=True):

        x = self.conv(inputs, training=training)

        if self.maxpool_use:
            y = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(inputs)
            x = tf.concat([x, y], axis=-1)

        x = self.bn(x, training=training)
        x = layers.ReLU()(x)

        return x


class MininetV2Upsample(tf.keras.Model):
    def __init__(self, filters, kernel_size=3, strides=2):
        super(MininetV2Upsample, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv = transposeConv(filters=filters, kernel_size=kernel_size, strides=strides)
        self.bn = layers.BatchNormalization()

    def call(self, inputs, last=False, training=True):
        x = self.conv(inputs)
        if not last:
            x = self.bn(x, training=training)
            x = layers.ReLU()(x)

        return x




class MiniNetv2(tf.keras.Model):

    def __init__(self, num_classes, include_top, **kwargs):
        super(MiniNetv2, self).__init__(**kwargs)
        self.down_b = MininetV2Downsample(3, 16)
        self.down_b2 = MininetV2Downsample(16, 64)

        self.down1 = MininetV2Downsample(3, 16)
        self.conv_mod_0 = MininetV2Module(32, 3, strides=1, dilation_rate=1)

        self.down2 = MininetV2Downsample(16, 64)
        self.conv_mod_1 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.conv_mod_2 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.conv_mod_3 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.conv_mod_4 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.conv_mod_5 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.down3 = MininetV2Downsample(64, 128)
        self.conv_mod_6 = MininetV2Module_dil(128, 3, strides=1, dilation_rate=2)
        self.conv_mod_7 = MininetV2Module_dil(128, 3, strides=1, dilation_rate=4)
        self.conv_mod_8 = MininetV2Module_dil(128, 3, strides=1, dilation_rate=8)
        self.conv_mod_9 = MininetV2Module_dil(128, 3, strides=1, dilation_rate=16)
        self.conv_mod_10 = MininetV2Module_dil(128, 3, strides=1, dilation_rate=1)
        self.conv_mod_11 = MininetV2Module_dil(128, 3, strides=1, dilation_rate=2)
        self.conv_mod_12 = MininetV2Module_dil(128, 3, strides=1, dilation_rate=4)
        self.conv_mod_13 = MininetV2Module_dil(128, 3, strides=1, dilation_rate=8)
        self.conv_mod_14 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.conv_mod_15 = MininetV2Module(64, 3, strides=1, dilation_rate=1)

        self.up1 = MininetV2Upsample(64)
        self.classify = MininetV2Upsample(num_classes)
        self.include_top=include_top
        
    def call(self, inputs, training=True, reshape=True):
        x1 = self.down1(inputs, training=training)
        x2 = self.down2(x1, training=training)
        x = self.conv_mod_1(x2, training=training)
        x = self.conv_mod_2(x, training=training)
        x = self.conv_mod_3(x, training=training)
        x = self.conv_mod_4(x, training=training)
        x3 = self.conv_mod_5(x, training=training)
        x = self.down3(x3, training=training)
        x = self.conv_mod_6(x, training=training)
        x = self.conv_mod_7(x, training=training)
        x = self.conv_mod_8(x, training=training)
        x = self.conv_mod_9(x, training=training)
        x = self.conv_mod_10(x, training=training)
        x = self.conv_mod_11(x, training=training)
        x = self.conv_mod_12(x, training=training)
        x = self.conv_mod_13(x, training=training)
        """        
        if self.include_top:
            
            x = self.up1(x, training=training)

            aux = self.down_b2(self.down_b(inputs, training=training))
            x = x + aux

            x = self.conv_mod_14(x, training=training)
            x = self.conv_mod_15(x, training=training)
            
            x = self.classify(x, training=training, last=True)
            if reshape:
                x = reshape_into(x, inputs)
            x = tf.keras.activations.softmax(x, axis=-1)
        """

        return x
    
class MiniNetDecod(tf.keras.Model):

    def __init__(self, num_classes, include_top, **kwargs):
        super(MiniNetDecod, self).__init__(**kwargs)
        self.down_b = MininetV2Downsample(3, 16)
        self.down_b2 = MininetV2Downsample(16, 64)

        self.down1 = MininetV2Downsample(3, 16)
        self.conv_mod_0 = MininetV2Module(32, 3, strides=1, dilation_rate=1)

        self.down2 = MininetV2Downsample(16, 64)
        self.conv_mod_1 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.conv_mod_2 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.conv_mod_3 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.conv_mod_4 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.conv_mod_5 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.down3 = MininetV2Downsample(64, 128)
        self.conv_mod_6 = MininetV2Module_dil(128, 3, strides=1, dilation_rate=2)
        self.conv_mod_7 = MininetV2Module_dil(128, 3, strides=1, dilation_rate=4)
        self.conv_mod_8 = MininetV2Module_dil(128, 3, strides=1, dilation_rate=8)
        self.conv_mod_9 = MininetV2Module_dil(128, 3, strides=1, dilation_rate=16)
        self.conv_mod_10 = MininetV2Module_dil(128, 3, strides=1, dilation_rate=1)
        self.conv_mod_11 = MininetV2Module_dil(128, 3, strides=1, dilation_rate=2)
        self.conv_mod_12 = MininetV2Module_dil(128, 3, strides=1, dilation_rate=4)
        self.conv_mod_13 = MininetV2Module_dil(128, 3, strides=1, dilation_rate=8)
        self.conv_mod_14 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.conv_mod_15 = MininetV2Module(64, 3, strides=1, dilation_rate=1)

        self.up1 = MininetV2Upsample(64)
        self.classify = MininetV2Upsample(num_classes)
        self.include_top=include_top
        
    def call(self, input_1, input_2 , training=True, reshape=True):
        """
        x1 = self.down1(inputs, training=training)
        x2 = self.down2(x1, training=training)
        x = self.conv_mod_1(x2, training=training)
        x = self.conv_mod_2(x, training=training)
        x = self.conv_mod_3(x, training=training)
        x = self.conv_mod_4(x, training=training)
        x3 = self.conv_mod_5(x, training=training)
        x = self.down3(x3, training=training)
        x = self.conv_mod_6(x, training=training)
        x = self.conv_mod_7(x, training=training)
        x = self.conv_mod_8(x, training=training)
        x = self.conv_mod_9(x, training=training)
        x = self.conv_mod_10(x, training=training)
        x = self.conv_mod_11(x, training=training)
        x = self.conv_mod_12(x, training=training)
        x = self.conv_mod_13(x, training=training)
        """        
        #if self.include_top:
        
        x = self.up1(input_2, training=training)
        
        aux = self.down_b2(self.down_b(input_1, training=training))
        x = x + aux

        x = self.conv_mod_14(x, training=training)
        x = self.conv_mod_15(x, training=training)
        
        x = self.classify(x, training=training, last=True)
        if reshape:
            x = reshape_into(x, input_1)
        x = tf.keras.activations.softmax(x, axis=-1)
        

        return x

class MiniNetClassif(tf.keras.Model):

    def __init__(self, **kwargs):
        
        super(MiniNetClassif, self).__init__(**kwargs)
        """
        self.down_b = MininetV2Downsample(3, 16)
        self.down_b2 = MininetV2Downsample(16, 64)

        self.down1 = MininetV2Downsample(3, 16)
        self.conv_mod_0 = MininetV2Module(32, 3, strides=1, dilation_rate=1)

        self.down2 = MininetV2Downsample(16, 64)
        self.conv_mod_1 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.conv_mod_2 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.conv_mod_3 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.conv_mod_4 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.conv_mod_5 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.down3 = MininetV2Downsample(64, 128)
        self.conv_mod_6 = MininetV2Module_dil(128, 3, strides=1, dilation_rate=2)
        self.conv_mod_7 = MininetV2Module_dil(128, 3, strides=1, dilation_rate=4)
        self.conv_mod_8 = MininetV2Module_dil(128, 3, strides=1, dilation_rate=8)
        self.conv_mod_9 = MininetV2Module_dil(128, 3, strides=1, dilation_rate=16)
        self.conv_mod_10 = MininetV2Module_dil(128, 3, strides=1, dilation_rate=1)
        self.conv_mod_11 = MininetV2Module_dil(128, 3, strides=1, dilation_rate=2)
        self.conv_mod_12 = MininetV2Module_dil(128, 3, strides=1, dilation_rate=4)
        self.conv_mod_13 = MininetV2Module_dil(128, 3, strides=1, dilation_rate=8)
        self.conv_mod_14 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.conv_mod_15 = MininetV2Module(64, 3, strides=1, dilation_rate=1)

        self.up1 = MininetV2Upsample(64)
        self.classify = MininetV2Upsample(num_classes)
        self.include_top=include_top
        """
        #self.layer_input_1 = tf.keras.Input(shape=input_shape)
        #self.model = model
        #self.index = index

    def call(self, inputs, training=True, reshape=True):
        """
        x1 = self.down1(inputs, training=training)
        x2 = self.down2(x1, training=training)
        x = self.conv_mod_1(x2, training=training)
        x = self.conv_mod_2(x, training=training)
        x = self.conv_mod_3(x, training=training)
        x = self.conv_mod_4(x, training=training)
        x3 = self.conv_mod_5(x, training=training)
        x = self.down3(x3, training=training)
        x = self.conv_mod_6(x, training=training)
        x = self.conv_mod_7(x, training=training)
        x = self.conv_mod_8(x, training=training)
        x = self.conv_mod_9(x, training=training)
        x = self.conv_mod_10(x, training=training)
        x = self.conv_mod_11(x, training=training)
        x = self.conv_mod_12(x, training=training)
        x = self.conv_mod_13(x, training=training)
        """        
        #if self.include_top:
        """    
        x = self.up1(x, training=training)

        aux = self.down_b2(self.down_b(inputs, training=training))
        x = x + aux

        x = self.conv_mod_14(x, training=training)
        x = self.conv_mod_15(x, training=training)
        
        x = self.classify(x, training=training, last=True)
        if reshape:
            x = reshape_into(x, inputs)
        x = tf.keras.activations.softmax(x, axis=-1)
        """
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs, training)
        x = tf.keras.layers.Dense(1,activation=tf.nn.sigmoid)(x)

        return x


class MiniNetEncodClassif(tf.keras.Model):

    def __init__(self, num_classes, include_top, **kwargs):
        super(MiniNetEncodClassif, self).__init__(**kwargs)
        """
        self.down_b = MininetV2Downsample(3, 16)
        self.down_b2 = MininetV2Downsample(16, 64)

        self.down1 = MininetV2Downsample(3, 16)
        self.conv_mod_0 = MininetV2Module(32, 3, strides=1, dilation_rate=1)

        self.down2 = MininetV2Downsample(16, 64)
        self.conv_mod_1 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.conv_mod_2 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.conv_mod_3 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.conv_mod_4 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.conv_mod_5 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.down3 = MininetV2Downsample(64, 128)
        self.conv_mod_6 = MininetV2Module_dil(128, 3, strides=1, dilation_rate=2)
        self.conv_mod_7 = MininetV2Module_dil(128, 3, strides=1, dilation_rate=4)
        self.conv_mod_8 = MininetV2Module_dil(128, 3, strides=1, dilation_rate=8)
        self.conv_mod_9 = MininetV2Module_dil(128, 3, strides=1, dilation_rate=16)
        self.conv_mod_10 = MininetV2Module_dil(128, 3, strides=1, dilation_rate=1)
        self.conv_mod_11 = MininetV2Module_dil(128, 3, strides=1, dilation_rate=2)
        self.conv_mod_12 = MininetV2Module_dil(128, 3, strides=1, dilation_rate=4)
        self.conv_mod_13 = MininetV2Module_dil(128, 3, strides=1, dilation_rate=8)
        self.conv_mod_14 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.conv_mod_15 = MininetV2Module(64, 3, strides=1, dilation_rate=1)

        self.up1 = MininetV2Upsample(64)
        """
        self.basemod = MiniNetv2(num_classes=2, include_top=False)
        #self.classify = MininetV2Upsample(num_classes)
        self.include_top=include_top
        self.globav = tf.keras.layers.GlobalAveragePooling2D()
        self.densfin = tf.keras.layers.Dense(1,activation=tf.nn.sigmoid)
        
        

    def call(self, inputs, training=True, reshape=True):
        """
        x1 = self.down1(inputs, training=training)
        x2 = self.down2(x1, training=training)
        x = self.conv_mod_1(x2, training=training)
        x = self.conv_mod_2(x, training=training)
        x = self.conv_mod_3(x, training=training)
        x = self.conv_mod_4(x, training=training)
        x3 = self.conv_mod_5(x, training=training)
        x = self.down3(x3, training=training)
        x = self.conv_mod_6(x, training=training)
        x = self.conv_mod_7(x, training=training)
        x = self.conv_mod_8(x, training=training)
        x = self.conv_mod_9(x, training=training)
        x = self.conv_mod_10(x, training=training)
        x = self.conv_mod_11(x, training=training)
        x = self.conv_mod_12(x, training=training)
        x = self.conv_mod_13(x, training=training)
        """
        x = self.basemod(inputs,training=training)
        x = self.globav(x, training=training)
        x = self.densfin(x)
        """      
        if self.include_top:
            
            x = self.up1(x, training=training)

            aux = self.down_b2(self.down_b(inputs, training=training))
            x = x + aux

            x = self.conv_mod_14(x, training=training)
            x = self.conv_mod_15(x, training=training)
            
            x = self.classify(x, training=training, last=True)
            if reshape:
                x = reshape_into(x, inputs)
            x = tf.keras.activations.softmax(x, axis=-1)
        """

        

        return x
