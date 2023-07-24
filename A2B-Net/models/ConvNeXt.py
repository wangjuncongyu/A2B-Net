import tensorflow as tf
import tensorflow.keras.layers as layers

class ConvNeXt_Block(layers.Layer):
    r""" ConvNeXt Block.
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = layers.DepthwiseConv2D(kernel_size=7, padding='same')  # depthwise conv
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        # pointwise/1x1 convs, implemented with linear layers
        self.pwconv1 = layers.Dense(4 * dim)
        self.act = layers.Activation('gelu')
        self.pwconv2 = layers.Dense(dim)
        self.drop_path = DropPath(drop_path)
        self.dim = dim
        self.layer_scale_init_value = layer_scale_init_value

    def build(self, input_shape):
        self.gamma = tf.Variable(
            initial_value=self.layer_scale_init_value * tf.ones((self.dim)),
            trainable=True,
            name='_gamma')
        self.built = True

    def call(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x

        x = input + self.drop_path(x)
        return x

class Downsample_Block(layers.Layer):
    """The Downsample Block in ConvNeXt

        Args:
            dim (int): number of channels
    """

    def __init__(self, dim):
        super().__init__()
        self.LN = layers.LayerNormalization(epsilon=1e-6)
        self.conv = layers.Conv2D(dim, kernel_size=2, strides=2)

    def build(self, input_shape):
        self.built = True

    def call(self, x):
        x = self.LN(x)
        x = self.conv(x)
        return x

class DropPath(tf.keras.layers.Layer):
    """The Drop path in ConvNeXt

        Reference:
            https://github.com/rishigami/Swin-Transformer-TF/blob/main/swintransformer/model.py
    """
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        return self._drop_path(x, self.drop_prob, training)

    
    def _drop_path(self, inputs, drop_prob, is_training):
        if (not is_training) or (drop_prob == 0.):
            return inputs

        # Compute keep_prob
        keep_prob = 1.0 - drop_prob

        # Compute drop_connect tensor
        random_tensor = keep_prob
        shape = (tf.shape(inputs)[0],) + (1,) * \
            (len(tf.shape(inputs)) - 1)
        random_tensor += tf.random.uniform(shape, dtype=inputs.dtype)
        binary_tensor = tf.floor(random_tensor)
        output = tf.math.divide(inputs, keep_prob) * binary_tensor
        return output

class ConvNeXtNet(tf.keras.Model):
    """ Function to construct the ConvNeXt Model
        
        Args:
            input_shape (tuple): (Width, Height , Channels)
            depths (list): a list of size 4. denoting each stage's depth
            dims (list): a list of size 4. denoting number of kernel's in each stage
            num_classes (int): the number of classes
            drop_path (float): Stochastic depth rate. Default: 0.0
            layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        Returns:
            ConvNeXt model: an instance of tf.keras.Model
    """
    def __init__(self, depths=[3, 3, 9, 3], dims=[64, 128, 256, 512], drop_path=0., layer_scale_init_value=1e-6):
        super(ConvNeXtNet, self).__init__()
      
        # Stem + res2
        self.y0 = layers.Conv2D(dims[0], kernel_size=4, strides=2, padding="same")
        self.norm0 = layers.LayerNormalization(epsilon=1e-6)
        
        self.group1 = []
        for i in range(depths[0]):
            self.group1.append(ConvNeXt_Block(dims[0], drop_path, layer_scale_init_value))

        # downsample + res3
        self.ds2 = Downsample_Block(dims[1])
        self.group2 = []
        for i in range(depths[1]):
            self.group2.append(ConvNeXt_Block(dims[1], drop_path, layer_scale_init_value))

        # downsample + res4
        self.ds3 = Downsample_Block(dims[2])
        self.group3 = []
        for i in range(depths[2]):
            self.group3.append(ConvNeXt_Block(dims[2], drop_path, layer_scale_init_value))
    
        # downsample + res5
        self.ds4 = Downsample_Block(dims[3])
        self.group4 = []
        for i in range(depths[3]):
            self.group4.append(ConvNeXt_Block(dims[3], drop_path, layer_scale_init_value))

        
        # final norm layer
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = layers.LayerNormalization(epsilon=1e-6)
        self.norm4 = layers.LayerNormalization(epsilon=1e-6)
        
        #lateral layers
        self.L1 = layers.Conv2D(256, (1, 1), (1, 1), padding="same")
        self.L2 = layers.Conv2D(256, (1, 1), (1, 1), padding="same")
        self.L3 = layers.Conv2D(256, (1, 1), (1, 1), padding="same")
        self.L4 = layers.Conv2D(256, (1, 1), (1, 1), padding="same")
        
        #smooth layers     
        self.S1 = layers.Conv2D(256, (3, 3), (1, 1),  padding="same")  
        self.S2 = layers.Conv2D(256, (3, 3), (1, 1),  padding="same")
        self.S3 = layers.Conv2D(256, (3, 3), (1, 1),  padding="same")
        
        self.norm_out = layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, inputs, training=True):
        C0= self.norm0(self.y0(inputs))
        
        C1 = C0
        for op in self.group1:
            C1 = op(C1)
        C1 = self.norm1(C1)
        
        C2 = self.ds2(C1)
        for op in self.group2:
            C2 = op(C2)
        C2 = self.norm2(C2)
        
        C3 = self.ds3(C2)
        for op in self.group3:
            C3 = op(C3)
        C3 = self.norm3(C3)
        
        C4 = self.ds4(C3)
        for op in self.group4:
            C4 = op(C4)
     
        C4 = self.norm4(C4)
        #print(C4.shape)
       
        #top-->down
        P4 = self.L1(C4)
        P3 = self.S3(self._upsample_add(P4, self.L2(C3)))#s8
        P2 = self.S2(self._upsample_add(P3, self.L3(C2)))#s4
        P1 = self.S1(self._upsample_add(P2, self.L4(C1)))#s2
       
        return self.norm_out(P1)
        
    
    def _upsample_add(self, x, y):
        _, H, W, C = y.shape
        x = tf.image.resize(x, size=(H, W), method='bilinear')
        return layers.Add()([x, y])

if __name__ == '__main__':
   
    net = ConvNeXtNet()
    input = tf.random.normal((1, 512, 512, 3))
    output = net(input)
    print(output.shape)
        
        
      
