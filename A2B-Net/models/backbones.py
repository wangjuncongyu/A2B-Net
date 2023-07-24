# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 16:26:49 2021

@author: wjcongyu
"""
import tensorflow as tf
from tensorflow.keras import layers as KL
from .residual_block import make_basic_block_layer, make_bottleneck_layer
from .hrnet import HRnet
from .ResNest import ResNest
from .ConvNeXt import ConvNeXtNet
from .SwinTrans import SwinTransformer
class BackboneFactory(object):
    def get_backbone(self, backbone_name):
        print('Building backbone:', backbone_name)
        if backbone_name == 'ResNet-18':
            return ResNetTypeI(layer_params=[2, 2, 2, 2])
        elif backbone_name == 'ResNet-34':            
            return ResNetTypeI(layer_params=[3, 4, 6, 3])
        elif backbone_name == 'ResNet-50':
            return ResNetTypeII(layer_params=[3, 4, 6, 3])
        elif backbone_name == 'ResNet-101':
            return ResNetTypeII(layer_params=[3, 4, 23, 3])
        elif backbone_name == 'ResNet-152':
            return ResNetTypeII(layer_params=[3, 8, 36, 3])
        elif backbone_name == 'ResNet-18-UNet':
            return ResNetTypeI_UNet(layer_params=[2, 2, 2, 2])
        elif backbone_name == 'UNet':
            return UNet(filters=[64,128,256,512])
       
        elif backbone_name == 'A2BNet-B':
            return A2Net( out_filters=[64, 128, 256, 512], \
                            dilate_filter_scales=[2, 2, 2, 2],\
                            dilates=[4, 4, 4, 4])
        elif backbone_name == 'A2BNet-L':
            return A2Net( out_filters=[96, 192, 384, 768], \
                            dilate_filter_scales=[2, 2, 2, 2],\
                            dilates=[8, 8, 4, 4])
        elif backbone_name == 'A2BNet-S':
            return A2Net( out_filters=[32, 64, 128, 256], \
                            dilate_filter_scales=[2, 2, 2, 2],\
                            dilates=[4, 4, 4, 4])
        elif backbone_name == 'A2BNet-Sv2':
            return A2Net( out_filters=[32, 64, 96, 192], \
                            dilate_filter_scales=[2, 2, 2, 2],\
                            dilates=[4, 4, 4, 4])
        elif backbone_name == 'A2BNet-T':
            return A2Net( out_filters=[16, 32, 64, 128], \
                            dilate_filter_scales=[2, 2, 2, 2],\
                            dilates=[2, 2, 2, 2])
        elif backbone_name == 'ANet-L':
            return ANet( out_filters=[128, 256, 512, 1024], \
                            dilate_filter_scales=[2, 2, 2, 2],\
                            dilates=[4, 4, 4, 4])
        elif backbone_name == 'HRNet-w18':
            return HRnet((512, 512, 3),'hrnetv2_w18')
        elif backbone_name == 'HRNet-w32':
            return HRnet((512, 512, 3),'hrnetv2_w32')
        elif backbone_name == 'HRNet-w48':
            return HRnet((512, 512, 3),'hrnetv2_w48')
        elif backbone_name == 'ResNest-50':
            return ResNest(blocks_set=[3, 4, 6, 3], stem_width=32).build()
        elif backbone_name == 'ResNest-101':
            return ResNest(blocks_set=[3, 4, 23, 3], stem_width=64).build()
        elif backbone_name == 'ConvNeXt-B':
            return ConvNeXtNet(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
        elif backbone_name == 'ConvNeXt-S':
            return ConvNeXtNet(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768])
        elif backbone_name == 'ConvNeXt-L':
            return ConvNeXtNet(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536])
        elif backbone_name == 'ConvNeXt-T':
            return ConvNeXtNet(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])
        elif backbone_name == 'Swin-B':
            return SwinTransformer(model_name='swin_base_512')
        elif backbone_name == 'Swin-S':
            return SwinTransformer(model_name='swin_small_512')
        elif backbone_name == 'Swin-T':
            return SwinTransformer(model_name='swin_tiny_512')
        elif backbone_name == 'Swin-L':
            return SwinTransformer(model_name='swin_large_512')
        

class ConvBnRelu(tf.keras.Model):
    def __init__(self, nfilter, kernel=(3, 3), stride=(1, 1), dilation_rate = (1, 1)):
        super(ConvBnRelu, self).__init__()
        self.x = KL.Conv2D(nfilter, kernel, stride, dilation_rate = dilation_rate, padding='same')  
        self.bn = KL.LayerNormalization(axis=-1)     
        self.ac = KL.ReLU()
        
    def call(self, x, training):
        return self.ac(self.bn(self.x(x), training))
        
class ResNetTypeI(tf.keras.Model):
    def __init__(self, layer_params):
        super(ResNetTypeI, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.LayerNormalization()       

        #bottom-->up layers
        self.C1 = make_basic_block_layer(filter_num=64,
                                             blocks=layer_params[0])
        self.C2 = make_basic_block_layer(filter_num=128,
                                             blocks=layer_params[1],
                                             stride=2)
        self.C3 = make_basic_block_layer(filter_num=256,
                                             blocks=layer_params[2],
                                             stride=2)
        self.C4 = make_basic_block_layer(filter_num=512,
                                             blocks=layer_params[3],
                                             stride=2)

        #smooth layers      
        self.S4 = KL.Conv2D(256, (3, 3), (1, 1),  padding="same")
        
        #lateral layers
        self.L1 = KL.Conv2D(256, (1, 1), (1, 1),  padding="valid")
        self.L2 = KL.Conv2D(256, (1, 1), (1, 1),  padding="valid")
        self.L3 = KL.Conv2D(256, (1, 1), (1, 1),  padding="valid")
        self.L4 = KL.Conv2D(256, (1, 1), (1, 1),  padding="valid")
        self.L5 = KL.Conv2D(256, (1, 1), (1, 1),  padding="valid")
    
       
    def call(self, inputs, training=True, mask=None):
        C0 = tf.nn.relu(self.bn1(self.conv1(inputs), training = training))#stride 2
        #C0 = tf.nn.max_pool2d(C0, ksize=3, strides=2, padding='SAME')#stride 4
        #botton-->up
        C1 = self.C1(C0, training=training)#stride 8
        C2 = self.C2(C1, training=training)#stride 16
        C3 = self.C3(C2, training=training)#stride 32
        C4 = self.C4(C3, training=training)#stride 64
        
        #top-->down
        P4 = self.L1(C4)
        P3 = self._upsample_add(P4, self.L2(C3))#stride 16
        P2 = self._upsample_add(P3, self.L3(C2))#stride 8
        P1 = self._upsample_add(P2, self.L4(C1))#stride 4
        '''P0 = self._upsample_add(P1, self.L5(C0))#stride 4'''
        
        P1 = self.S4(P1)
        return P1
    
    def _upsample_add(self, x, y):
        _, H, W, C = y.shape
        x = tf.image.resize(x, size=(H, W), method='bilinear')
        return KL.Add()([x, y])
    

class ResNetTypeI_UNet(tf.keras.Model):
    def __init__(self, layer_params):
        super(ResNetTypeI_UNet, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")
        self.bn = tf.keras.layers.LayerNormalization(axis=-1)       

        #bottom-->up layers
        self.C1 = make_basic_block_layer(filter_num=64,
                                             blocks=layer_params[0])
        self.C2 = make_basic_block_layer(filter_num=128,
                                             blocks=layer_params[1],
                                             stride=2)
        self.C3 = make_basic_block_layer(filter_num=256,
                                             blocks=layer_params[2],
                                             stride=2)
        self.C4 = make_basic_block_layer(filter_num=512,
                                             blocks=layer_params[3],
                                             stride=2)

        self.S1 = KL.Conv2D(256, (3, 3), (1, 1), activation='relu', padding="same")
        self.bn1 = KL.LayerNormalization(axis=-1)     
        self.S2 = KL.Conv2D(128, (3, 3), (1, 1), activation='relu', padding="same")
        self.bn2 = KL.LayerNormalization(axis=-1)     
        self.S3 = KL.Conv2D(64, (3, 3), (1, 1), activation='relu', padding="same")
        self.bn3 = KL.LayerNormalization(axis=-1)     
      
        
       
    def call(self, inputs, training=True, mask=None):
        C0 = tf.nn.relu(self.bn(self.conv1(inputs), training = training))#stride 2
        #C0 = tf.nn.max_pool2d(C0, ksize=3, strides=2, padding='SAME')#stride 4
        #botton-->up
        C1 = self.C1(C0, training=training)
        C2 = self.C2(C1, training=training)
        C3 = self.C3(C2, training=training)
        C4 = self.C4(C3, training=training)
        
        #top-->down
                
        P4 = self.bn1(self.S1(self._upsample_concat(C4, C3)), training = training)
        P3 = self.bn2(self.S2(self._upsample_concat(P4, C2)), training = training)
        P2 = self.bn3(self.S3(self._upsample_concat(P3, C1)), training = training)
       
        return P2
    
    def _upsample_concat(self, x, y):
        _, H, W, C = y.shape
        x = tf.image.resize(x, size=(H, W), method='bilinear')
        return KL.Concatenate()([x, y])


class ResNetTypeII(tf.keras.Model):
    def __init__(self, layer_params):
        super(ResNetTypeII, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.LayerNormalization()
        
        self.C1 = make_bottleneck_layer(filter_num=64,
                                            blocks=layer_params[0])
        self.C2 = make_bottleneck_layer(filter_num=128,
                                            blocks=layer_params[1],
                                            stride=2)
        self.C3 = make_bottleneck_layer(filter_num=256,
                                            blocks=layer_params[2],
                                            stride=2)
        self.C4 = make_bottleneck_layer(filter_num=512,
                                            blocks=layer_params[3],
                                            stride=2)
      
        #lateral layers
        self.L1 = KL.Conv2D(256, (1, 1), (1, 1), padding="valid")
        self.L2 = KL.Conv2D(256, (1, 1), (1, 1), padding="valid")
        self.L3 = KL.Conv2D(256, (1, 1), (1, 1), padding="valid")
        self.L4 = KL.Conv2D(256, (1, 1), (1, 1), padding="valid")
        self.L5 = KL.Conv2D(256, (1, 1), (1, 1),  padding="valid")
        
        #smooth layers     
        self.S1 = KL.Conv2D(256, (3, 3), (1, 1),  padding="same")  
        self.S2 = KL.Conv2D(256, (3, 3), (1, 1),  padding="same")
        self.S3 = KL.Conv2D(256, (3, 3), (1, 1),  padding="same")
        
    def call(self, inputs, training=True, mask=None):
        C0 = tf.nn.relu(self.bn1(self.conv1(inputs), training = training))
        #C0 = tf.nn.max_pool2d(C0, ksize=3, strides=2, padding='SAME')
        #botton-->up
        C1 = self.C1(C0, training=training)
        C2 = self.C2(C1, training=training)
        C3 = self.C3(C2, training=training)
        C4 = self.C4(C3, training=training)
        
        #top-->down
        P4 = self.L1(C4)
        P3 = self.S3(self._upsample_add(P4, self.L2(C3)))#s8
        P2 = self.S2(self._upsample_add(P3, self.L3(C2)))#s4
        P1 = self.S1(self._upsample_add(P2, self.L4(C1)))#s2
       
        return P1
    
    def _upsample_add(self, x, y):
        _, H, W, C = y.shape
        x = tf.image.resize(x, size=(H, W), method='bilinear')
        return KL.Add()([x, y])
       

class DownSample(tf.keras.Model):
    def __init__(self, nfilter, kernel=(3, 3), stride=(1, 1)):
        super(DownSample, self).__init__()
        self.x1 = KL.Conv2D(nfilter, kernel, stride, padding='same')  
        self.bn1 = KL.LayerNormalization(axis=-1)     
        self.ac1 = KL.ReLU()
        self.x2 = KL.Conv2D(nfilter, kernel, stride, padding='same')  
        self.bn2 = KL.LayerNormalization(axis=-1)     
        self.ac2 = KL.ReLU()
        self.down = KL.MaxPooling2D(pool_size=(2, 2))
        
    def call(self, x, training):
        x1 = self.ac1(self.bn1(self.x1(x)))
        x2 = self.ac2(self.bn2(self.x2(x1)))
        return self.down(x2)
    
class UpSample(tf.keras.Model):
    def __init__(self, nfilter, kernel=(3, 3), stride=(1, 1)):
        super(UpSample, self).__init__()
        self.u1 = KL.Conv2DTranspose(nfilter, kernel, (2, 2), padding='same')  
        self.x1 = KL.Conv2D(nfilter, kernel, stride, padding='same')  
        self.bn1 = KL.LayerNormalization(axis=-1)     
        self.ac1 = KL.ReLU()
        self.x2 = KL.Conv2D(nfilter, kernel, stride, padding='same')  
        self.bn2 = KL.LayerNormalization(axis=-1)     
        self.ac2 = KL.ReLU()
       
    def call(self, x, training):
        u1 = self.u1(x)
        x1 = self.ac1(self.bn1(self.x1(u1)))
        x2 = self.ac2(self.bn2(self.x2(x1)))
        return x2
    
class UNet(tf.keras.Model):
    def __init__(self, filters=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        
        self.x = KL.Conv2D(filters[0], (3, 3), (1, 1), activation='relu', padding="same")
        self.bn = KL.LayerNormalization(axis=-1)  
        
        self.d1 = DownSample(filters[0])
        self.d2 = DownSample(filters[1])
        self.d3 = DownSample(filters[2])
        self.d4 = DownSample(filters[3])
        
        self.u1 = UpSample(filters[3])
        self.u2 = UpSample(filters[2])
        self.u3 = UpSample(filters[1])
        self.u4 = UpSample(filters[0])
       
        self.s1 = KL.Conv2D(filters[3], (3, 3), (1, 1), activation='relu', padding="same")
        self.bn1 = KL.LayerNormalization(axis=-1)  
        
        self.s2 = KL.Conv2D(filters[2], (3, 3), (1, 1), activation='relu', padding="same")
        self.bn2 = KL.LayerNormalization(axis=-1) 
        
        self.s3 = KL.Conv2D(filters[1], (3, 3), (1, 1), activation='relu', padding="same")
        self.bn3 = KL.LayerNormalization(axis=-1) 
        
        self.s4 = KL.Conv2D(filters[0], (3, 3), (1, 1), activation='relu', padding="same")
        self.bn4 = KL.LayerNormalization(axis=-1) 
        
       
    def call(self, x, training):
        x = self.bn(self.x(x))
        d1 = self.d1(x, training)#2
        d2 = self.d2(d1, training)#4
        d3 = self.d3(d2, training)#8
        d4 = self.d4(d3, training)#16
        
        u1 = self.u1(d4, training)
        
        c1 = KL.Concatenate()([u1, d3])
        c1 = self.bn1(self.s1(c1))
        
        u2 = self.u2(c1, training)
        c2 = KL.Concatenate()([u2, d2])
        c2 = self.bn2(self.s2(c2))
        
        u3 = self.u3(c2, training)
        c3 = KL.Concatenate()([u3, d1])
        c3 = self.bn3(self.s3(c3))
        
        '''u4 = self.u4(c3, training)
        c4 = KL.Concatenate()([u4, x])
        c4 = self.bn4(self.s4(c4), training)'''
       
        return c3



class AtrousAttentionBlock(tf.keras.Model):
    def __init__(self, output_filters, dilate_filter_scale=32, n_dilates=16):
        super(AtrousAttentionBlock, self).__init__() 
        self.n_dilates = n_dilates        
        dfilters = output_filters//dilate_filter_scale
        
       
        self.ks = []
        self.qs = []
        self.vs = []
        self.dwn_xs = []#reducing computaional costs, lile the bottle neck
        for i in range(n_dilates):
           self.dwn_xs.append(KL.Conv2D(dfilters, (1, 1), activation = 'relu', padding='same'))
           self.ks.append(KL.Conv2D(dfilters, (3, 3), dilation_rate = i+1, activation = 'relu', padding='same'))
           self.qs.append(KL.Conv2D(dfilters, (3, 3), dilation_rate = i+1, activation = 'relu', padding='same'))
           self.vs.append(KL.Conv2D(dfilters, (3, 3), dilation_rate = i+1, activation = 'relu', padding='same'))

        self.smooth_x = KL.Conv2D(output_filters, (3, 3), dilation_rate = 1, activation = 'relu', padding='same')

        self.norm_inputs = KL.LayerNormalization(axis=-1) 
        self.norm_updates = KL.LayerNormalization(axis=-1) 
        self.norm_outputs = KL.LayerNormalization(axis=-1) 
        
    def call(self, x, training=True):     
        x = self.norm_inputs(x)

        dxs = [x]

      
        for i in range(self.n_dilates):
            dx = self.dwn_xs[i](x)
            N, H, W, _ = dx.shape
           
            k = self.ks[i](dx)
            q = self.qs[i](dx)
            v = self.vs[i](dx)
           
            kq = k*q
            kq = tf.reshape(kq, (N, H*W, -1))
            attn =  tf.nn.softmax(kq, axis=-2)
           
            attn = tf.reshape(attn, (N, H, W, -1))
           
            updates = attn*v
            dxs.append(self.norm_updates(updates))
       
        di_concat = KL.Concatenate(axis=-1)(dxs)
        xsmooth = self.norm_outputs(self.smooth_x(di_concat) )  
     
        return xsmooth


class A2Net(tf.keras.Model):
    def __init__(self, out_filters=[32, 64, 128, 256, 512], dilate_filter_scales=[2, 2, 2, 2, 2], dilates=[6, 6, 6, 6, 6]):
        super(A2Net, self).__init__()
        self.operations = tf.keras.Sequential()
        self.operations.add( tf.keras.layers.Conv2D(filters=out_filters[0],
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same"))
        self.operations.add(tf.keras.layers.LayerNormalization())
       
        for i, nfilters in enumerate(out_filters):
            self.operations.add(AtrousAttentionBlock(nfilters, dilate_filter_scales[i], dilates[i]))
          
    def call(self, inputs, training=True):  
        x = self.operations(inputs)
        return x

class AtrousBlock(tf.keras.Model):
    def __init__(self, output_filters, dilate_filter_scale=32, n_dilates=16):
        super(AtrousBlock, self).__init__() 
        self.n_dilates = n_dilates        
        dfilters = output_filters//dilate_filter_scale
        
       
       
        self.vs = []
        self.dwn_xs = []#reducing computaional costs, lile the bottle neck
        for i in range(n_dilates):
           self.dwn_xs.append(KL.Conv2D(dfilters, (1, 1), activation = 'relu', padding='same'))
           self.vs.append(KL.Conv2D(dfilters, (3, 3), dilation_rate = i+1, activation = 'relu', padding='same'))

        self.smooth_x = KL.Conv2D(output_filters, (3, 3), dilation_rate = 1, activation = 'relu', padding='same')

        self.norm_inputs = KL.LayerNormalization(axis=-1) 
        self.norm_updates = KL.LayerNormalization(axis=-1) 
        self.norm_outputs = KL.LayerNormalization(axis=-1) 
        
    def call(self, x, training=True):     
        x = self.norm_inputs(x)

        dxs = [x]

      
        for i in range(self.n_dilates):
            dx = self.dwn_xs[i](x)
            N, H, W, _ = dx.shape
           
            v = self.vs[i](dx)
            dxs.append(self.norm_updates(v))
       
        di_concat = KL.Concatenate(axis=-1)(dxs)
        xsmooth = self.norm_outputs(self.smooth_x(di_concat) )  
     
        return xsmooth


class ANet(tf.keras.Model):
    def __init__(self, out_filters=[32, 64, 128, 256, 512], dilate_filter_scales=[2, 2, 2, 2, 2], dilates=[6, 6, 6, 6, 6]):
        super(ANet, self).__init__()
        self.operations = tf.keras.Sequential()
        self.operations.add( tf.keras.layers.Conv2D(filters=out_filters[0],
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same"))
        self.operations.add(tf.keras.layers.LayerNormalization())
       
        for i, nfilters in enumerate(out_filters):
            self.operations.add(AtrousBlock(nfilters, dilate_filter_scales[i], dilates[i]))
          
    def call(self, inputs, training=True):  
        x = self.operations(inputs)
        return x

        
         
        
