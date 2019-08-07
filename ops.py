import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from tensorflow.contrib.layers.python.layers import instance_norm
from tensorflow.layers import average_pooling2d
from tensorflow.layers import max_pooling2d
from tensorflow.image import resize_bilinear

#the implements of leakyRelu
def lrelu(x , alpha = 0.2 , name="LeakyReLU"):
    return tf.maximum(x , alpha*x)

def conv2d(x, W , b , strides=1, padding='SAME'):

    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x , W , strides=[1, strides , strides, 1], padding=padding)
    x = tf.nn.bias_add(x, b)

    return x

def de_conv(x , W , b , out_shape):
    with tf.name_scope('deconv') as scope:
        deconv = tf.nn.conv2d_transpose(x , W ,
        out_shape , [1 , 2 ,2 , 1] , padding='SAME', name=None)
        out = tf.nn.bias_add(deconv , b)
        return out

def fully_connect(x , weight , bias):
    return tf.add(tf.matmul(x , weight) , bias)

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()

    return tf.concat([x , y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2] , y_shapes[3]])], 3)

def batch_normal(input , scope="scope" , reuse=False):
    return batch_norm(input , epsilon=1e-5, decay=0.9 , scale=True, scope=scope , reuse = reuse)

def instance_normal(input , scope="scope" , reuse=False):
    return instance_norm(input , epsilon=1e-5 , scale=True, scope=scope , reuse = reuse)



def max_pool(input,scope="scope", pool = [2,2], stride = 2, reuse=False):
    return max_pooling2d(input,pool,stride,padding='VALID')
def mean_pool(input,scope="scope", pool = [2,2], stride = 2, reuse=False):
    return average_pooling2d(input,pool,stride,padding='VALID')

def bilinear_up(input,multiply = 2):
    shapes = tf.shape(input)
    return resize_bilinear(input,[shapes[1]*multiply,shapes[1]*multiply])


def sep_conv2d(x, W1, W2 , b , strides=1, padding='SAME'):
    x = tf.nn.separable_conv2d(x , W1 , W2, strides=[1, strides , strides, 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    return x

