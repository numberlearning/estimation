#!/usr/bin/env py thon
import warnings
warnings.filterwarnings('ignore')


import tensorflow as tf
from tensorflow.examples.tutorials import mnist
import numpy as np
import os
import random
from scipy import misc
import time
import sys
import load_input
from model_settings import learning_rate, batch_size, img_height, img_width, min_edge, max_edge, min_blobs_train, max_blobs_train, min_blobs_test, max_blobs_test # MT

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_boolean("read_attn", False, "enable attention for reader")

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

if not os.path.exists("model_runs"):
    os.makedirs("model_runs")

if sys.argv[1] is not None:
        model_name = sys.argv[1]

folder_name = "model_runs/" + model_name

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

start_restore_index = 0

sys.argv = [sys.argv[0], sys.argv[1], "true", "true", "true", "true", "true",
folder_name + "/classify_log.csv",
folder_name + "/classifymodel_" + str(start_restore_index) + ".ckpt",
folder_name + "/classifymodel_",
folder_name + "/zzzdraw_data_5000.npy",
"false", "true", "false",
"false", # restore
"true"]
print(sys.argv)

train_iters = 7000000#20000000000
eps = 1e-8 # epsilon for numerical stability
rigid_pretrain = True
log_filename = sys.argv[7]
settings_filename = folder_name + "/settings.txt"
load_file = sys.argv[8]
save_file = sys.argv[9]
draw_file = sys.argv[10]
pretrain = str2bool(sys.argv[11]) #False
classify = str2bool(sys.argv[12]) #True
pretrain_restore = False
translated = str2bool(sys.argv[13])
dims = [img_height, img_width]
img_size = dims[1]*dims[0] # canvas size
read_n = 15  # read glimpse grid width/height
read_size = read_n*read_n
output_size = max_blobs_train - min_blobs_train + 1 # QSampler output size
h_size = 250
restore = str2bool(sys.argv[14])
start_non_restored_from_random = str2bool(sys.argv[15])
# delta, sigma2
delta_1=max(dims[0],dims[1])/(read_n-1)
sigma2_1=delta_1*delta_1/4 # sigma=delta/2 

## BUILD MODEL ## 

REUSE = None

x = tf.placeholder(tf.float32,shape=(batch_size, img_size))
onehot_labels = tf.placeholder(tf.float32, shape=(batch_size, output_size))

def linear(x,output_dim):
    """
    affine transformation Wx+b
    assumes x.shape = (batch_size, num_features)
    """
    #JLM: small initial weights instead of N(0,1)
    w=tf.get_variable("w", [x.get_shape()[1], output_dim], initializer=tf.random_uniform_initializer(minval=-.1, maxval=.1))
    #w=tf.get_variable("w", [x.get_shape()[1], output_dim], initializer=tf.random_normal_initializer())
    b=tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
    return tf.matmul(x,w)+b

def filterbank(gx, gy, N):
    grid_i = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])
    mu_x_1 = gx + (grid_i - N / 2 + 0.5) * delta_1 # eq 19 batch_size x N
    mu_y_1 = gy + (grid_i - N / 2 + 0.5) * delta_1 # eq 20 batch_size x N
    a = tf.reshape(tf.cast(tf.range(dims[0]), tf.float32), [1, 1, -1]) # 1 x 1 x dims[0]
    b = tf.reshape(tf.cast(tf.range(dims[1]), tf.float32), [1, 1, -1]) # 1 x 1 x dims[1]

    mu_x_1 = tf.reshape(mu_x_1, [-1, N, 1]) # batch_size x N x 1
    mu_y_1 = tf.reshape(mu_y_1, [-1, N, 1])
    Fx_1 = tf.exp(-tf.square(a - mu_x_1) / (2*sigma2_1)) # batch_size x N x dims[0]
    Fy_1 = tf.exp(-tf.square(b - mu_y_1) / (2*sigma2_1)) # batch_size x N x dims[1]
    # normalize, sum over A and B dims
    Fx_1=Fx_1/tf.maximum(tf.reduce_sum(Fx_1,2,keep_dims=True),eps)
    Fy_1=Fy_1/tf.maximum(tf.reduce_sum(Fy_1,2,keep_dims=True),eps)
    return Fx_1,Fy_1

def attn_window(scope,N):

    gx=dims[0]/2
    gy=dims[1]/2
    gx=np.reshape([gx]*batch_size, [batch_size,1])
    gy=np.reshape([gy]*batch_size, [batch_size,1])
    Fx_1, Fy_1 = filterbank(gx, gy, N)
    return Fx_1, Fy_1, gx, gy


## READ ## 

def read(x):
    Fx_1, Fy_1, gx, gy = attn_window("read", read_n)
    stats = Fx_1, Fy_1
    new_stats = gx, gy
    def filter_img(img, Fx_1, Fy_1, N):
        Fxt_1 = tf.transpose(Fx_1, perm=[0,2,1])
        # img: 1 x img_size
        img = tf.reshape(img,[-1, dims[1], dims[0]])
        fimg_1 = tf.matmul(Fy_1, tf.matmul(img, Fxt_1))
        fimg_1 = tf.reshape(fimg_1,[-1, N*N])
        # normalization (if do norm, Pc will be nan)
        # scalar_1 = tf.reshape(tf.reduce_max(fimg_1, 1), [batch_size, 1])
        # fimg_1 = fimg_1/tf.reduce_max(fimg_1, 1, keep_dims=True)
        fimg = fimg_1
        return fimg

    xr = filter_img(x, Fx_1, Fy_1, read_n) # batch_size x (read_n*read_n)
    return xr, new_stats # concat along feature axis

## STATE VARIABLES ##############
# initial states
r, stats = read(x)
rr=r
maxr=tf.reduce_max(rr,1, keep_dims=True)
classifications = list()

with tf.variable_scope("hidden",reuse=REUSE):
    hidden = tf.nn.relu(linear(r, h_size))
with tf.variable_scope("output",reuse=REUSE):
    classification = tf.nn.softmax(linear(hidden, output_size))
    classifications.append({
        "classification":classification,
        "r":r,
        })
        
REUSE=True

## LOSE FUNCTION ################################
predquality = -tf.reduce_sum(tf.log(classification + 1e-5) * onehot_labels, 1) # cross-entropy
predcost = tf.reduce_mean(predquality)

correct = tf.arg_max(onehot_labels, 1)
prediction = tf.arg_max(classification, 1)

# all-knower
R = tf.cast(tf.equal(correct, prediction), tf.float32)
reward = tf.reduce_mean(R)

def evaluate():
    data = load_input.InputData()
    data.get_test(1, min_blobs_test, max_blobs_test) # MT
    batches_in_epoch = len(data.images) // batch_size
    accuracy = 0
    sumlabels = np.zeros(output_size)
