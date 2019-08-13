#!/usr/bin/python

import cPickle
import numpy as np
import os
import csv
from skimage import io
from skimage import transform
import tensorflow as tf
from multiprocessing.pool import ThreadPool
#import matplotlib.pyplot as plt

def cropImg(target_img):
  ###############################
  #       shape[0]: height      #
  #       shape[1]: width       #
  ###############################

  #Grayscale Img and convert it to RGB
  if len(target_img.shape) == 2:
    RGB_img = np.zeros((target_img.shape[0],target_img.shape[1],3))
    RGB_img[:,:,0] = target_img
    RGB_img[:,:,1] = target_img
    RGB_img[:,:,2] = target_img

    target_img = RGB_img


  #print target_img.shape
  #print absfile
  if target_img.shape[0] < target_img.shape[1]:
    width = int(target_img.shape[1]*256/target_img.shape[0])
    #if width < 224:
    #  width = 224

    offset = int((width-224)/2)

    target_img = transform.resize(target_img, (256,width,3))
    target_img = target_img[16:240, offset:224+offset, :]
  else:
    height = int(target_img.shape[0]*256/target_img.shape[1])
    #if height < 224:
    #  height = 224

    offset = int((height-224)/2)

    target_img = transform.resize(target_img, (height,256,3))
    target_img = target_img[offset:224+offset, 16:240, :]

  return target_img


def batchCroppedImgRead(thread_name, dirpath, image_name, partial_batch_idx):
  #print "%s is cropping the images..." % thread_name
  img_batch = []
  label_batch = np.empty(1)

  for i in partial_batch_idx:
    absfile = os.path.join(dirpath, image_name[i])
    target_img = io.imread(absfile)

    croppedImg = cropImg(target_img)

    if len(img_batch) == 0:
      img_batch = croppedImg
      #label_batch = input_label[batch_idx[i]]
    else:
      img_batch = np.vstack((img_batch, croppedImg))
      #label_batch = np.hstack((label_batch, input_label[batch_idx[i]]))


  return img_batch

def batchRead(image_name, class_dict, pool):
  batch_idx = np.random.randint(0,len(image_name),mini_batch)
  dirpath = '/home/hhwu/ImageNet/train/'

  test_y = np.zeros((mini_batch,K))
  #print class_dict
  for i in range(0, len(batch_idx)):
    image_class_name = image_name[batch_idx[i]].split("_")[0]
    #print image_class_name
    #print class_dict[image_class_name]
    test_y[i][int(class_dict[image_class_name])] = 1

    #print "test_y[%d][%d] = %d" % (i,int(class_dict[image_class_name]),test_y[i][int(class_dict[image_class_name])])

  #img_batch = np.empty((224,224,3), dtype=np.float32)
  img_batch = []

  async_result_0 = pool.apply_async(batchCroppedImgRead, ("Thread-0", dirpath, image_name, batch_idx[:int(mini_batch/8)]))
  async_result_1 = pool.apply_async(batchCroppedImgRead, ("Thread-1", dirpath, image_name, batch_idx[int(mini_batch/8):int(2*mini_batch/8)]))
  async_result_2 = pool.apply_async(batchCroppedImgRead, ("Thread-2", dirpath, image_name, batch_idx[int(2*mini_batch/8):int(3*mini_batch/8)]))
  async_result_3 = pool.apply_async(batchCroppedImgRead, ("Thread-3", dirpath, image_name, batch_idx[int(3*mini_batch/8):int(4*mini_batch/8)]))
  async_result_4 = pool.apply_async(batchCroppedImgRead, ("Thread-4", dirpath, image_name, batch_idx[int(4*mini_batch/8):int(5*mini_batch/8)]))
  async_result_5 = pool.apply_async(batchCroppedImgRead, ("Thread-5", dirpath, image_name, batch_idx[int(5*mini_batch/8):int(6*mini_batch/8)]))
  async_result_6 = pool.apply_async(batchCroppedImgRead, ("Thread-6", dirpath, image_name, batch_idx[int(6*mini_batch/8):int(7*mini_batch/8)]))
  async_result_7 = pool.apply_async(batchCroppedImgRead, ("Thread-7", dirpath, image_name, batch_idx[int(7*mini_batch/8):]))

  img_batch    = async_result_0.get()
  return_val_1 = async_result_1.get()
  return_val_2 = async_result_2.get()
  return_val_3 = async_result_3.get()
  return_val_4 = async_result_4.get()
  return_val_5 = async_result_5.get()
  return_val_6 = async_result_6.get()
  return_val_7 = async_result_7.get()


  img_batch = np.vstack((img_batch, return_val_1))
  img_batch = np.vstack((img_batch, return_val_2))
  img_batch = np.vstack((img_batch, return_val_3))
  img_batch = np.vstack((img_batch, return_val_4))
  img_batch = np.vstack((img_batch, return_val_5))
  img_batch = np.vstack((img_batch, return_val_6))
  img_batch = np.vstack((img_batch, return_val_7))
    

  
  img_batch = img_batch.reshape(mini_batch,224,224,3)

  #print img_batch.shape
  #print class_dict

  #convert to one hot labels
  #train_y = np.zeros((mini_batch,K))
  #for i in range(mini_batch):
  #  train_y[i][label_batch[i]] = 1


  #return img_batch, label_batch
  #return img_batch, train_y
  return img_batch, test_y


def loadClassName(filename):
  class_name = []
  with open(filename, 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    i = 0
    for row in spamreader:
      class_name.append(row[1])
      i = i+1
      if i >= 1000:
        break

  class_dict = {}
  for i in xrange(0,len(class_name)):
    class_dict[class_name[i].replace('\'', '')] = i


  image_name = []
  for dirpath, dirnames, filenames in os.walk('/home/hhwu/ImageNet/train/'):
    print "dirpath: ", dirpath
    print "dirnames: ", dirnames
    print "The number of files: %d" % len(filenames)

    image_name = filenames

  print "The number of classes: %d" % len(class_name)
  return class_dict, image_name



if __name__ == '__main__':
  print '===== Start loading the mean of ILSVRC2012 ====='
  fo = open('mean.bin', 'rb')
  mean_img = cPickle.load(fo)
  fo.close()
  print mean_img.shape


  class_dict, image_name  = loadClassName('synset.csv')

  pool = ThreadPool(processes=8)
  print "Multi-threads begin!"


  #########################################
  #  Configuration of CNN architecture    #
  #########################################
  mini_batch = 128
  #num_training_imgs = tr_data10.shape[0]
  #epoch_num = num_training_imgs/mini_batch

  K = 1000 # number of classes
  NUM_FILTER_1  = 64
  NUM_FILTER_2  = 64

  NUM_FILTER_3  = 128
  NUM_FILTER_4  = 128
  NUM_FILTER_5  = 256
  NUM_FILTER_6  = 256
  NUM_FILTER_7  = 256

  NUM_FILTER_8  = 512
  NUM_FILTER_9  = 512
  NUM_FILTER_10 = 512

  NUM_FILTER_11 = 512
  NUM_FILTER_12 = 512
  NUM_FILTER_13 = 512

  NUM_NEURON_1 = 4096
  NUM_NEURON_2 = 4096

  DROPOUT_PROB_1 = 1.00

  LEARNING_RATE = 1e-1
 
  reg = 5e-4 # regularization strength


  # Dropout probability
  keep_prob_1 = tf.placeholder(tf.float32)
  keep_prob_2 = tf.placeholder(tf.float32)


  # initialize parameters randomly
  X  = tf.placeholder(tf.float32, shape=[None, 224,224,3])
  Y_ = tf.placeholder(tf.float32, shape=[None,K])

  W1  = tf.Variable(tf.truncated_normal([3,3,3,NUM_FILTER_1], stddev=0.1))
  b1  = tf.Variable(tf.ones([NUM_FILTER_1])/10)

  W2  = tf.Variable(tf.truncated_normal([3,3,NUM_FILTER_1,NUM_FILTER_2], stddev=0.1))
  b2  = tf.Variable(tf.ones([NUM_FILTER_2])/10)

  W3  = tf.Variable(tf.truncated_normal([3,3,NUM_FILTER_2,NUM_FILTER_3], stddev=0.1))
  b3  = tf.Variable(tf.ones([NUM_FILTER_3])/10)

  W4  = tf.Variable(tf.truncated_normal([3,3,NUM_FILTER_3,NUM_FILTER_4], stddev=0.1))
  b4  = tf.Variable(tf.ones([NUM_FILTER_4])/10)

  W5  = tf.Variable(tf.truncated_normal([3,3,NUM_FILTER_4,NUM_FILTER_5], stddev=0.1))
  b5  = tf.Variable(tf.ones([NUM_FILTER_5])/10)

  W6  = tf.Variable(tf.truncated_normal([3,3,NUM_FILTER_5,NUM_FILTER_6], stddev=0.1))
  b6  = tf.Variable(tf.ones([NUM_FILTER_6])/10)

  W7  = tf.Variable(tf.truncated_normal([3,3,NUM_FILTER_6,NUM_FILTER_7], stddev=0.1))
  b7  = tf.Variable(tf.ones([NUM_FILTER_7])/10)

  W8  = tf.Variable(tf.truncated_normal([3,3,NUM_FILTER_7,NUM_FILTER_8], stddev=0.1))
  b8  = tf.Variable(tf.ones([NUM_FILTER_8])/10)

  W9  = tf.Variable(tf.truncated_normal([3,3,NUM_FILTER_8,NUM_FILTER_9], stddev=0.1))
  b9  = tf.Variable(tf.ones([NUM_FILTER_9])/10)

  W10 = tf.Variable(tf.truncated_normal([3,3,NUM_FILTER_9,NUM_FILTER_10], stddev=0.1))
  b10 = tf.Variable(tf.ones([NUM_FILTER_10])/10)

  W11 = tf.Variable(tf.truncated_normal([3,3,NUM_FILTER_10,NUM_FILTER_11], stddev=0.1))
  b11 = tf.Variable(tf.ones([NUM_FILTER_11])/10)

  W12 = tf.Variable(tf.truncated_normal([3,3,NUM_FILTER_11,NUM_FILTER_12], stddev=0.1))
  b12 = tf.Variable(tf.ones([NUM_FILTER_12])/10)

  W13 = tf.Variable(tf.truncated_normal([3,3,NUM_FILTER_12,NUM_FILTER_13], stddev=0.1))
  b13 = tf.Variable(tf.ones([NUM_FILTER_13])/10)



  W14 = tf.Variable(tf.truncated_normal([7*7*NUM_FILTER_13,NUM_NEURON_1], stddev=0.1))
  b14 = tf.Variable(tf.ones([NUM_NEURON_1])/10)

  W15 = tf.Variable(tf.truncated_normal([NUM_NEURON_1,NUM_NEURON_2], stddev=0.1))
  b15 = tf.Variable(tf.ones([NUM_NEURON_2])/10)

  W16 = tf.Variable(tf.truncated_normal([NUM_NEURON_2,K], stddev=0.1))
  b16 = tf.Variable(tf.ones([K])/10)


  #===== architecture =====#
  Y1  = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME')+b1)
  Y2  = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1,1,1,1], padding='SAME')+b2), ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

  Y3  = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1,1,1,1], padding='SAME')+b3)
  Y4  = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(Y3, W4, strides=[1,1,1,1], padding='SAME')+b4), ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

  Y5  = tf.nn.relu(tf.nn.conv2d(Y4, W5, strides=[1,1,1,1], padding='SAME')+b5)
  Y6  = tf.nn.relu(tf.nn.conv2d(Y5, W6, strides=[1,1,1,1], padding='SAME')+b6)
  Y7  = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(Y6, W7, strides=[1,1,1,1], padding='SAME')+b7), ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

  Y8  = tf.nn.relu(tf.nn.conv2d(Y7, W8, strides=[1,1,1,1], padding='SAME')+b8)
  Y9  = tf.nn.relu(tf.nn.conv2d(Y8, W9, strides=[1,1,1,1], padding='SAME')+b9)
  Y10 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(Y9, W10, strides=[1,1,1,1], padding='SAME')+b10), ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

  Y11 = tf.nn.relu(tf.nn.conv2d(Y10, W11, strides=[1,1,1,1], padding='SAME')+b11)
  Y12 = tf.nn.relu(tf.nn.conv2d(Y11, W12, strides=[1,1,1,1], padding='SAME')+b12)
  Y13 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(Y12, W13, strides=[1,1,1,1], padding='SAME')+b13), ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

  YY = tf.reshape(Y13, shape=[-1,7*7*NUM_FILTER_13])


  Y14 = tf.nn.relu(tf.matmul(YY,W14)+b14)
  Y14_drop = tf.nn.dropout(Y14, keep_prob_1)

  Y15 = tf.nn.relu(tf.matmul(Y14_drop,W15)+b15)
  Y15_drop = tf.nn.dropout(Y15, keep_prob_1)

  Y  = tf.nn.softmax(tf.matmul(Y15_drop,W16)+b16)

  global_step = tf.Variable(0, trainable=False)
  starter_learning_rate = LEARNING_RATE
  learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                             100000, 0.9, staircase=True)

  diff = tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Y)
  reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  cross_entropy = tf.reduce_mean(diff) + reg*sum(reg_losses)

  correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


  # Passing global_step to minimize() will increment it at each step.
  #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
  train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cross_entropy, global_step=global_step)
 

  # Add ops to save and restore all the variables.
  saver = tf.train.Saver() 

  #learning_rate = tf.placeholder(tf.float32, shape=[])
  #train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  # Restore variables from disk.
  #saver.restore(sess, "./checkpoint/model_990000.ckpt")
  #print("Model restored.")

  #te_x, te_y = batchTestRead(te_data10, te_labels10)
  print '  Start training... '
  idx_start = 0
  epoch_counter = 0
  epoch_num     = 1

  max_test_acc = 0
  num_images = 1281167
  for itr in xrange(1000000):
    x, y = batchRead(image_name, class_dict, pool)
    #x, y = batchRead(tr_data10, tr_labels10, idx_start)
    sess.run(train_step, feed_dict={X: x, Y_: y, keep_prob_1: DROPOUT_PROB_1})
 
 
    if itr % 5 == 0:
      print "Iter %d:  learning rate: %f  dropout:%.1f cross entropy: %f  accuracy: %f" % (itr,
                                                              learning_rate.eval(session=sess, feed_dict={X: x, Y_: y, 
                                                                                                          keep_prob_1: DROPOUT_PROB_1}),
                                                              DROPOUT_PROB_1,
                                                              cross_entropy.eval(session=sess, feed_dict={X: x, Y_: y, 
                                                                                                          keep_prob_1: DROPOUT_PROB_1}),
                                                              accuracy.eval(session=sess, feed_dict={X: x, Y_: y, 
                                                                                                     keep_prob_1: DROPOUT_PROB_1}))


    if epoch_counter > int(num_images/mini_batch):
      print "Epoch %d" % epoch_num
      epoch_counter = 0
      epoch_num = epoch_num + 1
    else:
      epoch_counter = epoch_counter + 1
    #  test_acc = accuracy.eval(session=sess, feed_dict={X: te_x, Y_: te_y, keep_prob_1: 1.0, keep_prob_2: 1.0})

    #  if test_acc > max_test_acc:
    #    max_test_acc = test_acc

    #  print "Test Accuracy: %f (max: %f)" % (test_acc, max_test_acc) 
    #  test_result.write("Test Accuracy: %f (max: %f)" % (test_acc, max_test_acc))
    #  test_result.write("\n")

    #  epoch_counter += 1


    #if itr % 10000 == 0 and itr != 0:
    #  model_name = "./checkpoint/model_%d.ckpt" % itr
    #  save_path = saver.save(sess, model_name)
    #  #save_path = saver.save(sess, "./checkpoint/model.ckpt")
    #  print("Model saved in file: %s" % save_path)


    ##print "batch: ", idx_start
    #if idx_start+mini_batch >= len(tr_data10):
    #  idx_start = 0
    #else:
    #  idx_start += mini_batch


    #if itr % 50000 == 0 and itr != 0:
    #  DROPOUT_PROB_1 = 1.0
    #  DROPOUT_PROB_2 = 0.5


  #te_data10 = np.subtract(te_data10, center_img, casting='unsafe')
  #te_data10 = te_data10/255.0
  #te_x, te_y = batchTestRead(te_data10, te_labels10)
