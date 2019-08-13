import cPickle
import numpy as np
import sys
import tensorflow as tf

###################################
#   10K Hz for sin and cos waves  #
#   500K Hz for sampling rate     #
###################################


def readInput(filename):
  fo = open(filename, 'rb')
  IQ_file = cPickle.load(fo)
  fo.close()

  return IQ_file


def constructPlane(IQ_file, start_idx, channel_length, num_samples):
  plane = np.ones((128,128,channel_length))
 
  pixel_intensity = 100
  for i in range(start_idx, start_idx+num_samples):
    offset = i - start_idx

    idx_I = int((IQ_file[i][0]+0.0005)*128000)
    idx_Q = int((IQ_file[i][1]+0.0005)*128000)

    if idx_I >= 127:
      idx_I =127

    if idx_Q >= 127:
      idx_Q =127

    plane[idx_I][idx_Q][offset%channel_length] = pixel_intensity + offset

  return plane

  #for i in range(0,len(IQ_file)):
  #for i in range(0,10):
  # 
  #  if IQ_file[i][sel] > -1 and IQ_file[i][sel] <= -0.8:
  #    hist_I[0] = int(hist_I[0]) + 1
  #  elif IQ_file[i][sel] > -0.8 and IQ_file[i][sel] <= -0.6:
  #    hist_I[1] = int(hist_I[1]) + 1
  #  elif IQ_file[i][sel] > -0.6 and IQ_file[i][sel] <= -0.4:
  #    hist_I[2] = int(hist_I[2]) + 1
  #  elif IQ_file[i][sel] > -0.4 and IQ_file[i][sel] <= -0.2:
  #    hist_I[3] = int(hist_I[3]) + 1
  #  elif IQ_file[i][sel] > -0.2 and IQ_file[i][sel] <= 0.0:
  #    hist_I[4] = int(hist_I[4]) + 1
  #  elif IQ_file[i][sel] > 0.0 and IQ_file[i][sel] <= 0.2:
  #    hist_I[5] = int(hist_I[5]) + 1
  #  elif IQ_file[i][sel] > 0.2 and IQ_file[i][sel] <= 0.4:
  #    hist_I[6] = int(hist_I[6]) + 1
  #  elif IQ_file[i][sel] > 0.4 and IQ_file[i][sel] <= 0.6:
  #    hist_I[7] = int(hist_I[7]) + 1
  #  elif IQ_file[i][sel] > 0.6 and IQ_file[i][sel] <= 0.8:
  #    hist_I[8] = int(hist_I[8]) + 1
  #  else:
  #    hist_I[9] = int(hist_I[9]) + 1
  #  print "(%d, %d)" % (idx_I,idx_Q)


def displayFrame(plane, channel_length):

  for k in range(0,channel_length):
    for i in range(0,128):
      for j in range(0,128):
        if plane[i][j][k] == 0.0:
          print " ",
        else:
          print "%d" % plane[i][j][k],

      print " ---"

def batchRead(input_data, channel_length, batch_size):
  num_samples = 1000
  batch_idx = np.random.randint(0,len(input_data)-channel_length*num_samples, batch_size)  

  #print len(batch_idx)

  planes = constructPlane(input_data, batch_idx[0], channel_length, num_samples)
  for i in range(1,batch_size):
    planes = np.vstack((planes,constructPlane(input_data, batch_idx[i], channel_length, num_samples)))

  return planes
 
def getLabels(K, mini_batch):
  y = np.zeros((mini_batch,K))
  for i in range(0,10):
    y[i][0] = 1

  for i in range(10,20):
    y[i][1] = 1

  for i in range(20,30):
    y[i][2] = 1

  for i in range(30,40):
    y[i][3] = 1

  return y


def getTrainingData(IQ_file_1, IQ_file_2, IQ_file_3, IQ_file_4, channel_length,  mini_batch):
  planes_1 = batchRead(IQ_file_1, channel_length, mini_batch/4)
  planes_2 = batchRead(IQ_file_2, channel_length, mini_batch/4)
  planes_3 = batchRead(IQ_file_3, channel_length, mini_batch/4)
  planes_4 = batchRead(IQ_file_4, channel_length, mini_batch/4)
  
  planes = np.vstack((planes_1,planes_2))
  planes = np.vstack((planes,planes_3))
  planes = np.vstack((planes,planes_4))

  planes = planes.reshape((mini_batch,128,128,channel_length))
  #print planes.shape

  return planes
 

if __name__ == '__main__':
  

  print '===== Start loading datasets ====='
  #fo = open(sys.argv[1], 'rb')
  IQ_file_1 = readInput("/home/hhwu/tensorflow_work/cs231n/signal_cnn/test_input_1.bin")
  print "IQ_file_1: %d" % len(IQ_file_1)
  IQ_file_2 = readInput("/home/hhwu/tensorflow_work/cs231n/signal_cnn/test_input_2.bin")
  print "IQ_file_2: %d" % len(IQ_file_2)
  IQ_file_3 = readInput("/home/hhwu/tensorflow_work/cs231n/signal_cnn/test_input_3.bin")
  print "IQ_file_3: %d" % len(IQ_file_3)
  IQ_file_4 = readInput("/home/hhwu/tensorflow_work/cs231n/signal_cnn/test_input_4.bin")
  print "IQ_file_4: %d" % len(IQ_file_4)



  #########################################
  #  Configuration of CNN architecture    #
  #########################################
  mini_batch = 40
  channel_length = 10

  K = 4 # number of classes
  NUM_FILTER_1 = 8
  NUM_FILTER_2 = 8
  NUM_FILTER_3 = 16 
  NUM_FILTER_4 = 16 
  NUM_FILTER_5 = 32 
  NUM_FILTER_6 = 32 

  NUM_NEURON_1 = 128
  NUM_NEURON_2 = 128

  DROPOUT_PROB_1 = 1.0
  DROPOUT_PROB_2 = 1.0

  LEARNING_RATE = 5e-2
 
  reg = 1e-3 # regularization strength


  # Dropout probability
  keep_prob_1 = tf.placeholder(tf.float32)
  keep_prob_2 = tf.placeholder(tf.float32)






  # initialize parameters randomly
  X  = tf.placeholder(tf.float32, shape=[None, 128,128,channel_length])
  Y_ = tf.placeholder(tf.float32, shape=[None,K])

  W1 = tf.Variable(tf.truncated_normal([3,3,channel_length,NUM_FILTER_1], stddev=0.1))
  b1 = tf.Variable(tf.ones([NUM_FILTER_1])/10)

  W2 = tf.Variable(tf.truncated_normal([3,3,NUM_FILTER_1,NUM_FILTER_2], stddev=0.1))
  b2 = tf.Variable(tf.ones([NUM_FILTER_2])/10)

  W3 = tf.Variable(tf.truncated_normal([3,3,NUM_FILTER_2,NUM_FILTER_3], stddev=0.1))
  b3 = tf.Variable(tf.ones([NUM_FILTER_3])/10)

  W4 = tf.Variable(tf.truncated_normal([3,3,NUM_FILTER_3,NUM_FILTER_4], stddev=0.1))
  b4 = tf.Variable(tf.ones([NUM_FILTER_4])/10)

  W5 = tf.Variable(tf.truncated_normal([3,3,NUM_FILTER_4,NUM_FILTER_5], stddev=0.1))
  b5 = tf.Variable(tf.ones([NUM_FILTER_5])/10)

  W6 = tf.Variable(tf.truncated_normal([3,3,NUM_FILTER_5,NUM_FILTER_6], stddev=0.1))
  b6 = tf.Variable(tf.ones([NUM_FILTER_6])/10)

  W7 = tf.Variable(tf.truncated_normal([16*16*NUM_FILTER_6,NUM_NEURON_1], stddev=0.1))
  b7 = tf.Variable(tf.ones([NUM_NEURON_1])/10)

  W8 = tf.Variable(tf.truncated_normal([NUM_NEURON_1,NUM_NEURON_2], stddev=0.1))
  b8 = tf.Variable(tf.ones([NUM_NEURON_2])/10)


  W9 = tf.Variable(tf.truncated_normal([NUM_NEURON_2,K], stddev=0.1))
  b9 = tf.Variable(tf.ones([K])/10)



  #===== architecture =====#
  Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME')+b1)
  #Y1 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME')+b1), ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
  Y2 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1,1,1,1], padding='SAME')+b2), ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

  Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1,1,1,1], padding='SAME')+b3)
  #Y3 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1,1,1,1], padding='SAME')+b3), ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
  Y4 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(Y3, W4, strides=[1,1,1,1], padding='SAME')+b4), ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

  Y5 = tf.nn.relu(tf.nn.conv2d(Y4, W5, strides=[1,1,1,1], padding='SAME')+b5)
  #Y5 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(Y4, W5, strides=[1,1,1,1], padding='SAME')+b5), ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
  Y6 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(Y5, W6, strides=[1,1,1,1], padding='SAME')+b6), ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

  YY = tf.reshape(Y6, shape=[-1,16*16*NUM_FILTER_6])

  Y7 = tf.nn.relu(tf.matmul(YY,W7)+b7)
  Y8 = tf.nn.relu(tf.matmul(Y7,W8)+b8)

  Y  = tf.nn.softmax(tf.matmul(Y8,W9)+b9)

  global_step = tf.Variable(0, trainable=False)
  starter_learning_rate = LEARNING_RATE
  learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                             100000, 0.95, staircase=True)

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
  max_test_acc = 0
  #num_input_data =tr_data10.shape[0]
  y = getLabels(K, mini_batch)
  for itr in xrange(10000):
    x = getTrainingData(IQ_file_1, IQ_file_2, IQ_file_3, IQ_file_4, channel_length,  mini_batch)
    sess.run(train_step, feed_dict={X: x, Y_: y, keep_prob_1: DROPOUT_PROB_1, keep_prob_2: DROPOUT_PROB_2})
 
 
    if itr % 10 == 0:
      print "Iter %d:  learning rate: %f  dropout: (%.1f %.1f) cross entropy: %f  accuracy: %f" % (itr,
                                                              learning_rate.eval(session=sess, feed_dict={X: x, Y_: y, 
                                                                                                          keep_prob_1: DROPOUT_PROB_1, 
                                                                                                          keep_prob_2: DROPOUT_PROB_2}),
                                                              DROPOUT_PROB_1,
                                                              DROPOUT_PROB_2,
                                                              cross_entropy.eval(session=sess, feed_dict={X: x, Y_: y, 
                                                                                                          keep_prob_1: DROPOUT_PROB_1, 
                                                                                                          keep_prob_2: DROPOUT_PROB_2}),
                                                              accuracy.eval(session=sess, feed_dict={X: x, Y_: y, 
                                                                                                     keep_prob_1: DROPOUT_PROB_1, 
                                                                                                     keep_prob_2: DROPOUT_PROB_2}))
