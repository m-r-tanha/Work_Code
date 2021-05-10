def Classification(data):

    import tensorflow as tf
    import pandas as pd

    from sklearn import preprocessing
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import scale, normalize, minmax_scale
    from tensorflow.contrib import rnn
    from tensorflow.contrib.layers import batch_norm as batch_norm
    from pandas import ExcelWriter
    import xlsxwriter
    
#        data_path=path_dir
#        data=pd.read_excel(data_path,sheet_name='data')
#        data=data.fillna(0.0)
#        data.rename(index=data.Cell,inplace=True)
#        data.drop('Cell',axis=1,inplace=True)


    Normalized_X=scale(data, axis=1)
    Mi=Normalized_X.min()
    Normalized_X=(-1*Mi)+Normalized_X

    # Training Parameters
    '''
    num_steps = 30
    batch_size = 100
    display_step = 5
    '''
    strides = 1
    k = 1

    # Network Parameters
    num_input = 31  #  data input (img shape: 28*28)
    num_hidden = 100
    num_classes = 6  #  total classes (0-9 digits)
    dropout = 0.7  # Dropout, probability to keep units

    # tf Graph input
    X = tf.placeholder(tf.float32, [None, num_input])
    #Y = tf.placeholder(tf.float32, [None, num_classes])
    #keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

    is_training = tf.placeholder(tf.bool, name='MODE')
    #is_training=True

    # Batch Normal
    def batch_norm_layer(inputT, is_training=True, scope=None):
        # Note: is_training is tf.placeholder(tf.bool) type
        return tf.cond(is_training,
                        lambda: batch_norm(inputT, is_training=True,
                        center=True, scale=True, activation_fn=tf.nn.relu, decay=0.9, scope=scope),
                        lambda: batch_norm(inputT, is_training=False,
                        center=True, scale=True, activation_fn=tf.nn.relu, decay=0.9,
                        scope=scope, reuse = True))
    #
    # Store layers weight & bias
    # The first three convolutional layer
    w_c_1 = tf.Variable(tf.random_normal([1, 3, 1, 28]))
    w_c_2 = tf.Variable(tf.random_normal([1, 3, 28, 56]))
    w_c_3 = tf.Variable(tf.random_normal([1, 3, 56, 112]))
    b_c_1 = tf.Variable(tf.zeros([28]))
    b_c_2 = tf.Variable(tf.zeros([56]))
    b_c_3 = tf.Variable(tf.zeros([112]))

    # The second three convolutional layer weights
    w_c_4 = tf.Variable(tf.random_normal([1, 3, 112, 224]))
    w_c_5 = tf.Variable(tf.random_normal([1, 3, 224, 448]))
    w_c_6 = tf.Variable(tf.random_normal([1, 3, 448, 896]))
    b_c_4 = tf.Variable(tf.zeros([224]))
    b_c_5 = tf.Variable(tf.zeros([448]))
    b_c_6 = tf.Variable(tf.zeros([896]))

    # Fully connected weight
    w_f_1 = tf.Variable(tf.random_normal([1 * 31 * 896, 1792])) # fully connected, 1*3*896 inputs, 2048 outputs
    w_f_2 = tf.Variable(tf.random_normal([1792, 896]))
    w_f_3 = tf.Variable(tf.random_normal([896, 448]))
    b_f_1 = tf.Variable(tf.zeros([1792]))
    b_f_2 = tf.Variable(tf.zeros([896]))
    b_f_3 = tf.Variable(tf.zeros([448]))

    # output layer weight
    w_out = tf.Variable(tf.random_normal([448, num_classes]))
    b_out = tf.Variable(tf.zeros([num_classes]))

    #
    # Define model
    x = tf.reshape(X, shape=[-1, 1, 31, 1])

    # first layer convolution
    conv1 = tf.nn.conv2d(x, w_c_1, strides=[1, 1, 1, 1], padding='SAME') + b_c_1
    conv1=batch_norm(conv1, is_training=True,center=True, scale=True, activation_fn=tf.nn.relu, decay=0.9, scope='conv1_bn')
    conv1 = tf.nn.relu(conv1)

    # second layer convolution
    conv2 = tf.nn.conv2d(conv1, w_c_2, strides=[1, strides, strides, 1], padding='SAME') + b_c_2
    conv2=batch_norm(conv2, is_training=True,center=True, scale=True, activation_fn=tf.nn.relu, decay=0.9, scope='conv2_bn')
    conv2 = tf.nn.relu(conv2)

    # third layer convolution
    conv3 = tf.nn.conv2d(conv2, w_c_3, strides=[1, strides, strides, 1], padding='SAME') + b_c_3
    conv3=batch_norm(conv3, is_training=True,center=True, scale=True, activation_fn=tf.nn.relu, decay=0.9, scope='conv3_bn')
    conv3 = tf.nn.relu(conv3)

    # first Max Pooling (down-sampling)
    pool_1 = tf.nn.max_pool(conv3, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    # fourth layer convolution
    conv4 = tf.nn.conv2d(pool_1, w_c_4, strides=[1, strides, strides, 1], padding='SAME') + b_c_4
    conv4=batch_norm(conv4, is_training=True,center=True, scale=True, activation_fn=tf.nn.relu, decay=0.9, scope='conv4_bn')
    conv4 = tf.nn.relu(conv4)
    conv4 = tf.nn.dropout(conv4, dropout)


    # fifth layer convolution
    conv5 = tf.nn.conv2d(conv4, w_c_5, strides=[1, strides, strides, 1], padding='SAME') + b_c_5
    conv5=batch_norm(conv5, is_training=True,center=True, scale=True, activation_fn=tf.nn.relu, decay=0.9, scope='conv5_bn')
    conv5 = tf.nn.relu(conv5)

    # sixth layer convolution
    conv6 = tf.nn.conv2d(conv5, w_c_6, strides=[1, strides, strides, 1], padding='SAME') + b_c_6
    conv6=batch_norm(conv6, is_training=True,center=True, scale=True, activation_fn=tf.nn.relu, decay=0.9, scope='conv6_bn')
    conv6 = tf.nn.relu(conv6)

    # second Max Pooling (down-sampling)
    pool_2 = tf.nn.max_pool(conv6, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    pool_2 = tf.reshape(pool_2, [-1,31,896])

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([num_hidden,num_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([num_classes]))
    }

    def RNN(x, weights, biases):

        lstm_cell = rnn.BasicLSTMCell(num_hidden,reuse=tf.AUTO_REUSE)

        # Get lstm cell output
        outputs, states = tf.nn.dynamic_rnn(lstm_cell, x , dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[:,-1], weights['out']) + biases['out']

    logits = RNN(pool_2, weights, biases)
    prediction = tf.nn.softmax(logits)


    saver = tf.train.Saver()

    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #config=config
    with tf.Session() as sess:

        saver.restore(sess,"D:/1 Research/HEsam/RCNN/TF_ModelwBN_for_BRCell.ckpt")
        out=sess.run(prediction, feed_dict={X: Normalized_X})


    writer = pd.ExcelWriter('payload_rcnn3.xlsx')
    out=pd.DataFrame(out)
    out.columns=['N','DS','SI','GD','SD','GI']
    out.to_excel(writer,'result')
    data.to_excel(writer,'data')
    writer.save()
    return out