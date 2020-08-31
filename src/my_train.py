import os
import sys
import re
import numpy as np
from datetime import datetime
import tensorflow as tf
import tensorflow.contrib.slim as slim

from src.nets import mobilenet_v1

#============================================================

def train_stem(data_dir, lfw_dir, pairs_txt, model_root, model_name, batch_size, 
               epoch_size, max_epochs, image_height, image_width, embedding_size, 
               weight_decay, moving_average_decay, optimize_method, pretrained_model, 
               learning_rate_init, learning_rate_decay_epochs, gpu_memory_fraction):
    
    time_string = datetime.strftime(datetime.now(), '_%Y%m%d%H%M%S')
    subdir = model_name + time_string
    model_dir = os.path.join(model_root, subdir)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    
    train_set = get_dataset(data_dir)
    image_path_list, label_list = get_image_path_and_label_list(train_set)
    category_num = len(train_set)
    total_image_num = len(image_path_list)
    
    lfw_pairs = read_pairs(pairs_txt)
    lfw_path_list, issame_list = get_image_path_and_issame_list(lfw_dir, lfw_pairs)
    lfw_image_num = len(lfw_path_list)
    lfw_path_array = np.expand_dims(np.array(lfw_path_list),1)
    lfw_label_array = np.expand_dims(np.arange(0,lfw_image_num),1)
    lfw_batch_num = lfw_image_num // batch_size
    
    with tf.Graph().as_default():
        print('Building graph......')
        image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')
        labels_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='labels')
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        global_step = tf.Variable(0, trainable=False)
        
        index_queue = tf.train.range_input_producer(total_image_num, num_epochs=None, 
                                                    shuffle=True, seed=None, capacity=32)
        index_dequeue_op = index_queue.dequeue_many(batch_size*epoch_size, 'index_dequeue')
        input_queue = tf.FIFOQueue(capacity=2000000, 
                                   dtypes=[tf.string, tf.int32], 
                                   shapes=[(1,), (1,)], 
                                   shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder], 
                                              name='enqueue_op')
        
        image_batch, label_batch = get_batch(input_queue, batch_size_placeholder, 
                                             image_height, image_width)
        image_batch = tf.identity(image_batch, 'image_batch')
        label_batch = tf.identity(label_batch, 'label_batch')
        image_batch = tf.identity(image_batch, 'input')
        
        print('Number of classes in training set: %d' % category_num)
        print('Number of samples in training set: %d' % total_image_num)
        print('Building training graph')
        
        mobilenet = mobilenet_v1.MobileNetV1(image_batch, 
                                             num_classes=embedding_size, 
                                             is_training=phase_train_placeholder)
        prelogits = mobilenet.logits
        prelogits = tf.identity(prelogits, 'prelogits')
        
        logits = slim.fully_connected(prelogits, 
                                      category_num, 
                                      activation_fn=None, 
                                      weights_initializer=slim.initializers.xavier_initializer(), 
                                      weights_regularizer=slim.l2_regularizer(weight_decay), 
                                      scope='Logits', reuse=False)

        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, 
                                                   global_step, 
                                                   learning_rate_decay_epochs*epoch_size, 
                                                   1.0, staircase=True)
        loss = get_loss(label_batch, logits)
        variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        with tf.control_dependencies([variables_averages_op]):
            train_op = train_step(loss, learning_rate, global_step, optimize_method, tf.global_variables())
        accuracy = get_accuracy(label_batch, logits)

        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)
        
        with sess.as_default():
            if pretrained_model:
                print('Restoring pretrained model: %s' % pretrained_model)
                saver.restore(sess, tf.train.latest_checkpoint(pretrained_model))
            
            print('Running train......')
            for i_epoch in range(1, max_epochs + 1):
                step = sess.run(global_step, feed_dict=None)
                control = train_epoch(sess, i_epoch, epoch_size, batch_size, image_path_list, 
                                      label_list, index_dequeue_op, enqueue_op, loss, train_op, 
                                      accuracy, learning_rate_init, learning_rate_decay_epochs, 
                                      image_paths_placeholder, labels_placeholder, batch_size_placeholder, 
                                      learning_rate_placeholder, phase_train_placeholder)
                if not control:
                    print('Trainning False!')
                    break
                save_variables_and_metagraph(sess, saver, step, model_dir, model_name)
                if (i_epoch > 100) and (i_epoch % 10 == 0):
                    validate_epoch(sess, enqueue_op, prelogits, label_batch, lfw_path_array, 
                               batch_size, lfw_label_array, lfw_image_num, lfw_batch_num, 
                               issame_list, image_paths_placeholder, labels_placeholder, 
                               batch_size_placeholder, phase_train_placeholder)

def train_epoch(sess, epoch, epoch_size, batch_size, image_path_list, label_list, index_dequeue_op, 
                enqueue_op, loss, train_op, accuracy, learning_rate_init, learning_rate_decay_epochs, 
                image_paths_placeholder, labels_placeholder, batch_size_placeholder, 
                learning_rate_placeholder, phase_train_placeholder):
    learning_rate = get_learning_rate_against_epoch(learning_rate_init, learning_rate_decay_epochs, epoch)
    
    index = sess.run(index_dequeue_op)
    image_path_array = np.array(image_path_list)[index]
    label_array = np.array(label_list)[index]
    image_path_array = np.expand_dims(np.array(image_path_array), 1)
    label_array = np.expand_dims(np.array(label_array), 1)
    sess.run(enqueue_op, {image_paths_placeholder: image_path_array, 
                          labels_placeholder: label_array})
    
    i_batch = 0
    feed_dict = {
                    batch_size_placeholder: batch_size, 
                    learning_rate_placeholder: learning_rate, 
                    phase_train_placeholder: True
                }
    tensor_list = [loss, train_op, accuracy]
    while i_batch < epoch_size:
        loss_, _, accuracy_ = sess.run(tensor_list, feed_dict=feed_dict)
        print('Epoch: [%d][%d/%d]\tLoss %2.3f\tAccuracy %2.3f' % 
              (epoch, i_batch+1, epoch_size, loss_, accuracy_))
        i_batch += 1
    
    return True

def validate_epoch(sess, enqueue_op, prelogits, label_batch, lfw_path_array, 
                   batch_size, lfw_label_array, lfw_image_num, lfw_batch_num, 
                   issame_list, image_paths_placeholder, labels_placeholder, 
                   batch_size_placeholder, phase_train_placeholder):
    print("Validating on lfw......")
    sess.run(enqueue_op, {image_paths_placeholder: lfw_path_array, 
                          labels_placeholder: lfw_label_array})
    embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
    embedding_size = int(embeddings.get_shape()[1])
    emb_array = np.zeros((lfw_image_num, embedding_size))
    for i_batch in range(lfw_batch_num):
        feed_dict = {phase_train_placeholder: False, batch_size_placeholder: batch_size}
        emb, lab = sess.run([embeddings, label_batch], feed_dict=feed_dict)
        emb_array[lab, :] = emb
        if i_batch % 10 == 9:
            print('.', end='')
            sys.stdout.flush()
    print('')
    accuracy_, _ = get_accuracy_and_threshold(emb_array, issame_list)
    print("Accuracy: %2.3f" % np.mean(accuracy_))

#============================================================

def get_file_path_list(file_dir):
    file_path_list = []
    if os.path.isdir(file_dir):
        files = os.listdir(file_dir)
        for file in files:
            file_path = os.path.join(file_dir, file)
            if not os.path.isdir(file_path):
                file_path_list.append(file_path)
    return file_path_list

class IMAGE_CLASS():
    def __init__(self, name, image_path_list):
        self.name = name
        self.image_path_list = image_path_list

def get_dataset(data_dir):
    name_list = []
    for name in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, name)):
            name_list.append(name)
    dataset = []
    for name in name_list:
        image_path_list = get_file_path_list(os.path.join(data_dir, name))
        dataset.append(IMAGE_CLASS(name, image_path_list))
    return dataset

def get_image_path_and_label_list(dataset):
    image_path_list = []
    label_list = []
    for i in range(len(dataset)):
        image_path_list += dataset[i].image_path_list
        label_list += [i] * len(dataset[i].image_path_list)
    return image_path_list, label_list

def read_pairs(pairs_file):
    pairs = []
    with open(pairs_file, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)

def get_image_path_and_issame_list(data_dir, pairs):
    len_skipped_pair = 0
    image_path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = add_suffix(os.path.join(data_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_suffix(os.path.join(data_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
            issame = True
        elif len(pair) == 4:
            path0 = add_suffix(os.path.join(data_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_suffix(os.path.join(data_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):
            image_path_list += (path0,path1) # just only a list, double length
            issame_list.append(issame)
        else:
            len_skipped_pair += 1
    if len_skipped_pair>0:
        print('Skipped %d image pairs' % len_skipped_pair)
    
    return image_path_list, issame_list

def add_suffix(path):
    if os.path.exists(path+'.jpg'):
        return path+'.jpg'
    elif os.path.exists(path+'.png'):
        return path+'.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)

def get_batch(input_queue, batch_size_placeholder, image_height, image_width):
    image_size = (image_height, image_width)
    images_and_labels_list = []
    images = []
    filenames, labels = input_queue.dequeue()
    for filename in tf.unstack(filenames):
        file_contents = tf.read_file(filename)
        image = tf.image.decode_image(file_contents, channels=3)
        image = tf.image.resize_image_with_crop_or_pad(image, image_height, image_width)
        image = tf.image.per_image_standardization(image)
        image.set_shape(image_size + (3,))
        images.append(image)
    images_and_labels_list.append([images, labels])

    image_batch, label_batch = tf.train.batch_join(images_and_labels_list, 
                                                   batch_size=batch_size_placeholder, 
                                                   shapes=[image_size + (3,), ()], 
                                                   enqueue_many=True, 
                                                   capacity=1600,
                                                   allow_smaller_final_batch=True)
    return image_batch, label_batch

def get_loss(labels, logits):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(cross_entropy, name='loss')
    return loss

def train_step(loss, learning_rate, global_step, opt_method, update_gradient_vars):
    if opt_method == 'ADAGRAD':
        optimizer = tf.train.AdagradOptimizer(learning_rate)
    elif opt_method == 'ADADELTA':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
    elif opt_method == 'ADAM':
        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
    elif opt_method == 'RMSPROP':
        optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
    elif opt_method == 'MOM':
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
    else:
        raise ValueError('Invalid optimization algorithm')
    grads = optimizer.compute_gradients(loss, update_gradient_vars)
    apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
    with tf.control_dependencies([apply_gradient_op]):
        train_step = optimizer.minimize(loss, global_step=global_step)
    return train_step

def get_accuracy(labels, logits):
    correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(labels, tf.int64)), tf.float32)
    prediction = tf.reduce_mean(correct_prediction)
    return prediction

def get_learning_rate_against_epoch(learning_rate_init, learning_rate_decay_epochs, epoch):
    return learning_rate_init * (0.1 ** (epoch // learning_rate_decay_epochs))

def prewhiten_image(image):
    mean = np.mean(image)
    std = np.std(image)
    std_adj = np.maximum(std, 1.0/np.sqrt(image.size))
    whiten = np.multiply(np.subtract(image, mean), 1/std_adj)
    return whiten

def get_accuracy_and_threshold(embeddings, actual_issame):
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    dist_sq = get_dist_square(embeddings1, embeddings2)
    threshold = get_threshold(thresholds, dist_sq, actual_issame)
    accuracy_ = get_accuracy_with_dist(dist_sq, threshold, actual_issame)
    return accuracy_, threshold

def get_dist_square(emb1, emb2):
    diff = np.subtract(emb1, emb2)
    dist = np.sum(np.square(diff),1)
    return dist

def get_accuracy_with_dist(dist, threshold, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_same = np.sum(np.logical_and(predict_issame, actual_issame))
    false_same = np.sum(np.logical_and(np.logical_not(predict_issame), 
                                       np.logical_not(actual_issame)))
    accuracy = float(true_same + false_same) / dist.size
    return accuracy

def get_threshold(thresholds, dist, actual_issame):
    accuracy_array = np.zeros(len(thresholds))
    for index, threshold in enumerate(thresholds):
        accuracy_array[index] = get_accuracy_with_dist(dist, threshold, actual_issame)
    best_index = np.argmax(accuracy_array)
    best_threshold = thresholds[best_index]
    return best_threshold

def save_variables_and_metagraph(sess, saver, step, model_dir, model_name):
    meta_file = os.path.join(model_dir, 'model_%s.meta' % model_name)
    ckpt_file = os.path.join(model_dir, 'model_%s.ckpt' % model_name)
    if not os.path.exists(meta_file):
        print('Saving metagraph......')
        saver.export_meta_graph(meta_file)
    print('Saving variables......')
    saver.save(sess, ckpt_file, global_step=step, write_meta_graph=False)
    print('Done.')

def load_model(model_dir, input_map=None):
    print('Model directory: %s' % model_dir)
    meta_file, ckpt_file = get_model_filenames(model_dir)
    print('Metagraph file: %s' % meta_file)
    print('Checkpoint file: %s' % ckpt_file)
    saver = tf.train.import_meta_graph(os.path.join(model_dir, meta_file), input_map=input_map)
    saver.restore(tf.get_default_session(), os.path.join(model_dir, ckpt_file))
    
def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    ckpt_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in ckpt_files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file

def my_net(images, embedding_size, image_height, image_width):
    with tf.name_scope('reshape'):
        x_image = tf.reshape(images, [-1, image_height, image_width, 3])

    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([7, 7, 3, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('conv3'):
        W_conv3 = weight_variable([5, 5, 64, 128])
        b_conv3 = bias_variable([128])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    with tf.name_scope('pool3'):
        h_pool3 = max_pool_2x2(h_conv3)

    with tf.name_scope('conv4'):
        W_conv4 = weight_variable([5, 5, 128, 256])
        b_conv4 = bias_variable([256])
        h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
    with tf.name_scope('pool4'):
        h_pool4 = max_pool_2x2(h_conv4)

    with tf.name_scope('conv5'):
        W_conv5 = weight_variable([5, 5, 256, 512])
        b_conv5 = bias_variable([512])
        h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
    with tf.name_scope('pool5'):
        h_pool5 = max_pool_2x2(h_conv5)

    with tf.name_scope('fc1'):
        n_height = int(h_pool5.get_shape()[1])
        n_width = int(h_pool5.get_shape()[2])
        W_fc1 = weight_variable([n_height * n_width * 512, 1024])
        b_fc1 = bias_variable([1024])
        h_pool5_flat = tf.reshape(h_pool5, [-1, n_height * n_width * 512])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)

    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, embedding_size])
        b_fc2 = bias_variable([embedding_size])
        y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
    return y_conv

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def run_train():
#     data_dir = '../data/VGGFace2_mtcnn_224'
    data_dir = '../data/lfw_mtcnn_224'
    lfw_dir = '../data/lfw_mtcnn_224'
    pairs_txt = '../data/pairs.txt'
    model_root = '../models'
#     model_name = "my_net"
    model_name = "mobilenet_v1"
    learning_rate_init = 0.05
    learning_rate_decay_epochs = 30
    optimize_method = 'ADAM'
    image_height = 224
    image_width = 224
    batch_size = 100
    epoch_size = 1000
    max_epochs = 90
    embedding_size = 1000
    weight_decay = 0.0005
    moving_average_decay = 0.9999
    pretrained_model = None
    gpu_memory_fraction = 0.7
    train_stem(data_dir, lfw_dir, pairs_txt, model_root, model_name, batch_size, 
               epoch_size, max_epochs, image_height, image_width, embedding_size, 
               weight_decay, moving_average_decay, optimize_method, pretrained_model, 
               learning_rate_init, learning_rate_decay_epochs, gpu_memory_fraction)

if __name__ == '__main__':
    run_train()
    print('____End____')



