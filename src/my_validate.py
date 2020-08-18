import os
import shutil
import sys
# import cv2
import time
import numpy as np
import tensorflow as tf
# from tensorflow.python import pywrap_tensorflow

from src import my_train
from src import my_utils

#============================================================

def get_model_speed(data_dir, model_dir, image_height, image_width, batch_size):
    data_set = my_train.get_dataset(data_dir)
    image_path_list, _ = my_train.get_image_path_and_label_list(data_set)
    np.random.shuffle(image_path_list)
    image_path_array = np.array(image_path_list)
    total_image_num = len(image_path_list)
    batch_num = total_image_num // batch_size
    test_image_num = batch_size * batch_num
    
    with tf.Graph().as_default():
        with tf.Session() as sess:
            my_train.load_model(model_dir)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("InceptionResnetV1/Bottleneck/BatchNorm/batchnorm/add_1:0")
            
            time_1 = time.time()
            for i_batch in range(batch_num):
                index_array = np.array(range(batch_size * i_batch, batch_size * (i_batch + 1)))
                image_path_batch = image_path_array[index_array]
                images = my_utils.get_images(image_path_batch, image_height, image_width)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                sess.run(embeddings, feed_dict=feed_dict)
            time_2 = time.time()
            total_time = (time_2 - time_1)
            print("Total image number: %d" % total_image_num)
            print("Test image number: %d" % test_image_num)
            print("Total time: %f" % total_time)

#============================================================

def my_classifier(old_dir, new_dir, model_dir, image_height, image_width):
    file_list = os.listdir(old_dir)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            my_train.load_model(model_dir)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            logits = tf.get_default_graph().get_tensor_by_name("InceptionResnetV1/Bottleneck/BatchNorm/batchnorm/add_1:0")
            classifier = tf.argmax(logits, 1)[0]
            
            for file in file_list:
                if not (file.endswith('.png') | file.endswith('.bmp')):
                    continue
                image_path = os.path.join(old_dir, file)
                images = my_utils.get_images([image_path], image_height, image_width)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                num = sess.run(classifier, feed_dict=feed_dict)
                dir_name = 'class_' + '%04d' % int(num)
                new_dir_2 = os.path.join(new_dir, dir_name)
                if not os.path.isdir(new_dir_2):
                    os.makedirs(new_dir_2)
                new_image_path = os.path.join(new_dir_2, file)
                print(new_dir_2)
                shutil.move(image_path, new_image_path)

def validate_main(lfw_dir, pairs_txt, model_dir, image_height, image_width, batch_size, 
                  gpu_memory_fraction):
    
    lfw_pairs = my_train.read_pairs(pairs_txt)
    lfw_path_list, issame_list = my_train.get_image_path_and_issame_list(lfw_dir, lfw_pairs)
    lfw_image_num = len(lfw_path_list)
    lfw_path_array = np.expand_dims(np.array(lfw_path_list),1)
    lfw_label_array = np.expand_dims(np.arange(0,lfw_image_num),1)
    lfw_batch_num = lfw_image_num // batch_size
    
    with tf.Graph().as_default():
        image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')
        labels_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='labels')
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        
        eval_input_queue = tf.FIFOQueue(capacity=2000000, 
                                        dtypes=[tf.string, tf.int32], 
                                        shapes=[(1,), (1,)], 
                                        shared_name=None, name=None)
        eval_enqueue_op = eval_input_queue.enqueue_many([image_paths_placeholder, labels_placeholder])
        image_batch, label_batch = my_train.get_batch(eval_input_queue, batch_size_placeholder, 
                                                      image_height, image_width)
        input_map = {'image_batch': image_batch, 
                     'label_batch': label_batch, 
                     'phase_train': phase_train_placeholder, 
                     'batch_size': batch_size_placeholder}
        
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)
        with sess.as_default():
            my_train.load_model(model_dir, input_map=input_map)
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
#             prelogits = tf.get_default_graph().get_tensor_by_name("prelogits:0")
#             embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
            embedding_size = int(embeddings.get_shape()[1])
            
            sess.run(eval_enqueue_op, {image_paths_placeholder: lfw_path_array, 
                                       labels_placeholder: lfw_label_array})
            
            emb_array = np.zeros((lfw_image_num, embedding_size))
            for i_batch in range(lfw_batch_num):
                feed_dict = {phase_train_placeholder: False, batch_size_placeholder: batch_size}
                emb, lab = sess.run([embeddings, label_batch], feed_dict=feed_dict)
                emb_array[lab, :] = emb
                if i_batch % 10 == 9:
                    print('.', end='')
                    sys.stdout.flush()
            print('')
            accuracy_, threshold_ = my_train.get_accuracy_and_threshold(emb_array, issame_list)
            print("Accuracy: %2.3f\nThreshold: %2.3f" % (np.mean(accuracy_), threshold_))

#============================================================

def get_trainable_variables(model_dir, model_name):
    file_name = "../data/variables/" + model_name + "_trainable_variables.txt"
    
    with tf.Graph().as_default():
        with tf.Session() as sess:
            my_train.load_model(model_dir)
            variable_names = [v.name for v in tf.trainable_variables()]
            values = sess.run(variable_names)
            with open(file_name, 'w') as file:
                for k, v in zip(variable_names, values):
                    file.write(k)
                    file.write('\n')


def get_weights_2(model_dir):
#     line = "InceptionResnetV1/Conv2d_1a_3x3/BatchNorm/moving_mean/read"
    line = "MobileNet/AssignMovingAvg/MobileNet/conv_1/bn/moving_mean/read"
    
    line_0 = line + ":0"
    with tf.Graph().as_default():
        with tf.Session() as sess:
            my_train.load_model(model_dir)
#             batch_size_placeholder = tf.get_default_graph().get_tensor_by_name("batch_size:0")
#             phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
#             feed_dict = {batch_size_placeholder: 1, phase_train_placeholder: False}
            weights = tf.get_default_graph().get_tensor_by_name(line_0)
            print(sess.run(weights))
    None

def get_weights(model_dir):
    file_lines = []
    with open('../data/mobilenet_v1_weights/mobilenet_v1_variables.txt','r') as f:
        for i in f.readlines():
            file_lines.append(i.strip())
    print(file_lines)
    
    with tf.Graph().as_default():
        with tf.Session() as sess:
            my_train.load_model(model_dir)
            for line in file_lines:
                line_0 = line + ":0"
                print(line_0)
                line_name = line.replace("/", "_")
                line_name = line_name.replace("_read", "")
                file_name = '../data/mobilenet_v1_weights/' + line_name + '.txt'
                weights = tf.get_default_graph().get_tensor_by_name(line_0)
                shape = weights.get_shape().as_list()
                shape_len = len(shape)
                with open(file_name, 'w') as file:
                    if shape_len == 4:
                        [height, width, channel, output] = shape
                        for h in range(height):
                            for w in range(width):
                                for c in range(channel):
                                    for n in range(output):
                                        weight = weights[h][w][c][n].eval()
                                        file.write(str(weight))
                                        file.write('\n')
                    elif shape_len == 2:
                        [height, width] = shape
                        for h in range(height):
                            for w in range(width):
                                weight = weights[h][w].eval()
                                file.write(str(weight))
                                file.write('\n')
                    elif shape_len == 1:
                        [output] = shape
                        print(output)
                        print(sess.run(weights))
                        for n in range(output):
                            weight = sess.run(weights[n])
                            file.write(str(weight))
                            file.write('\n')
    None

#============================================================

def run_validate():
    lfw_dir = '../data/my_data_160'
    pairs_txt = '../data/my_pairs.txt'
    model_dir = '../models/Inception_resnet_v1_20170512110547'
    image_height = 160
    image_width = 160
    batch_size = 100
    gpu_memory_fraction = 0.95
    validate_main(lfw_dir, pairs_txt, model_dir, image_height, image_width, 
                  batch_size, gpu_memory_fraction)

if __name__ == '__main__':
#     model_dir = '../models/Inception_resnet_v1_20170512110547'
    model_dir = '../models/mobilenet_v1_20200707123403'
    model_name = "mobilenet_v1"
    
#     get_weights_2(model_dir)
    get_trainable_variables(model_dir, model_name)
#     test_code()
#     test_get_images()
#     run_validate()
#     run_test()
    print('____End____')

























