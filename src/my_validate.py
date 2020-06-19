import os
import shutil
import sys
import cv2
import time
import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

from src import my_train

# ================================================================================================

def my_classifier(old_dir, new_dir, model_dir, image_width, image_height):
    old_file_list = os.listdir(old_dir)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            my_train.load_model(model_dir)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            logits = tf.get_default_graph().get_tensor_by_name("InceptionResnetV1/Bottleneck/BatchNorm/batchnorm/add_1:0")
            classifier = tf.argmax(logits, 1)[0]
            
            for file in old_file_list:
                if not (file.endswith('.png') | file.endswith('.bmp')):
                    continue
                image_path = os.path.join(old_dir, file)
                images = get_images(image_path, image_height, image_width)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                num = sess.run(classifier, feed_dict=feed_dict)
                dir_name = 'class_' + '%04d' % int(num)
                new_dir_2 = os.path.join(new_dir, dir_name)
                if not os.path.isdir(new_dir_2):
                    os.makedirs(new_dir_2)
                new_image_path = os.path.join(new_dir_2, file)
                print(new_dir_2)
                shutil.move(image_path, new_image_path)

def get_model_speed(data_dir, model_dir, image_height, image_width):
    data_set = my_train.get_dataset(data_dir)
    image_path_list, _ = my_train.get_image_path_and_label_list(data_set)
    image_num = len(image_path_list)
    total_time = 0.0
    with tf.Graph().as_default():
        with tf.Session() as sess:
            my_train.load_model(model_dir)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("InceptionResnetV1/Bottleneck/BatchNorm/batchnorm/add_1:0")
            for image_path in image_path_list:
                images = get_images(image_path, image_height, image_width)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                time_1 = time.time()
                sess.run(embeddings, feed_dict=feed_dict)
                time_2 = time.time()
                total_time += (time_2 - time_1)
            print("speed: %f" % (image_num / total_time))

def get_images(image_path, image_height, image_width):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
#     cv2.imshow('', image)
#     cv2.waitKey(0)
    image = cv2.resize(image, (image_width, image_height))
#     cv2.imshow('', image)
#     cv2.waitKey(0)
    prewhitened = my_train.prewhiten_image(image)
    images = np.stack([prewhitened])
#     print(np.shape(images))
    return images

def get_trainable_variables(model_dir):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            my_train.load_model(model_dir)
            variable_names = [v.name for v in tf.trainable_variables()]
            with open('../data/trainable_variables.txt', 'w') as file:
                values = sess.run(variable_names)
                i=0
                for k, v in zip(variable_names, values):
                    i+=1
                    if k.find('encode') != -1:
                        file.write("Variable: ", k)
                        file.write('\n')
                        file.write("Shape: ", v.shape)
                        file.write('\n')
                        file.write(v)
                        file.write('\n')
                graph = tf.get_default_graph()
                all_ops = graph.get_operations()
                for el in all_ops:
                    file.write(el.name)
                    file.write('\n')

# ================================================================================================

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
            
            prelogits = tf.get_default_graph().get_tensor_by_name("prelogits:0")
            embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
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

def run_validate():
    lfw_dir = '../data/lfw_mtcnn_224'
    pairs_txt = '../data/pairs.txt'
    model_dir = ''
    image_height = 224
    image_width = 224
    batch_size = 100
    gpu_memory_fraction = 0.95
    validate_main(lfw_dir, pairs_txt, model_dir, image_height, image_width, 
                  batch_size, gpu_memory_fraction)

# ================================================================================================

def run_test():
    data_dir = '../data/lfw_mtcnn_224'
    model_dir = '../models/Inception_resnet_v1_20170512110547'
    get_model_speed(data_dir, model_dir, 160, 160)

if __name__ == '__main__':
#     run_validate()
    run_test()
    print('____End____')

























