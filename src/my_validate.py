import os
import sys
import numpy as np
import tensorflow as tf

from src import my_train
from src import my_utils

#============================================================

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
            variables = tf.trainable_variables()
            with open(file_name, 'w') as file:
                for v in variables:
                    file.write(v.name)
                    file.write('\n')

def get_variables(model_dir, model_name):
    file_name = "../data/variables/" + model_name + "_variables.txt"
    with tf.Graph().as_default():
        with tf.Session() as sess:
            my_train.load_model(model_dir)
            variables = tf.contrib.graph_editor.get_tensors(tf.get_default_graph())
            with open(file_name, 'w') as file:
                for v in variables:
                    file.write(v.name)
                    file.write('\n')

def get_weights(model_dir, model_name):
    folder = os.path.join('../data/weights', model_name)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            my_train.load_model(model_dir)
            names = [v.name for v in tf.trainable_variables()]
            weights = sess.run(names)
            for name_0, weight in zip(names, weights):
                name = name_0.replace(":0", "")
                name = name.replace("/", "_")
                file_name = name + '.txt'
                file_path = os.path.join(folder, file_name)
                shape = weight.shape
                shape_len = len(shape)
                with open(file_path, 'w') as file:
                    if shape_len == 4:
                        if 'dw_conv' in name_0:
                            [height, width, channel, number] = shape
                            for n in range(number):
                                for c in range(channel):
                                    for h in range(height):
                                        for w in range(width):
                                            value = weight[h][w][c][n]
                                            file.write(str(value))
                                            file.write('\n')
                        else:
                            [height, width, channel, number] = shape
                            for n in range(number):
                                for c in range(channel):
                                    for h in range(height):
                                        for w in range(width):
                                            value = weight[h][w][c][n]
                                            file.write(str(value))
                                            file.write('\n')
                    elif shape_len == 2:
                        [height, width] = shape
                        for h in range(height):
                            for w in range(width):
                                value = weight[h][w]
                                file.write(str(value))
                                file.write('\n')
                    elif shape_len == 1:
                        [number] = shape
                        for n in range(number):
                            value = weight[n]
                            file.write(str(value))
                            file.write('\n')

def get_feature(model_dir, image_path, feature_0):
    name = feature_0.replace(":0", "")
    name = name.replace("/", "_")
    file_name = 'Amy_Smart_' + name + '.txt'
    file_path = os.path.join('../data/test', file_name)
    image_path_list = [image_path]
    image_path_array = np.array(image_path_list)
    images = my_utils.get_images(image_path_array, 224, 224)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            my_train.load_model(model_dir)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            variable = tf.get_default_graph().get_tensor_by_name(feature_0)
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            feature = sess.run(variable, feed_dict=feed_dict)
            print(feature.shape)
            shape_len = len(feature.shape)
            with open(file_path, 'w') as file:
                if shape_len == 4:
                    [number, height, width, channel] = feature.shape
                    for n in range(number):
                        for c in range(channel):
                            for h in range(height):
                                for w in range(width):
                                    value = feature[n][h][w][c]
                                    file.write(str(value))
                                    file.write('\n')
                elif shape_len == 2:
                    [height, width] = feature.shape
                    for h in range(height):
                        for w in range(width):
                            value = feature[h][w]
                            file.write(str(value))
                            file.write('\n')
                elif shape_len == 1:
                    [number] = feature.shape
                    for n in range(number):
                        value = feature[n]
                        file.write(str(value))
                        file.write('\n')

def get_weight(model_dir, weight_0):
    name = weight_0.replace(":0", "")
    name = name.replace("/", "_")
    file_name = name + '.txt'
    file_path = os.path.join('../data/test', file_name)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            my_train.load_model(model_dir)
            weight = sess.run(weight_0)
            shape = weight.shape
            print(shape)
            shape_len = len(shape)
            with open(file_path, 'w') as file:
                if shape_len == 4:
                    [height, width, channel, number] = shape
                    for n in range(number):
                        for c in range(channel):
                            for h in range(height):
                                for w in range(width):
                                    value = weight[h][w][c][n]
                                    file.write(str(value))
                                    file.write('\n')
                elif shape_len == 2:
                    [height, width] = shape
                    for h in range(height):
                        for w in range(width):
                            value = weight[h][w]
                            file.write(str(value))
                            file.write('\n')
                elif shape_len == 1:
                    [number] = shape
                    for n in range(number):
                        value = weight[n]
                        file.write(str(value))
                        file.write('\n')

#============================================================

#============================================================

def run_validate():
    lfw_dir = '../data/my_data_160'
    pairs_txt = '../data/my_pairs.txt'
    model_dir = '../data/models/Inception_resnet_v1_20170512110547'
    image_height = 160
    image_width = 160
    batch_size = 100
    gpu_memory_fraction = 0.95
    validate_main(lfw_dir, pairs_txt, model_dir, image_height, 
                  image_width, batch_size, gpu_memory_fraction)

if __name__ == '__main__':
    model_dir = '../models/MobileNetV1_20200831164010'
    model_name = "MobileNetV1"
    
#     get_weights(model_dir, model_name)
#     get_trainable_variables(model_dir, model_name)
#     get_variables(model_dir, model_name)
    
#     'MobileNet/conv_1/conv/Conv2D:0'
#     'MobileNet/conv_1/relu/Relu:0'
#     'MobileNet/ds_conv_2/dw_conv/depthwise:0'
#     'MobileNet/ds_conv_2/dw_relu/Relu:0'
#     'prelogits:0'

#     image_path = '../data/test/Amy_Smart_224.bmp'
#     variable_0 = 'MobileNet/ds_conv_2/dw_conv/depthwise:0'
#     get_feature(model_dir, image_path, variable_0)

#     weight_0 = 'MobileNet/ds_conv_2/dw_conv/Filter:0'

#     get_weight(model_dir, weight_0)
    

    print('____End____')

























