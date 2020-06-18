import os
import numpy as np

# ================================================================================================

def make_pairs(data_dir, same_num, diff_num):
    set_same = set([])
    set_diff = set([])
    
    name_list = os.listdir(data_dir)
    
    while len(set_diff) < diff_num:
        np.random.shuffle(name_list)
        name1 = name_list[0]
        name2 = name_list[1]
        file_dir1 = os.path.join(data_dir, name1)
        file_dir2 = os.path.join(data_dir, name2)
        file_list1 = os.listdir(file_dir1)
        file_list2 = os.listdir(file_dir2)
        file_num1 = len(file_list1)
        file_num2 = len(file_list2)
        if file_num1 < 1 or file_num2 < 1:
            continue
        index_list1 = list(range(1, file_num1 + 1))
        index_list2 = list(range(1, file_num2 + 1))
        np.random.shuffle(index_list1)
        np.random.shuffle(index_list2)
        pick1 = index_list1[0]
        pick2 = index_list2[0]
        name_pair = (name1, pick1, name2, pick2)
        set_diff.add(name_pair)
    
    while len(set_same) < same_num:
        np.random.shuffle(name_list)
        name = name_list[0]
        file_dir = os.path.join(data_dir, name)
        file_list = os.listdir(file_dir)
        file_num = len(file_list)
        if file_num < 2:
            continue
        index_list = list(range(1, file_num + 1))
        np.random.shuffle(index_list)
        pick1 = index_list[0]
        pick2 = index_list[1]
        if pick1 < pick2:
            name_pair = (name, pick1, pick2)
        else:
            name_pair = (name, pick2, pick1)
        set_same.add(name_pair)
    list_same = list(set_same)
    list_same.sort()
    list_diff = list(set_diff)
    list_diff.sort()
    return list_same, list_diff

def write_pairs(pairs_txt, list_same, list_diff):
    with open(pairs_txt, mode='w') as file:
        for pair_same in list_same:
            pair_str = ''
            for element in pair_same:
                pair_str += str(element)
                pair_str += '\t'
            pair_str = pair_str.strip()
            file.write(pair_str)
            file.write('\n')
        
        for pair_diff in list_diff:
            pair_str = ''
            for element in pair_diff:
                pair_str += str(element)
                pair_str += '\t'
            pair_str = pair_str.strip()
            file.write(pair_str)
            file.write('\n')

def make_dirs(root, name, num):
    for i in range(1, num + 1):
        dir_name = name + '%04d' % int(i)
        dir_path = os.path.join(root, dir_name)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

def rename_files(file_dir, name):
    file_list = os.listdir(file_dir)
    file_num = len(file_list)
    for index in range(1, file_num + 1):
        orig_file_path = os.path.join(file_dir, file_list[index - 1])
        dest_file_path = os.path.join(file_dir, name + '_' + '%06d' % int(index) + '.png')
        os.rename(orig_file_path, dest_file_path)

def remove_files(file_dir):
    if os.path.isdir(file_dir):
        files = os.listdir(file_dir)
        for file in files:
            file_path = os.path.join(file_dir, file)
            if os.path.isfile(file):
                os.remove(file_path)

# ================================================================================================

def test_dir():
    root = './'
    name = 'good_'
    make_dirs(root, name, 10)


# ================================================================================================

if __name__ == '__main__':
    test_dir()
#     test_str()
#     test_shuffle()

#     pairs_txt = '../data/my_pairs.txt'
#     data_dir = '../data/lfw_mtcnn_224'
#     same_num = 3
#     diff_num = 3
#     list_same, list_diff = make_pairs(data_dir, same_num, diff_num)
#     write_pairs(pairs_txt, list_same, list_diff)
    
    print('____End____')

# ================================================================================================
