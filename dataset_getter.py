# -*- coding: utf-8 -*
#from sklearn.model_selection import train_test_split, learning_curve, KFold
import pickle as pkl

import easygui
import numpy as np
from sklearn.model_selection import train_test_split

def get_dataset_no_partitioning():
    # без разбивки на трейн-тест
    filename_pkl = easygui.fileopenbox("выберите файл с датасетом")

    print("загружаем 2д-датасет из файла " + filename_pkl)
    infile = open(filename_pkl, 'rb')
    new_dict = pkl.load(infile)
    infile.close()

    #assert len(new_dict['x']) == len(new_dict['y'])
    if 'ann' in new_dict:
        x, y = np.array(new_dict['x']), np.array(new_dict['ann'])
        print(len(new_dict['x']), " записей ")
        return x, y
    else:
        x= np.array(new_dict['x'])
        print(len(new_dict['x']), " записей ")
        return x



def prepare_data(seg_len=None):
    """
    получаем тензоры, в котром каналы экг есть глубина, а не ширина
    :param seg_len: длина начального сегмента экг, такты
    :return: x_train, x_test, y_train, y_test
    """

    x, y = get_dataset_no_partitioning()
    # решейпим их, чтоб модель могла их сожрать как 2д а не 1д
    x = np.expand_dims(x, axis=2)  # reshape (n1, n2) -> (n1, n2, 1)

    x = np.swapaxes(x, 1, 3)
    if seg_len is not None:
        x = x[:, 0:seg_len, :, : ]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    print("y_train  имеет форму " + str(y_train.shape))
    print("x_train  имеет форму " + str(x_train.shape))
    print("x_test   имеет форму " + str(x_test.shape))
    print("y_test   имеет форму " + str(y_test.shape))

    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = prepare_data(seg_len=None)
