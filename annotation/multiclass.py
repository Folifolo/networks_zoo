# -*- coding: utf-8 -*
import matplotlib.pyplot as plt
import numpy as np
import easygui
import pickle as pkl
from keras.models import load_model
import os
from annotation.ann_generator import (
    get_mulimask_generator
)
from utils import open_pickle
from sklearn.model_selection import train_test_split
from annotation.model import unet
from annotation.one_lead_one_mask import unet_simple
from utils import save_history
from annotation.dice_koef import (
    dice_coef, dice_coef_loss
)
dataset_path = "./DSET_argentina.pkl"
segment_len=512


def split_dict_annotations(dict_dataset):
    # бьем датасет-мапу на трейновую и тестовую
    x = dict_dataset['x']
    ann = dict_dataset['ann']
    X_train, X_test, ann_train, ann_test = train_test_split(x, ann, test_size=0.33, random_state=42)
    return {'x':X_train, 'ann':ann_train},{'x':X_test, 'ann':ann_test}

def get_generators(train_batch, test_batch):
    """
    слепим два генератора- тестовый и трейновый
    :param train_batch: размер батча для трейнового генератора
    :param test_batch: размер батча для тестового генератора
    :return:
    """
    dataset_in = open_pickle(dataset_path)
    train_dset, test_dset = split_dict_annotations(dataset_in)
    my_generator_train = get_mulimask_generator(segment_len, batch_size=train_batch, dataset_in=train_dset)
    my_generator_test = get_mulimask_generator(segment_len, batch_size=test_batch, dataset_in=test_dset)
    return my_generator_train, my_generator_test

def get_model():
    #return unet(seg_len=segment_len)
    return unet_simple33(seg_len=segment_len)


def train(name):
    model = get_model()
    generator_train, generator_test = get_generators(train_batch=15, test_batch=50)
    history = model.fit_generator(generator=generator_train,
                                  steps_per_epoch=40,
                                  epochs=10,
                                  validation_data=generator_test,
                                  validation_steps=1)

    save_history(history, name)
    model.save(name + '.h5')

def eval_models_in_folder(num_pictures):
    """
    размечаем обученной моделью случайные сегменты экг-шек, рисуем картинки
    :param num_pictures: кол-во сегментов ЭКГ (взятых случайным образом), на которых хотим увидеть работу модели
    :return:
    """
    _, generator_test = get_generators(train_batch=0, test_batch=num_pictures)
    batch = next(generator_test)

    folder = "./"
    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        if filename.endswith(".h5"):
            model = load_model(os.path.join(folder,filename),custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef':dice_coef})
            test_model_multimask(model, batch, name="VIS_"+filename[0:-len(".h5")])


def test_model_multimask(model, batch, name):
    """

    :param model: бученная модель
    :param batch: батч из тестовго генератора, (x, ann)
    :param name: имя для серии картинок
    :return:
    """
    x = batch[0]
    ann = batch[1]
    print("модель предсказывает на х с формой " + str(x.shape))
    predictions = model.predict_on_batch(x)
    print("предсказания (ann) имеют форму " + str(predictions.shape))
    for i in range(len(predictions)):
        predicted = predictions[i]
        true_ans = ann[i]
        signal_in_channel = x[i]

        # для удобства рисования свопаем оси
        predicted = np.swapaxes(predicted, 0, 1)
        true_ans = np.swapaxes(true_ans, 0, 1)
        plot_name = "VIS" + name + "_" + str(i) + ".png"

        f, (ax1, ax2) = plt.subplots(1, 2, sharey=False, sharex=False)
        x = range(0, len(signal_in_channel))

        ax1.plot(predicted[0], 'r-', label="сырой отв.0")
        ax1.plot(predicted[1], 'y-', label="сырой отв.1")
        ax1.plot(predicted[2], 'b-', label="сырой отв.2")

        ax2.plot(predicted, 'm-', label="ЭКГ")
        ax2.fill_between(x, 0, 10, alpha=0.5, where=true_ans[0] > 0.6, label="правильн.отв.", facecolor='red')
        ax2.fill_between(x, 0, 10, alpha=0.5, where=true_ans[1] > 0.6, label="правильн.отв.", facecolor='yellow')
        ax2.fill_between(x, 0, 10, alpha=0.5, where=true_ans[2] > 0.6, label="правильн.отв.", facecolor='blue')

        plt.legend(loc=2)
        plt.savefig(plot_name)

    print("картинки сохранены!")

if __name__ == "__main__":
    name = "maria_annotator"

    train(name)
    eval_models_in_folder(40)



