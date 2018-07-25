import os
import json
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import pyedflib
import numpy as np
import pickle as pkl

def zero_to_nan(values):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x<0.01 else x for x in values]

def make_mask(values):
    return [0 if x <=0.5 else 1.0 for x in values]

def draw_prediction_and_reality(ecg_signal, prediction, right_answer, plot_name):
    """

    :param ecg_signal: сигнал некотего отведения
    :param prediction: предсказаные 3 бинарные маски для этого отведения
    :param right_answer: правильная маска этого отведения (тоже три штуки)
    :param plot_name: имя картинки, куда хотим отрисовать это
    :return:
    """
    figname = plot_name + "_.png"
    print("Рисуем:")
    print(prediction.shape, " prediction.shape")
    print(right_answer.shape, " right_answer.shape")
    print(ecg_signal.shape, " ecg_signal.shape")
    assert len(prediction) == len(right_answer) == 3 #3 бинарные маски

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, sharex=True)
    #ax1.plot(ecg_signal)
    #ax2.plot(ecg_signal)
    #ax3.plot(ecg_signal)
    x = range(0, len(ecg_signal))

    ax1.fill_between(x, 0, prediction[0],where=prediction[0]>0.5, label="мнение сети",facecolor='green',alpha=0.5 )
    ax2.fill_between(x, 0, prediction[1], where=prediction[1]>0.5,label="мнение сети.",facecolor='green',alpha=0.5)
    ax3.fill_between(x, 0, prediction[2],where=prediction[2]>0.5, label="мнение сети.",facecolor='green',alpha=0.5)

    ax1.plot(prediction[0], 'k-', label="сырой отв.")
    ax2.plot(prediction[1], 'k-', label="сырой отв.")
    ax3.plot(prediction[2], 'k-', label="сырой отв.")


    ax1.fill_between(x,0,right_answer[0], alpha=0.5, label="правильн.отв.", facecolor='red')
    ax2.fill_between(x,0,right_answer[1], alpha=0.5, label="правильн.отв.", facecolor='red')
    ax3.fill_between(x,0,right_answer[2], alpha=0.5, label="правильн.отв.", facecolor='red')

    plt.legend(loc=2)
    plt.savefig(figname)

def test_model(model, batch, name):
    """

    :param model: бученная модель
    :param batch: батч из тестовго генератора, (x, ann)
    :param name: имя для серии картинок
    :return:
    """
    print ("модель предсказывает на х с формой " + str(batch[0].shape))
    predictions = model.predict_on_batch(batch[0])
    print("предсказания (ann) имеют форму " + str(predictions.shape))
    for i in range(len(predictions)):
        predicted = predictions[i]
        true_ans = batch[1][i]

        signal_in_channel = batch[0][i]

        #signal_in_channel = batch[0][i][:,0] # i-тая кардиограмма, нулевое отведение

        #для удобства рисования свопаем оси
        predicted = np.swapaxes(predicted, 0, 1)
        true_ans = np.swapaxes(true_ans, 0, 1)

        draw_prediction_and_reality_simple(ecg_signal=signal_in_channel,
                                        prediction=predicted,
                                        right_answer=true_ans,
                                        plot_name=name + "_" + str(i))
    print("картинки сохранены!")

def draw_prediction_and_reality_simple(ecg_signal, prediction, right_answer, plot_name):
    """

    :param ecg_signal: сигнал некотего отведения
    :param prediction: предсказаные 3 бинарные маски для этого отведения
    :param right_answer: правильная маска этого отведения (тоже три штуки)
    :param plot_name: имя картинки, куда хотим отрисовать это
    :return:
    """
    figname = plot_name + "_.png"
    print("Рисуем:")
    print(prediction.shape, " prediction.shape")
    print(right_answer.shape, " right_answer.shape")
    print(ecg_signal.shape, " ecg_signal.shape")
    assert len(prediction) == len(right_answer) == 1 #1 бинарня маски

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=False, sharex=False)
    x = range(0, len(ecg_signal))

    ax1.fill_between(x, 0, prediction[0],where=prediction[0]>0.5, label="мнение сети",facecolor='green',alpha=0.5 )
    ax1.plot(prediction[0], 'k-', label="сырой отв.")
    ax1.fill_between(x,0,right_answer[0], alpha=0.5, label="правильн.отв.", facecolor='red')

    ax2.plot(ecg_signal, 'm-', label="ЭКГ")
    ax2.fill_between(x, 0, 10, alpha=0.5, where=right_answer[0]>0.6, label="правильн.отв.", facecolor='red')

    plt.legend(loc=2)
    plt.savefig(figname)



