import BaselineWanderRemoval as bwr
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.callbacks import (
    TensorBoard
)
from keras.layers import (
    LSTM,
    Bidirectional,
    Dense,
    Conv1D,
    MaxPooling1D,
    UpSampling1D
)
from keras.models import (
    Sequential, load_model
)

from dataset_getter import prepare_data

num_decoder_tokens = 4
Size = 2992
num_cores = 32
kernel = 8
num_latent = 50
leads = 12

x_train, x_test, y_train, y_test = prepare_data(seg_len=None)
# отрезаем куски без разметки
x_train = x_train[:, 1000:4000, :, :]
x_test = x_test[:, 1000:4000, :, :]
y_train = y_train[:, 1000:4000]
y_test = y_test[:, 1000:4000]


def one_hot(y):
    """
    переводит небинарную маску в one hot
    :param y:
    :return:
    """
    y_new = []
    for i in range(y.shape[0]):
        foo = []
        for j in range(y.shape[1]):
            tmp = np.zeros(num_decoder_tokens)
            if y[i, j] != 0:
                tmp[y[i, j]] = 1
            else:
                tmp[0] = 1
            foo.append(tmp)
        y_new.append(foo)

    return y_new


y_train = np.array(one_hot(y_train[:, :Size]))
y_test = np.array(one_hot(y_test[:, :Size]))

# выравниваем изолинию
for i in range(x_train.shape[0]):
    for j in range(leads):
        x_train[i, :, 0, j] = bwr.fix_baseline_wander(x_train[i, :, 0, j], 500)
for i in range(x_test.shape[0]):
    for j in range(leads):
        x_test[i, :, 0, j] = bwr.fix_baseline_wander(x_test[i, :, 0, j], 500)

model = Sequential()

model.add(Conv1D(num_cores, kernel_size=kernel,
                 activation=K.elu,
                 input_shape=(Size, leads), padding='same'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(num_cores, kernel_size=kernel, activation=K.elu, padding='same'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(num_cores, kernel_size=kernel, activation=K.elu, padding='same'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(num_cores, kernel_size=kernel, activation=K.elu, padding='same'))
model.add(MaxPooling1D(pool_size=2))


model.add(Bidirectional(LSTM(num_latent, return_sequences=True)))

model.add(UpSampling1D(2))
model.add(Conv1D(num_cores, kernel_size=kernel, activation=K.elu, padding='same'))
model.add(UpSampling1D(2))
model.add(Conv1D(num_cores, kernel_size=kernel, activation=K.elu, padding='same'))
model.add(UpSampling1D(2))
model.add(Conv1D(num_cores, kernel_size=kernel, activation=K.elu, padding='same'))
model.add(UpSampling1D(2))
model.add(Conv1D(num_cores, kernel_size=kernel, activation=K.elu, padding='same'))
model.add(Dense(num_decoder_tokens, activation='softmax'))

tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=1, write_graph=True, write_images=True)

model.summary()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

model.fit(x_train[:,:Size,0,0:leads], y_train,
          batch_size=64,
          epochs=10,
          validation_data=((x_test[:,:Size,0,0:leads]), y_test ), callbacks=[tbCallBack])


model.save('CNN1_12_leads1'+'.h5')

#model = load_model('./CNN1_12_leads1.h5')

pred_test = model.predict(x_test[:,:Size,0,0:leads])
predict_test = np.array(pred_test)

x_axis = np.arange(0, Size / 500, 1 / 500)


for i in range(139):
    plt.figure(figsize=(20, 5))
    plt.plot(x_axis, x_test[i, :Size, 0, 0], 'k')
    for j in range(Size):
        max = 0
    for j in range(Size):
        max = 0
        if pred_test[i, j, 0] > pred_test[i, j, 1]:
            max = pred_test[i, j, 0]
            predict_test[i, j, 1] = 0
            predict_test[i, j, 2] = 0
            predict_test[i, j, 3] = 0
        else:
            max = pred_test[i, j, 1]
            predict_test[i, j, 1] = 1
            predict_test[i, j, 2] = 0
            predict_test[i, j, 3] = 0
        if pred_test[i, j, 2] > max:
            max = pred_test[i, j, 2]
            predict_test[i, j, 1] = 0
            predict_test[i, j, 2] = 1
            predict_test[i, j, 3] = 0
        if pred_test[i, j, 3] > max:
            max = pred_test[i, j, 3]
            predict_test[i, j, 1] = 0
            predict_test[i, j, 2] = 0
            predict_test[i, j, 3] = 1

    plt.fill_between(x_axis, y_test[i, :Size, 1] * 25 + 25, 25, color='r', alpha=0.3)
    plt.fill_between(x_axis, y_test[i, :Size, 2] * 25 + 25, 25, color='g', alpha=0.3)
    plt.fill_between(x_axis, y_test[i, :Size, 3] * 25 + 25, 25, color='b', alpha=0.3)
    plt.fill_between(x_axis, list(predict_test[i, :, 1] * 25), 0, color='r', alpha=0.3)
    plt.fill_between(x_axis, list(predict_test[i, :, 2] * 25), 0, color='g', alpha=0.3)
    plt.fill_between(x_axis, list(predict_test[i, :, 3] * 25), 0, color='b', alpha=0.3)
    #plt.savefig("test_fig" + str(i) + ".png", dpi = 200)
    plt.show()
    plt.clf()
