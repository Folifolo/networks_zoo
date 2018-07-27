import  BaselineWanderRemoval as bwr
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
    Sequential,
)

from dataset_getter import prepare_data

num_decoder_tokens = 4
Size = 3000
num_cores = 16
num_latent = 50

x_train, x_test, y_train, y_test = prepare_data(seg_len=None)
#отрезаем куски без разметки
x_train = x_train[:,1000:4000,:,:]
x_test = x_test[:,1000:4000,:,:]
y_train = y_train[:,1000:4000]
y_test = y_test[:,1000:4000]

def one_hot(y):
    """
    переводит небинарную маску в one hot
    :param y:
    :return:
    """
    y_new = []
    for i in range(y.shape[0]):
        foo = []
        for j in range (y.shape[1]):
            tmp = np.zeros(num_decoder_tokens)
            if y[i,j] != 0:
                tmp[y[i,j]] = 1
            else:
                tmp[0] = 1
            foo.append(tmp)
        y_new.append(foo)

    return y_new


y_train = np.array(one_hot(y_train[:,:Size]))
y_test = np.array(one_hot(y_test[:,:Size]))


#выравниваем изолинию
for i in range(x_train.shape[0]):
    x_train[i, :,0,0] = bwr.fix_baseline_wander( x_train[i, :,0,0], 500)
for i in range(x_test.shape[0]):
    x_test[i, :,0,0] = bwr.fix_baseline_wander( x_test[i, :,0,0], 500)


model = Sequential()

model.add(Conv1D(num_cores, kernel_size=6,
                 activation=K.elu,
                 input_shape=(None, 1), padding='same'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(num_cores,kernel_size=6, activation=K.elu, padding='same'))
model.add(MaxPooling1D(pool_size=2))

model.add(Bidirectional(LSTM(num_latent,return_sequences=True)))

model.add(Conv1D(num_cores,kernel_size=6, activation=K.elu, padding='same'))
model.add(UpSampling1D(2))
model.add(Conv1D(num_cores,kernel_size=6, activation=K.elu, padding='same'))
model.add(UpSampling1D(2))
model.add(Dense(num_decoder_tokens, activation='softmax'))



tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=1, write_graph=True, write_images=True)

model.summary()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

model.fit(x_train[:,:Size,0:1,0], y_train,
          batch_size=64,
          epochs=20,
          validation_data=((x_test[:,:Size,0:1,0]), y_test ), callbacks=[tbCallBack])

model.save('CNN5'+'.h5')

#model = load_model('./CNN5.h5')


pred_test = model.predict(x_test[:,:Size,0:1,0])
predict_test = np.array(pred_test)



x_axis = np.arange(0, Size/500, 1/500)

# это мне почему-то взбрело в голову ручками максимум писать
for i in range (100):
    plt.plot(x_axis, x_test[i,:Size,0,0],'k')
    for j in range (Size):
        max = 0
    for j in range (Size):
        max = 0
        if pred_test[i,j,0]>pred_test[i,j,1]:
            max=pred_test[i,j,0]
            predict_test[i,j,1]=0
            predict_test[i,j,2]=0
            predict_test[i,j,3]=0
        else:
            max = pred_test[i,j,1]
            predict_test[i,j,1]=1
            predict_test[i,j,2]=0
            predict_test[i,j,3]=0
        if pred_test[i,j,2]>max:
            max = pred_test[i,j,2]
            predict_test[i,j,1]=0
            predict_test[i,j,2]=1
            predict_test[i,j,3]=0
        if  pred_test[i,j,3]>max:
            max= pred_test[i,j,3]
            predict_test[i,j,1]=0
            predict_test[i,j,2]=0
            predict_test[i,j,3]=1

    plt.fill_between(x_axis, y_test[i,:Size,1]*25+25, 25, color=   'r', alpha= 0.3)
    plt.fill_between(x_axis, y_test[i,:Size,2]*25+25, 25, color=   'g', alpha= 0.3)
    plt.fill_between(x_axis, y_test[i,:Size,3]*25+25, 25, color=   'b', alpha= 0.3)
    plt.fill_between(x_axis, list(predict_test[i,:,1]*25), 0, color=   'r', alpha= 0.3)
    plt.fill_between(x_axis, list(predict_test[i,:,2]*25), 0, color=   'g', alpha= 0.3)
    plt.fill_between(x_axis, list(predict_test[i,:,3]*25), 0, color=   'b', alpha= 0.3)
    plt.show()
