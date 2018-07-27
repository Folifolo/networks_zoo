import matplotlib.pyplot as plt
import numpy as np
from keras.layers import (
    LSTM,
    Bidirectional,
    Input,
    Dense
)
from keras.models import (
    Sequential,
    load_model
)

from dataset_getter import prepare_data

latent_dim = 100
num_decoder_tokens = 4
Size = 500


x_train, x_test, y_train, y_test = prepare_data(seg_len=None)
x_train = x_train[:,1000:4000,:,:]
x_test = x_test[:,1000:4000,:,:]
y_train = y_train[:,1000:4000]
y_test = y_test[:,1000:4000]

def one_hot(y):
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

y_tr = np.array(one_hot(y_train[:,:Size]))
y_tst = np.array(one_hot(y_test[:,:Size]))
'''
xes = np.arange(0, Size/500, 1/500)
plt.plot(xes, y_tr[0,:Size,0]*100,'b')
plt.plot(xes, y_tr[0,:Size,1]*100,'r')
plt.plot(xes, y_tr[0,:Size,2]*100,'g')
plt.plot(xes, x_train[0,:Size,0,0], 'k')
plt.show()
'''
print (y_tr.shape)


def encoder():
    model = Sequential()
    model.add(LSTM(latent_dim, input_shape=(Size, 1)))
    print ('enc shape = ' + str(model.output_shape))
    return model

def decoder():
    model = Sequential()
    model.add(LSTM(latent_dim, input_shape=(Size, latent_dim)))
    model.add(Dense(num_decoder_tokens, activation='softmax'))
    print ('dec shape = ' + str(model.output_shape))
    return model


base_encoder = encoder()
base_decoder = decoder()
encoder_inputs = Input(shape=(Size, 1))
decoder_inputs = Input(shape=(Size, latent_dim))

enmodel = base_encoder(encoder_inputs)
demodel = base_decoder(decoder_inputs)

model = Sequential()
model.add(Bidirectional(LSTM(latent_dim,return_sequences=True), input_shape=(Size, 1)))
#model.add(LSTM(latent_dim, return_sequences=True))
model.add(Dense(num_decoder_tokens, activation='softmax'))


input = encoder_inputs
#encode = base_encoder(input)
#model = Model(input, decoder()(encode))
model.summary()
print (x_train[:,:,0:1,0].shape)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

#model.fit(x_train[:,:Size,0:1,0], y_tr,
#          batch_size=64,
#          epochs=10,
#          validation_split=0.2)
#model.save('test4'+'.h5')

model = load_model('C:\\Users\\donte_000\\Documents\\networks_zoo\\attention\\test4.h5')
pred = model.predict(x_test[:,:Size,0:1,0])
print(pred.shape)
plt.figure(1)
xes = np.arange(0, Size/500, 1/500)
predict = np.array(pred)
for i in range (100):
    plt.plot(xes, x_test[i,:Size,0,0],'k')
    for j in range (Size):
        max = 0
        if pred[i,j,0]>pred[i,j,1]:
            max=pred[i,j,0]
            predict[i,j,1]=0
            predict[i,j,2]=0
            predict[i,j,3]=0
        else:
            max = pred[i,j,1]
            predict[i,j,1]=1
            predict[i,j,2]=0
            predict[i,j,3]=0
        if pred[i,j,2]>max:
            max = pred[i,j,2]
            predict[i,j,1]=0
            predict[i,j,2]=1
            predict[i,j,3]=0
        if  pred[i,j,3]>max:
            max= pred[i,j,3]
            predict[i,j,1]=0
            predict[i,j,2]=0
            predict[i,j,3]=1

    plt.fill_between(xes, y_test[i,:Size]*100, 'y')
    plt.plot(xes, list(predict[i,:,1]*100),'r')
    plt.plot(xes, list(predict[i,:,2]*100),'g')
    plt.plot(xes, list(predict[i,:,3]*100),'b')

    plt.show()



