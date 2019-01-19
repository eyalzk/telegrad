''' Demonstrate the usage of the telegram callback on the official Keras MNIST example.
(https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py)
Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
By: Eyal Zakkay, 2019
https://eyalzk.github.io/
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


##################################################################################################
"""  The following block is all you need in order 
  to use the Keras Telegram bot callback. Just 
  add telegram_callback to the list of        
  callbacks passed to model.fit       """

# Telegram Bot imports
from dl_bot import DLBot
from telegram_bot_callback import TelegramBotCallback

telegram_token = "TOKEN"  # replace TOKEN with your bot's token

# user id is optional, however highly recommended as it limits the access to you alone.
telegram_user_id = None   # replace None with your telegram user id (integer):

# Create a DLBot instance
bot = DLBot(token=telegram_token, user_id=telegram_user_id)
# Create a TelegramBotCallback instance
telegram_callback = TelegramBotCallback(bot)

##################################################################################################

# Run MNIST example
batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[telegram_callback])

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# You can also send messages regardless of the callback. Just use bot.send_message(some_string)
bot.send_message('Test loss:' + str(score[0]))
bot.send_message('Test accuracy:' + str(score[1]))
