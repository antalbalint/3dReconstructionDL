import glob
from itertools import chain

from keras.engine import Input
from keras.engine import Model
from keras.engine import merge
from keras.layers import Dropout, Convolution2D, Convolution3D, Flatten, Activation, MaxPooling2D, BatchNormalization
from keras.layers.convolutional import MaxPooling3D
from keras.optimizers import Adadelta
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from skimage import io
import pandas as pd
from skimage.exposure import rescale_intensity
from sklearn import preprocessing
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense
import numpy as np


def load_images(directory, filetype='*.png', start=0, stop=np.inf):
    filenames = glob.glob(directory + filetype)
    imgs = []
    for i in xrange(start, min(len(filenames), stop)):
        name = filenames[i]
        img = io.imread(name)
        imgs.append(img)

    return imgs


def show_disparity_map(map_file, width, height, im_name='distance_map.png'):
    disparity_map = []
    with open(map_file) as f:
        content = f.readlines()
        for line in content:
            disparity_map.append(float(line))
    disparity_map = np.asarray(disparity_map)
    disparity_map = np.reshape(disparity_map, (width, height))
    io.imsave(im_name, rescale_intensity(disparity_map, in_range='image', out_range='dtype'))
    plt.imshow(disparity_map, cmap='Greys_r')
    plt.show()


def load_gt(directory, start=0, stop=np.inf):
    filenames = glob.glob(directory)
    gts = []
    for i in xrange(start, min(len(filenames), stop)):
        name = filenames[i]
        gt = []
        with open(name) as f:
            content = f.readlines()
            for line in content:
                terms = line.split()
                for j in xrange(len(terms)):
                    terms[j] = float(terms[j])
                gt.append(terms)
        gts.append(gt)
    return gts


def get_gt_index(frame_no):
    return int((round(((frame_no / 25.0 + 0.466667) * 30) % 20)))


def visualize(gt, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for x, y, z in gt:
        ax.scatter(x, y, z)
    plt.savefig(filename)


def generate_vectors(lefts, rights, gts, offset=0, window_size=12):
    data_r = []
    data_l = []
    data_y = []
    for idx in xrange(len(lefts)):
        left = rescale_intensity(lefts[idx], in_range='image', out_range='float')
        right = rescale_intensity(rights[idx], in_range='image', out_range='float')
        gt_idx = get_gt_index(idx + offset)
        gt = gts[gt_idx]
        width, height, channels = left.shape
        for i in xrange(window_size,width-window_size):
            for j in xrange(window_size,height-window_size):
                y = gt[i + width * j]
                if y == [0, 0, 0]:
                    continue
                l = left[i-window_size:i+window_size, j-window_size:j+window_size]
                r = right[i-window_size:i+window_size, j-window_size:j+window_size]

                # x = (l)

                # x = []
                # for p, q in zip(l, r):
                #     x.append(p)
                #     x.append(q)
                data_l.append(l)
                data_r.append(r)
                data_y.append(y)
    return np.asarray(data_l, dtype=float), np.asarray(data_r, dtype=float),np.asarray(data_y, dtype=float)


def generate_training_data(left_imgs, right_imgs, gts):
    l, r, y = generate_vectors(left_imgs, right_imgs, gts)

    # l = preprocessing.scale(l)
    # r = preprocessing.scale(r)
    n_samples = len(l)
    idx_rnd = np.random.permutation(n_samples)
    l = l[idx_rnd]
    r = r[idx_rnd]
    y = y[idx_rnd]
    l_train = l[0:n_samples / 2]
    r_train = r[0:n_samples / 2]
    y_train = y[0:n_samples / 2]
    l_test = l[n_samples / 2 + 1:-1]
    r_test = l[n_samples / 2 + 1:-1]

    y_test = y[n_samples / 2 + 1:-1]
    return (l_train,r_train, y_train), (l_test, r_test, y_test)


def train(l_train, r_train, y_train, p_batch_size=200, p_nb_epochs=10, p_validation_split=0.05, p_reg=0.01, p_dropout=0.5):
    l_train = l_train
    r_train = r_train
    y_train = y_train

    in_neurons = len(l_train[0])
    print l_train[0].shape
    out_neurons = len(y_train[0])
    hidden_neurons = 500


    input_1 = Input(shape=l_train[0].shape)
    x = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform')(input_1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)
    x = Convolution2D(64, 3, 3, border_mode='same', init='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(64, 3, 3, border_mode='same', init='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)
    x = Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)
    # x = Convolution2D(32, 3, 3, border_mode='same', activation='relu', W_regularizer=l2(p_reg),
    #                   init='glorot_normal')(x)
    # # x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = Convolution2D(64, 3, 3, border_mode='same', activation='relu', W_regularizer=l2(p_reg),
    #                   init='glorot_normal')(x)
    # # x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = Convolution2D(64, 3, 3, border_mode='same', activation='relu', W_regularizer=l2(p_reg),
    #                   init='glorot_normal')(x)
    # # x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = Convolution2D(128, 3, 3, border_mode='same', activation='relu', W_regularizer=l2(p_reg),
    #                   init='glorot_normal')(x)
    # # x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = Convolution2D(128, 3, 3, border_mode='same', activation='relu', W_regularizer=l2(p_reg),
    #                   init='glorot_normal')(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    input_2 = Input(shape=r_train[0].shape)
    y = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform')(input_2)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = MaxPooling2D()(y)
    y = Convolution2D(64, 3, 3, border_mode='same', init='glorot_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Convolution2D(64, 3, 3, border_mode='same', init='glorot_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = MaxPooling2D()(y)
    y = Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = MaxPooling2D()(y)
    # y = Convolution2D(128, 5, 5, border_mode='same', activation='relu', W_regularizer=l2(p_reg),
    #                   init='glorot_normal')(input_2)
    # # y = MaxPooling2D(pool_size=(2, 2))(y)
    # y = Convolution2D(128, 5, 5, border_mode='same', activation='relu', W_regularizer=l2(p_reg),
    #                   init='glorot_normal')(y)
    # # y = MaxPooling2D(pool_size=(2, 2))(y)
    # y = Convolution2D(64, 3, 3, border_mode='same', activation='relu', W_regularizer=l2(p_reg),
    #                   init='glorot_normal')(y)
    # # y = MaxPooling2D(pool_size=(2, 2))(y)
    # y = Convolution2D(64, 3, 3, border_mode='same', activation='relu', W_regularizer=l2(p_reg),
    #                   init='glorot_normal')(y)
    # # y = MaxPooling2D(pool_size=(2, 2))(y)
    # y = Convolution2D(32, 3, 3, border_mode='same', activation='relu', W_regularizer=l2(p_reg),
    #                   init='glorot_normal')(y)
    # # y = MaxPooling2D(pool_size=(2, 2))(y)
    # y = Convolution2D(32, 3, 3, border_mode='same', activation='relu', W_regularizer=l2(p_reg),
    #                   init='glorot_normal')(y)
    # y = MaxPooling2D(pool_size=(2, 2))(y)
    z = merge([x, y], mode='concat')
    z = Flatten()(z)
    z = Dense(4096, init='glorot_uniform')(z)
    z = BatchNormalization()(z)
    z = Activation('relu')(z)
    z = Dense(out_neurons, init='glorot_uniform')(z)
    # z = BatchNormalization()(z)
    # z = Activation('linear')(z)

    # l_datagen = ImageDataGenerator(zca_whitening=True)
    # r_datagen = ImageDataGenerator(zca_whitening=True)
    #
    # l_datagen.fit(l_train)
    # r_datagen.fit(r_train)
    model = Model(input=[input_1, input_2], output=z)
    # model = Sequential()
    # model.add(Convolution3D(32, 3, 3, 3, input_shape=x_train[0].shape,border_mode='same'))
    # model.add(Activation('relu'))
    # model.add(MaxPooling3D(pool_size=(2,2,1)))
    # model.add(Convolution3D(32, 3, 3, 3,border_mode='same'))
    # model.add(Activation('relu'))
    # model.add(MaxPooling3D(pool_size=(2,2,1)))
    # model.add(Convolution3D(64, 3, 3, 3,border_mode='same'))
    # model.add(Activation('relu'))
    # model.add(MaxPooling3D(pool_size=(2,2,1),dim_ordering='th'))
    # in_layer = Dense(hidden_neurons, input_dim=in_neurons, W_regularizer=l2(p_reg), activation='relu',
    #                  init='glorot_normal')
    # model.add(in_layer)
    # model.add(
    #     Dense(hidden_neurons, input_dim=hidden_neurons, W_regularizer=l2(p_reg), activation='relu',
    #           init='glorot_normal'))
    # model.add(Dropout(p_dropout))
    # hidden_layer = Dense(hidden_neurons, input_dim=hidden_neurons, W_regularizer=l2(p_reg), activation='relu',
    #                      init='glorot_normal')
    # model.add(hidden_layer)
    # drop_layer = Dropout(p_dropout)
    # model.add(drop_layer)
    #
    # model.add(
    #     Dense(hidden_neurons, input_dim=hidden_neurons, W_regularizer=l2(p_reg), activation='relu',
    #           init='glorot_normal'))
    # model.add(Dropout(p_dropout))
    # hidden_layer = Dense(hidden_neurons, input_dim=hidden_neurons, W_regularizer=l2(p_reg), activation='relu',
    #                      init='glorot_normal')
    # model.add(hidden_layer)
    # drop_layer = Dropout(p_dropout)
    # model.add(drop_layer)
    #
    # model.add(Flatten())
    # model.add(Dense(64))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(out_neurons))
    # model.add(Activation('relu'))
    # out_layer = Dense(out_neurons, activation='sigmoid', W_regularizer=l2(p_reg),
    #                   init='glorot_normal')
    # model.add(out_layer)
    opt = Adadelta()
    model.compile(loss="mse", optimizer=opt)
    print model.summary()

    model.fit([l_train, r_train], y_train, batch_size=p_batch_size, nb_epoch=p_nb_epochs, validation_split=p_validation_split)
    # model.fit_generator(zip(l_datagen, r_datagen))
    print model.summary()
    model.save("cnn_model.h5")
    return model


def main(left_dir, right_dir, gt_dir, out_file="predicted.csv"):
    stop = 1
    left_imgs = load_images(left_dir, stop=stop)
    right_imgs = load_images(right_dir, stop=stop)
    gts = load_gt(gt_dir)
    (l_train, r_train, y_train), (l_test, r_test, y_test) = generate_training_data(left_imgs, right_imgs, gts, stop)
    model = train(l_train, r_train, y_train)
    predicted = model.predict([l_test, r_test])
    rmse = np.sqrt(((predicted - y_test) ** 2).mean())
    print rmse
    pd.DataFrame(predicted).to_csv(out_file, index=False)


if __name__ == "__main__":
    base = '/home/balint/dev/datasets/heart/'
    main(base + 'left/', base + 'right/', base + 'gt/disparityMap*')
