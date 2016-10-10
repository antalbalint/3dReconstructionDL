import glob
from keras.layers import Dropout
from keras.optimizers import Adadelta
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
    return int(round((frame_no / 25.0 + 0.466667) * 30 % 20))


def visualize(gt, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for x, y, z in gt:
        ax.scatter(x, y, z)
    plt.savefig(filename)


def generate_vectors(lefts, rights, gts, offset=0):
    data_x = []
    data_y = []
    for idx in xrange(len(lefts)):
        left = lefts[idx]
        right = rights[idx]
        gt_idx = get_gt_index(idx + offset)
        gt = gts[gt_idx]
        width, height, channels = left.shape
        for i in xrange(width):
            for j in xrange(height):
                l = left[i, j]
                r = right[i, j]
                y = gt[i + width * j]
                if y == [0, 0, 0]:
                    continue
                x = [i, j]
                for p, q in zip(l, r):
                    x.append(p)
                    x.append(q)
                data_x.append(x)
                data_y.append(y)
    return np.asarray(data_x, dtype=float), np.asarray(data_y, dtype=float)


def generate_training_data(left_imgs, right_imgs, gts, stop):
    x, y = generate_vectors(left_imgs, right_imgs, gts, stop)
    print x.shape
    x = preprocessing.scale(x)
    n_samples = len(x)
    idx_rnd = np.random.permutation(n_samples)
    x = x[idx_rnd]
    y = y[idx_rnd]
    x_train = x[0:n_samples / 2]
    y_train = y[0:n_samples / 2]
    x_test = x[n_samples / 2 + 1:-1]
    y_test = y[n_samples / 2 + 1:-1]
    return (x_train, y_train), (x_test, y_test)


def train(x_train, y_train, p_batch_size=200, p_nb_epochs=50, p_validation_split=0.05, p_reg=0.01, p_dropout=0.5):
    in_neurons = len(x_train[0])
    out_neurons = len(y_train[0])
    hidden_neurons = 500

    model = Sequential()
    in_layer = Dense(hidden_neurons, input_dim=in_neurons, W_regularizer=l2(p_reg), activation='relu',
                     init='glorot_normal')
    model.add(in_layer)
    model.add(
        Dense(hidden_neurons, input_dim=hidden_neurons, W_regularizer=l2(p_reg), activation='relu',
              init='glorot_normal'))
    model.add(Dropout(p_dropout))
    hidden_layer = Dense(hidden_neurons, input_dim=hidden_neurons, W_regularizer=l2(p_reg), activation='relu',
                         init='glorot_normal')
    model.add(hidden_layer)
    drop_layer = Dropout(p_dropout)
    model.add(drop_layer)

    model.add(
        Dense(hidden_neurons, input_dim=hidden_neurons, W_regularizer=l2(p_reg), activation='relu',
              init='glorot_normal'))
    model.add(Dropout(p_dropout))
    hidden_layer = Dense(hidden_neurons, input_dim=hidden_neurons, W_regularizer=l2(p_reg), activation='relu',
                         init='glorot_normal')
    model.add(hidden_layer)
    drop_layer = Dropout(p_dropout)
    model.add(drop_layer)

    out_layer = Dense(out_neurons, input_dim=hidden_neurons, activation='relu', W_regularizer=l2(p_reg),
                      init='glorot_normal')
    model.add(out_layer)
    opt = Adadelta()
    model.compile(loss="mse", optimizer=opt)

    model.fit(x_train, y_train, batch_size=p_batch_size, nb_epoch=p_nb_epochs, validation_split=p_validation_split)
    return model


def main(left_dir, right_dir, gt_dir, out_file="predicted.csv"):
    stop = 1
    left_imgs = load_images(left_dir, stop=stop)
    right_imgs = load_images(right_dir, stop=stop)
    gts = load_gt(gt_dir)
    (x_train, y_train), (x_test, y_test) = generate_training_data(left_imgs, right_imgs, gts, stop)
    model = train(x_train, y_train)
    predicted = model.predict(x_test)
    rmse = np.sqrt(((predicted - y_test) ** 2).mean())
    print rmse
    pd.DataFrame(predicted).to_csv(out_file, index=False)


if __name__ == "__main__":
    base = 'd:\\dev\\datasets\\heart\\'
    main(base + 'left\\', base + 'right\\', base + 'gt\\disparityMap*')
