import glob
from keras.layers import Dense, Reshape, Deconvolution2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
import numpy as np
from skimage import io
from skimage.color import rgb2gray

__author__ = 'Balint'


def load_images(directory, filetype='*.png', start=0, stop=np.inf):
    filenames = glob.glob(directory + filetype)
    imgs = []
    for i in xrange(start, min(len(filenames), stop)):
        name = filenames[i]
        img = io.imread(name)
        imgs.append(img)

    return imgs

def get_gt_index(frame_no):
    return int(round((frame_no / 25.0 + 0.466667) * 30 % 20))

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

def train(gts, imgs,width, height, p_batch_size=200, p_nb_epochs=10, p_validation_split=0.05, p_reg=0.01, p_dropout=0.5):
    img_dims = len(imgs[0])
    print img_dims, width*height
    model = Sequential()
    model.add(Dense(1024, input_shape=gts[0].shape))
    model.add(Dense(width*height*32))
    model.add(LeakyReLU())
    model.add(Reshape((32, height, width)))
    model.add(Deconvolution2D(32,3,3, activation='relu'))
    # model.add(UpSampling2D())
    model.add(Deconvolution2D(3,3,3, activation='sigmoid'))

    # model.add(Reshape((9,img_dims)))
    # model.add(Reshape((128,64)))
    #
    # model.add(Dense(img_dims))

    model.compile(loss="mse", optimizer='rmsprop')
    model.fit(gts, imgs, batch_size=p_batch_size, nb_epoch=p_nb_epochs, validation_split=p_validation_split)
    print model.summary()
    return model



def main(left_dir, right_dir, gt_dir, out_file="predicted.csv"):
    stop = 10
    left_imgs = load_images(left_dir, stop=stop)
    # right_imgs = load_images(right_dir, stop=stop)
    x_train=[]
    y_train=[]
    width, height, channels = left_imgs[0].shape
    print left_imgs[0].shape
    for i in xrange(len(left_imgs)):
        x_train.append([get_gt_index(i)])
        y_train.append(rgb2gray(left_imgs[i]).flatten())
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    print x_train[0].shape
    # gts = load_gt(gt_dir)
    # (x_train, y_train), (x_test, y_test) = generate_training_data(left_imgs, right_imgs, gts, stop)
    model = train(x_train, y_train, width, height)
    # predicted = model.predict(x_test)
    # rmse = np.sqrt(((predicted - y_test) ** 2).mean())
    # print rmse
    # pd.DataFrame(predicted).to_csv(out_file, index=False)


if __name__ == "__main__":
    base = 'd:\\dev\\datasets\\heart\\'
    main(base + 'left\\', base + 'right\\', base + 'gt\\disparityMap*')