import os
import struct
import numpy as np
from matplotlib import pyplot
import matplotlib as mpl
"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

def read(dataset = "training"):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """
    if dataset is "training":
        fname_image = "./data/train-images.idx3-ubyte"
        fname_label = './data/train-labels.idx1-ubyte'
    elif dataset is "testing":
        fname_image = "./data/t10k-images.idx3-ubyte"
        fname_label = "./data/t10k-labels.idx1-ubyte"
    else:
        print("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_label, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        lbl = np.fromfile(f, dtype=np.int8)

    with open(fname_image, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])
    # Create an iterator which returns each image in turn
    for i in range(len(lbl)):
        yield get_img(i)

def showImg(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()

def main():
    #0: 10, 13, 9: 7,9, 12, 8: 110, 146,179, 6:161,164
    mnist = list(read("testing"))
    print(len(mnist))
    for i in range(100,200):
        label, pixels=mnist[i]
        print (i,": ",label)
    label, pixels = mnist[184]
    print(pixels.shape)
    print(pixels)
    showImg(pixels)

if __name__ == '__main__':
    main()