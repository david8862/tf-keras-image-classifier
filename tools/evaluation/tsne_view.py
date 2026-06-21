#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
use TSNE in sklearn to visualize feature distribution
in 2D or 3D dimension. PCA transform is optional
for decreasing original dimension.
'''
from time import time
import os, sys, argparse, random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from classifier.data import get_data_generator
from common.utils import get_classes, get_custom_objects, optimize_tf_gpu

import tensorflow as tf
optimize_tf_gpu(tf, K)


def get_data(data_generator, model):
    # Now loop through and extract features to build the sequence.
    sequence = []
    labels = []

    step = data_generator.samples // data_generator.batch_size
    pbar = tqdm(total=step)
    for i in range(step):
        image_data, label = data_generator.next()
        features = model.predict(image_data, verbose=0)
        # normalize feature vector
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        sequence.append(features[0])
        labels.append(np.argmax(label[0]))
        pbar.update(1)
    pbar.close()

    return np.array(sequence), np.array(labels), len(sequence), len(sequence[0])


def plot_embedding(data, label, title, dim):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    if dim == 2:
        ax = plt.subplot(111)
        #for i in range(data.shape[0]):
            #plt.text(data[i, 0], data[i, 1], str(label[i]),
                     #color=plt.cm.Set1(label[i] / 1.),
                     #fontdict={'weight': 'bold', 'size': 9})
        #plt.xticks([])
        #plt.yticks([])
        for c in range(len(np.unique(label))):
            ax.plot(data[label==c, 0], data[label==c, 1], '.', alpha=0.1)
    elif dim == 3:
        #ax = fig.gca(projection='3d')
        #ax.scatter(data[:, 0], data[:, 1], data[:, 2], c = label, s = 20)

        #ax.view_init(4, -72)
        #ax.set_zlabel('Z')
        #ax.set_ylabel('Y')
        #ax.set_xlabel('X')
        ax = Axes3D(fig)
        for c in range(len(np.unique(label))):
            ax.plot(data[label==c, 0], data[label==c, 1], data[label==c, 2], '.', alpha=0.1)
    plt.title(title)
    return fig



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', help='path to evaluation image dataset', type=str, required=True)
    parser.add_argument('--model_path', help='path to model file', type=str, required=True)
    parser.add_argument('--model_input_shape', help='model image input size as <height>x<width>, default=%(default)s', type=str, default='224x224')
    parser.add_argument('--tsne_dim', help='TSNE dimension to display, 2 or 3. default=2', type=int, default=2)
    parser.add_argument('--pca_level', help='add PCA transform to feature space (optional)', type=float)
    args = parser.parse_args()

    height, width = args.model_input_shape.split('x')
    args.model_input_shape = (int(height), int(width))

    # data generator
    batch_size = 1
    data_generator = get_data_generator(args.dataset_path, args.model_input_shape, batch_size, None, mode='eval')

    # load model
    custom_object_dict = get_custom_objects()
    model = load_model(args.model_path, compile=False, custom_objects=custom_object_dict)
    K.set_learning_phase(0)

    #os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
    data, label, n_samples, n_features = get_data(data_generator, model)

    if args.pca_level:
        pca = PCA(n_components=args.pca_level)
        data = pca.fit_transform(data)
        print('feature space dimension change to {} by PCA'.format(pca.n_components_))

    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=args.tsne_dim, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    print('t-SNE embedding Done')

    fig = plot_embedding(result, label,
                        't-SNE embedding of the feature (time %.2fs)'
                        % (time() - t0), dim=args.tsne_dim)
    plt.show(fig)


if __name__ == '__main__':
    main()

