import numpy as np

from math import ceil
import io

import tensorflow as tf

import matplotlib as mpl
mpl.use('agg')

import seaborn as sns
sns.set_style('white')
sns.set_context('paper')
sns.set_color_codes()

import matplotlib.pyplot as plt


def hidden_conv3D(inp, out_chnls, conv_patch=5, pool_patch=2, name='conv'):
    """Create hidden layer with 3D convolution and max pooling. This function
    performs transformation:

        y = max_pool(relu(conv(x, w) + b)))

    where x is an input layer and y is an output layer returned by this
    function. All created tensors are in the name scope defined by `name`.

    Parameters
    ----------
    inp: 5D tf.Tensor
        Input tensor
    out_chnls: int
        Number of filters in the convolution.
    conv_patch: int, optional
        Size of convolution patch
    pool_patch: int, optional
        Size of max pooling patch
    name: str, optional
        Name for the `variable_scope` with all created tensors.

    Returns
    -------
    h_pool: 5D tf.Tensor
        Output tensor
    """

    in_chnls = inp.get_shape()[-1].value
    with tf.variable_scope(name):
        w_shape = (conv_patch, conv_patch, conv_patch, in_chnls, out_chnls)

        w = tf.get_variable('w', shape=w_shape,
                            initializer=tf.truncated_normal_initializer(
                                stddev=0.001))
        b = tf.get_variable('b', shape=(out_chnls,), dtype=tf.float32,
                            initializer=tf.constant_initializer(0.1))

        conv = tf.nn.conv3d(inp, w, strides=(1, 1, 1, 1, 1), padding='SAME',
                            name='conv')
        h = tf.nn.relu(conv + b, name='h')

        pool_shape = (1, pool_patch, pool_patch, pool_patch, 1)
        h_pool = tf.nn.max_pool3d(h, ksize=pool_shape, strides=pool_shape,
                                  padding='SAME', name='h_pool')
    return h_pool


def hidden_fcl(inp, out_size, keep_prob, name='fc'):
    """Create fully-connected hidden layer with dropout. This function
    performs transformation:

        y = dropout(relu(wx + b)))

    where x is an input layer and y is an output layer returned by this
    function. All created tensors are in the name scope defined by `name`.

    Parameters
    ----------
    inp: 2D tf.Tensor
        Input tensor
    out_size: int
        Number of neurons in the layer.
    keep_prob: float or 0D tf.Tensor
        Keep probability for dropout layer
    name: str, optional
        Name for the `variable_scope`

    Returns
    -------
    h_drop: 2D tf.Tensor
        Output tensor
    """

    assert len(inp.get_shape()) == 2

    in_size = inp.get_shape()[1].value

    with tf.variable_scope(name):
        w = tf.get_variable('w', shape=(in_size, out_size),
                            initializer=tf.truncated_normal_initializer(
                                stddev=(1 / (in_size**0.5))))
        b = tf.get_variable('b', shape=(out_size,), dtype=tf.float32,
                            initializer=tf.constant_initializer(1))

        h = tf.nn.relu(tf.matmul(inp, w) + b, name='h')
        h_drop = tf.nn.dropout(h, keep_prob, name='h_dropout')

    return h_drop


def convolve3D(inp, channels, conv_patch=5, pool_patch=2):
    """Create block of 3D convolutional layers with max pooling using
    `hidden_conv3D`. The i'th layer in the block is in name scope called
    'conv<i>' and has channels[i] number of filters.

    Parameters
    ----------
    inp: 5D tf.Tensor
        Input tensor
    channels: array-like, shape = (N,)
        Numbers of filters in convolutions.
    conv_patch: int, optional
        Size of convolution patch
    pool_patch: int, optional
        Size of max pooling patch


    Returns
    -------
    output: 5D tf.Tensor
        Output of last layer
    """

    prev = inp
    i = 0
    for num_channels in channels:
        output = hidden_conv3D(prev, num_channels, conv_patch, pool_patch,
                               name='conv%s' % i)
        i += 1
        prev = output
    return output


def feedforward(inp, fc_sizes, keep_prob=1.0):
    """Create block of fully-connected layers with dropout using
    `hidden_fcl`. The i'th layer in the block is in name scope called
    'fc<i>' and has fc_sizes[i] number of neurons.

    Parameters
    ----------
    inp: 2D tf.Tensor
        Input tensor
    fc_sizes: array-like, shape = (N,)
        Numbers of neurons in layers.
    keep_prob: float or 0D tf.Tensor, optional
        Keep probability for dropout layer


    Returns
    -------
    output: 2D tf.Tensor
        Output of last layer
    """

    prev = inp
    i = 0
    for hsize in fc_sizes:
        output = hidden_fcl(prev, hsize, keep_prob, name='fc%s' % i)
        i += 1
        prev = output
    return output


def make_SB_network(isize=20, in_chnls=19, osize=1,
                    conv_patch=5, pool_patch=2, conv_channels=[64, 128, 256],
                    dense_sizes=[1000, 500, 200],
                    lmbda=0.001, learning_rate=1e-5,
                    seed=123):
    """Create network predicting binding affinity from 3D structure of
    protein-ligand complex.

    Network is composed of block of 3D convolutional layers with max pooling
    and block of fully-connected layers with dropout. Created graph is
    organised with variable scopes as follows:
        input
            structure, affinity
        convolutional
            conv0
                w, b, h
            conv1
                w, b, h
            ...
        fully_connected
            fc0
                w, b, h
            fc1
                w, b, h
            ...
        output
            w, b, prediction
        training
            global_step, mse, L2_cost, cost, optimizer, train
    where `input\structure` and `input\affinity` are placeholders for complex
    structure (shape = (batch_size, isize, isize, isize, in_chnls), you can
    create if with `tfbio.data.make_grid`) and target value (shape =
    (batch_size, osize)), respectively. Training operation (`training/train`)
    minimizes RMSE with L2 penalty using Adam optimizer.

    Parameters
    ----------
    isize: int
        Size of a box with structure
    in_chnls: int, optional
        Number of information channels describing the structure. By default it
        is 19 (default number of features generated by tfbio.data.Featurizer)
    osize: int, optional
        Number of output neurons, default = 1
    conv_patch: int, optional
        Size of convolution patch
    pool_patch: int, optional
            Size of max pooling patch
    conv_channels: array-like, shape = (N,), optional
        Numbers of filters in convolutions.
    dense_sizes: array-like, shape = (M,), optional
            Numbers of neurons in fully-connected layers.
    lmbda: float, optional
        Coefficient for L2 penalty
    learning_rate: float, optional
        Learning rate


    Returns
    -------
    graph: tf.Graph
        Graph with defined network and 4 collections: 'input', 'target',
        'output', and 'kp'
    """

    graph = tf.Graph()

    with graph.as_default():
        np.random.seed(seed)
        tf.set_random_seed(seed)
        with tf.variable_scope('input'):
            x = tf.placeholder(tf.float32,
                               shape=(None, isize, isize, isize, in_chnls),
                               name='structure')
            t = tf.placeholder(tf.float32, shape=(None, osize), name='affinity')

        with tf.variable_scope('convolution'):
            h_convs = convolve3D(x, conv_channels,
                                 conv_patch=conv_patch,
                                 pool_patch=pool_patch)
        hfsize = isize
        for _ in range(len(conv_channels)):
            hfsize = ceil(hfsize / pool_patch)
        hfsize = conv_channels[-1] * hfsize**3

        with tf.variable_scope('fully_connected'):
            h_flat = tf.reshape(h_convs, shape=(-1, hfsize), name='h_flat')

            prob1 = tf.constant(1.0, name='keep_prob_default')
            keep_prob = tf.placeholder_with_default(prob1, shape=(),
                                                    name='keep_prob')

            h_fcl = feedforward(h_flat, dense_sizes, keep_prob=keep_prob)

        with tf.variable_scope('output'):
            w = tf.get_variable('w', shape=(dense_sizes[-1], osize),
                                initializer=tf.truncated_normal_initializer(
                                    stddev=(1 / (dense_sizes[-1]**0.5))))
            b = tf.get_variable('b', shape=(osize,), dtype=tf.float32,
                                initializer=tf.constant_initializer(1))
            y = tf.nn.relu(tf.matmul(h_fcl, w) + b, name='prediction')

        with tf.variable_scope('training'):
            global_step = tf.get_variable('global_step', shape=(),
                                          initializer=tf.constant_initializer(0),
                                          trainable=False)

            mse = tf.reduce_mean(tf.pow((y - t), 2), name='mse')

            with tf.variable_scope('L2_cost'):
                # sum over all weights
                all_weights = [
                    graph.get_tensor_by_name('convolution/conv%s/w:0' % i)
                    for i in range(len(conv_channels))
                ] + [
                    graph.get_tensor_by_name('fully_connected/fc%s/w:0' % i)
                    for i in range(len(dense_sizes))
                ] + [w]

                l2 = lmbda * tf.reduce_sum([tf.reduce_sum(tf.pow(wi, 2))
                                            for wi in all_weights])

            cost = tf.add(mse, l2, name='cost')

            optimizer = tf.train.AdamOptimizer(learning_rate, name='optimizer')
            train = optimizer.minimize(cost, global_step=global_step,
                                       name='train')

    graph.add_to_collection('output', y)
    graph.add_to_collection('input', x)
    graph.add_to_collection('target', t)
    graph.add_to_collection('kp', keep_prob)

    return graph


def custom_summary_histogram(values, num_bins=200):
    """Create custom summary histogram for given values.
    This function returns tf.HistogramProto object which can be used as a value
    for tf.Summary, e.g.:

    >>> summary = tf.Summary()
    >>> summary.value.add(tag='my_summary',
    ...                   histo=custom_summary_histogram(values, num_bins=200))


    Parameters
    ----------
    values: np.ndarray
        Values to summarize
    num_bins: int, optional
        Number of bins in the histogram


    Returns
    -------
    histogram: tf.HistogramProto
        tf.HistogramProto object with a histogram.
    """

    if not isinstance(values, np.ndarray):
        raise TypeError('values must be an array, %s was given'
                        % type(values))
    if values.dtype.kind not in ['f', 'i']:
        raise ValueError('values must be floats, %s was given'
                         % values.dtype)

    if not isinstance(num_bins, int):
        raise TypeError('num_bins must be int, %s was given'
                        % type(num_bins))
    if num_bins <= 0:
        raise ValueError('num_bins must be positive, %s was given'
                         % num_bins)

    flat = values.flatten()
    hist, bins = np.histogram(flat, bins=num_bins)

    bins_middle = (bins[:-1] + bins[1:]) / 2

    histogram = tf.HistogramProto(min=flat.min(), max=flat.max(),
                                  num=len(flat), sum=flat.sum(),
                                  sum_squares=(flat ** 2).sum(),
                                  bucket_limit=bins_middle, bucket=hist)

    return histogram


def custom_summary_image(mpl_figure):
    """Create custom summary image for a given matplotlib.figure.Figure.
    This function returns tf.Summary.Image object which can be used as a value
    for tf.Summary, e.g.:

    >>> summary = tf.Summary()
    >>> summary.value.add(tag='my_summary',
    ...                   image=custom_summary_image(my_figure))


    Parameters
    ----------
    mpl_figure: matplotlib.figure.Figure
        Figure to convert to tf.Summary.Image object


    Returns
    -------
    image: tf.Summary.Image
        tf.Summary.Image object with a figure.
    """

    if not isinstance(mpl_figure, plt.Figure):
        raise TypeError('mpl_figure must be matplotlib.figure.Figure object,'
                        '%s was given' % type(mpl_figure))

    imgdata = io.BytesIO()
    mpl_figure.savefig(imgdata, format='png')
    imgdata.seek(0)

    width, height = mpl_figure.canvas.get_width_height()

    image = tf.Summary.Image(height=height, width=width, colorspace=3,
                             encoded_image_string=imgdata.getvalue())
    imgdata.close()

    return image


def feature_importance_plot(values, labels=None):
    """Create summary image with bar plot of feature importance.

    Parameters
    ----------
    values: array-like, shape = (N,)
        Values to plot
    labels: array-like, shape = (N,), optional
        Labels associated with the values. If not given, "F1", "F2", etc are
        used as labels.

    Returns
    -------
    image: tf.Summary.Image
        tf.Summary.Image object with a bar plot.
    """

    if not isinstance(values, (list, tuple, np.ndarray)):
        raise TypeError('values must be a 1D sequence')

    try:
        values = np.asarray(values, dtype='float')
    except:
        raise ValueError('values must be a 1D sequence of numbers')

    if values.ndim != 1:
        raise ValueError('values must be a 1D sequence of numbers')

    if labels is None:
        labels = ['F%s' % i for i in range(len(values))]
    elif not isinstance(labels, (list, tuple, np.ndarray)):
        raise TypeError('labels must be a 1D sequence')

    if len(values) != len(labels):
        raise ValueError('values and labels must have equal lengths')

    fig, ax = plt.subplots(figsize=(3, 3))
    sns.barplot(y=labels, x=values, ax=ax)
    fig.tight_layout()

    image = custom_summary_image(fig)
    plt.close(fig)

    return image


def make_summaries_SB(graph, feature_labels=None):
    """Create summaries for network created with `make_SB_network`"""

    with graph.as_default():
        with tf.variable_scope('net_properties'):
            # weights between input and the first layer
            wconv0 = graph.get_tensor_by_name('convolution/conv0/w:0')
            in_chnls = wconv0.shape[-2].value
            if feature_labels is None:
                feature_labels = ['F%s' % i for i in range(in_chnls)]
            else:
                assert in_chnls == len(feature_labels)
            feature_weights = tf.split(wconv0, in_chnls, axis=3)
            feature_importance = tf.reduce_sum(tf.abs(wconv0),
                                               reduction_indices=[0, 1, 2, 4],
                                               name='feature_importance')

        net_summaries = tf.summary.merge((
            tf.summary.histogram('weights', wconv0),
            *(tf.summary.histogram('weights_%s' % name, value)
              for name, value in zip(feature_labels, feature_weights)),
            tf.summary.histogram('predictions', graph.get_tensor_by_name('output/prediction:0'))
        ))

        training_summaries = tf.summary.merge((
            tf.summary.scalar('mse', graph.get_tensor_by_name('training/mse:0')),
            tf.summary.scalar('cost', graph.get_tensor_by_name('training/cost:0'))
        ))

    return net_summaries, training_summaries


class SummaryWriter():

    def __init__(self, *args, **kwargs):
        """Context manager for tf.summary.FileWriter"""
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):
        self.writer = tf.summary.FileWriter(*self.args, **self.kwargs)
        return self.writer

    def __exit__(self, *args):
        self.writer.close()
