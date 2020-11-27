import warnings
import torch


class Default(object):
    load_model_path = None  # load model path

    pretrain_model_path = './data/imagenet-vgg-f.mat'

    # visualization
    vis_env = 'main'  # visdom env
    vis_port = 8097  # visdom port

    # for flickr25k dataset
    dataset = 'flickr25k'
    data_path = './data/FLICKR-25K.mat'
    db_size = 18015
    num_label = 24
    query_size = 2000
    text_dim = 1386
    training_size = 10000

#     # # # for nus-wide dataset
#     dataset = 'nus-wide'
#     # # data_path = './data/NUS-WIDE-TC10.mat'
#     data_path = './data/NUS-WIDE-TC21.mat'
#     db_size = 193734
#     # # db_size = 184457
#     num_label = 21
#     # # num_label = 10
#     query_size = 2100
#     text_dim = 1000
#     training_size = 10000

    batch_size = 128
    image_dim = 4096
    hidden_dim = 8192
    modals = 2
    valid = True  # whether to use validation
    valid_freq = 1
    max_epoch = 300

    output_dim = 64  # hash code length
    lr = 0.0001  # initial learning rate

    device = 'cuda:0'

    # hyper-parameters
    alpha = 1
    gamma = 0.01
    beta = 1
    mu = 0.01
    delta = 1
    theta = 1

    margin = 0.4
    def parse(self, kwargs):
    """
    update configuration by kwargs.
    """
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Waning: opt has no attribute %s" % k)
        setattr(self, k, v)

    print('Configuration:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__') and str(k) != 'parse':
                print('\t{0}: {1}'.format(k, getattr(self, k)))



opt = Default()
