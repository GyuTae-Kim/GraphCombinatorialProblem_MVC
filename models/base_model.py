import tensorflow as tf
from tensorflow.keras import Model

import os


class BaseModel(Model):

    def __init__(self, config):
        super(BaseModel, self).__init__()

        self.config = config

        self.load = config['model_params']['load']
        self.t = config['model_params']['t']
        self.p = config['model_params']['p']
        self.init_lr = config['model_params']['init_lr']
        self.save_path = config['model_params']['save_path']
        self.decay_steps = config['model_params']['decay_steps']
        self.decay_rate = config['model_params']['decay_rate']
        self.grad_clip = config['model_params']['grad_clip']

        self.G, self.node_list, self.adj, self.feature = None, None, None, None
        self.checkpoint_format = os.path.join(self.save_path, "{type}-{epoch:04d}.ckpt")

    def call(self, x, mu, weights, adj):
        pass

    def load_network(self):
        if not os.path.exits(self.save_path):
            os.mkdir(self.save_path)
            print(" [*] Couldn't find checkpoint.")
            return
        
        lastest = tf.train.latest_checkpoint(self.save_path)
        if lastest is None:
            print(" [*] Couldn't find checkpoint.")
            return
        
        print(' [*] Load network from {}'.format(lastest))
        self.load_weights(lastest)

    def save_network(self, ep, _type='cp'):
        checkpoint_format = self.checkpoint_format.format(type=_type, epoch=ep)
        self.save_weights(checkpoint_format)
        print(' [*] Save network: {}'.format(checkpoint_format))
