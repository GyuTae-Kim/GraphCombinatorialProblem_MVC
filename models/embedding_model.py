import tensorflow as tf
import numpy as np

from .networks import Structure2Vec, Evaluation
from .base_model import BaseModel


class EmbeddingModel(BaseModel):

    def __init__(self, config):
        super(BaseModel, self).__init__(config)

        self.s2v = Structure2Vec(self.p)
        self.eval = Evaluation(self.p)
        self.global_step = tf.Variable(0, trainable=False)
        lr_decay = tf.compat.v1.train.exponential_decay(self.init_lr,
                                                        self.global_step,
                                                        self.decay_steps,
                                                        self.decay_rate,
                                                        staircase=True)
        self.opt = tf.compat.v1.train.AdamOptimizer(lr_decay)
        print(' [*] Created network.')

        if self.load:
            self.load_network()
            print(' [*] Loaded network.')
    
    def set_instance(self, G):
        if G is None:
            ValueError
        
        self.G = G
        self.node_list = G.node_list
        self.w = tf.convert_to_tensor(self.G.weights)
        self.adj = tf.convert_to_tensor(self.G.adj)
        self.mu = tf.zeros((len(self.node_list), self.p), dtype=tf.float32)

    def gen_input(self):
        return self.G.x, self.mu, self.w, self.mu
        
    def embedding(self, x=None, mu=None, w=None, adj=None):
        if x is None:
            x, mu, w, adj = self.gen_input()
        else:
            x = tf.convert_to_tensor(x, dtype=tf.float32)
            mu = tf.convert_to_tensor(mu, dtype=tf.float32)
            w = tf.convert_to_tensor(w, dtype=tf.float32)
            adj = tf.convert_to_tensor(adj, dtype=tf.float32)
        
        for t in range(self.t):
            mu = self.s2v(x, mu, w, adj)
        
        return mu

    def evaluate(self, idx, mu):
        sum_mu = tf.reduce_sum(mu, axis=0, keepdims=True)
        sum_mu *= tf.ones((len(idx), 1), dtype=tf.float32)
        node_mu = specific_value(mu, idx)
        Q = self.eval(sum_mu, node_mu)

        return Q

    def call(self, idx, x, mu, w, adj):
        mu = self.embedding(x, mu, w, adj)
        Q = self.evaluate(idx, mu)

        return Q
    
    def update(self, idx, x, mu, w, adj, opt_Q):
        with tf.GradientTape() as tape:
            Q = self.__call__(idx, x, mu, w, adj)
            loss = tf.keras.losses.mean_squared_error(opt_Q, Q)
        tvars = self.trainable_variables
        grads = tape.gradient(loss, tvars)
        grads, _ = tf.clip_by_global_norm(grads, self.grad_clip)
        self.opt.apply_gradients(zip(grads, tvars), self.global_step)
        self.global_step.assign_add(1)

        return loss


def specific_value(mu, idx):
    len_idx = len(idx)
    brod = tf.ones((len_idx, *mu.shape), dtype=tf.float32)
    brod_mu = tf.expand_dims(mu, axis=0) * brod
    h = np.zeros_like(brod_mu, dtype=np.float32)
    h[np.arange(len_idx), idx, :] = 1.
    h = tf.convert_to_tensor(h, dtype=tf.float32)
    s_val = brod_mu * h
    s_val = tf.reduce_sum(s_val, axis=1)

    return s_val
