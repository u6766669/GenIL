import tensorflow as tf
from tf_commons.ops import *
import numpy as np
from tqdm import tqdm

from dataset import gen_traj,TrajDataset, batch_dataset



class MyModel(object):
    def __init__(self,include_action,ob_dim,ac_dim,batch_size=64,num_layers=2,embedding_dims=256):
        self.include_action = include_action
        in_dims = ob_dim+ac_dim if include_action else ob_dim

        self.inp = tf.placeholder(tf.float32,[None,in_dims])
        self.x = tf.placeholder(tf.float32,[64,None,in_dims])
        self.y = tf.placeholder(tf.float32,[64,None,in_dims])
        # [0 when x is better 1 when y is better]
        self.l = tf.placeholder(tf.int32,[batch_size])
        self.l2_reg = tf.placeholder(tf.float32,[])

        with tf.variable_scope('weights') as param_scope:
            self.fcs = []
            last_dims = in_dims
            for l in range(num_layers):
                self.fcs.append(Linear('fc%d'%(l+1),last_dims,embedding_dims))
                last_dims = embedding_dims
            self.fcs.append(Linear('fc%d'%(num_layers+1),last_dims,1))

        self.param_scope = param_scope

        # build graph
        def _reward(x):
            for fc in self.fcs[:-1]:
                x = tf.nn.relu(fc(x))
            r = tf.squeeze(self.fcs[-1](x),axis=1)
            return x, r

        self.fv, self.r = _reward(self.inp)

        v_x = []
        for i in range(self.x.shape[0]):
            _, r_x = _reward(self.x[i])
            v_x.append(tf.reduce_sum(r_x))
        self.v_x = v_x

        v_y = []
        for j in range(self.y.shape[0]):
            _, r_y = _reward(self.y[j])
            v_y.append(tf.reduce_sum(r_y))
        self.v_y = v_y

        logits = tf.stack([self.v_x,self.v_y],axis=1) #[None,2]
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=self.l)
        self.loss = tf.reduce_mean(loss,axis=0)

        weight_decay = 0.
        for fc in self.fcs:
            weight_decay += tf.reduce_sum(fc.w**2)

        self.l2_loss = self.l2_reg * weight_decay

        pred = tf.cast(tf.greater(self.v_y,self.v_x),tf.int32)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(pred,self.l),tf.float32))

        self.optim = tf.train.AdamOptimizer(1e-4)
        self.update_op = self.optim.minimize(self.loss+self.l2_loss,var_list=self.parameters(train=True))

        self.saver = tf.train.Saver(var_list=self.parameters(train=False),max_to_keep=0)

    def parameters(self,train=False):
        if train:
            return tf.trainable_variables(self.param_scope.name)
        else:
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,self.param_scope.name)

    def train(self,trajdataset,batch_size=64,iter=3000,l2_reg=0.01,output_size=5000,step_size=100,debug=False):

        sess = tf.get_default_session()
        output,label = trajdataset.generate_joint_input(output_size=output_size,step_size=step_size)
        trainset, validset = trajdataset.output_train_valid(output,label)

        for it in tqdm(range(iter),dynamic_ncols=True):
            batch = batch_dataset(dataset=trainset,batch_size=batch_size)
            op, b_l = zip(*batch)
            b_x,b_y = zip(*op)
            loss,l2_loss,acc,_ = sess.run([self.loss,self.l2_loss,self.acc,self.update_op],feed_dict={
                self.x:b_x,
                self.y:b_y,
                self.l:b_l,
                self.l2_reg:l2_reg,
            })

            if debug:
                if it % 100 == 0 or it < 10:
                    batch = batch_dataset(dataset=validset, batch_size=batch_size)
                    op, b_l = zip(*batch)
                    b_x, b_y = zip(*op)
                    valid_acc = sess.run(self.acc,feed_dict={
                        self.x:b_x,
                        self.y:b_y,
                        self.l:b_l
                    })
                    tqdm.write(('loss: %f (l2_loss: %f), acc: %f, valid_acc: %f'%(loss,l2_loss,acc,valid_acc)))



    def get_reward(self,obs,acs,batch_size=1024):
        sess = tf.get_default_session()

        if self.include_action:
            inp = np.concatenate((obs,acs),axis=1)
        else:
            inp = obs

        b_r = []
        for i in range(0,len(obs),batch_size):
            r = sess.run(self.r,feed_dict={
                self.inp:inp[i:i+batch_size]
            })

            b_r.append(r)

        return np.concatenate(b_r,axis=0)


