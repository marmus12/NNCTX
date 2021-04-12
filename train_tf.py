# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os, inspect
from shutil import copyfile
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import random
from datetime import datetime
from tensorflow.keras.layers import Conv2D
from scipy.io import loadmat
from usefuls import setdiff,find_common_rows,find_diff_rows,find_diff_rows_with_ind
#import plyfile
from pcloud_functs import collect_blocks, collect_counts,ctxbits2block
from train_utils import CL_criterion

import open3d as o3d
#from models import MyModel4,MyModel5,MyModel6

from dataset import ctx_dataset, ctx_dataset2

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

#%%config
num_epochs = 30000
batch_size = 200000


from models import tfModel10 as mymodel



ctx_type=100
# train_data_dir = ['/home/emre/Documents/DATA/andrew_david_sarah_6_122/', 
#                   '/home/emre/Documents/DATA/longdress_18_122/']
# train_data_dir = '/media/emre/Data/DATA/ads6_longdress18_122/'
val_data_dir = '/media/emre/Data/DATA/a1_sol1_100/'
train_data_dir = '/media/emre/Data/DATA/ads6_ld9_sol9_100/'
# val_data_dir = '/media/emre/Data/DATA/ricardo1_soldier1_100_minco_1/'
           #     '/home/emre/Documents/DATA/soldier_1_122/']
                

train_ds = ctx_dataset2(train_data_dir,ctx_type)

val_ds = ctx_dataset2(val_data_dir,ctx_type)



#%%
mdl= mymodel(ctx_type)


#%%
curr_date = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = '/home/emre/Documents/train_logs/' + curr_date + '/'


os.mkdir(logdir)

checkpoint_path = logdir +'checkpoint.npy'

curr_file = inspect.getfile(inspect.currentframe()) # script filename (usually with path)
copyfile(curr_file,logdir + curr_date + "__" + curr_file.split("/")[-1])    
 



tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)




#%%


#%%
#loss_fn = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
# loss_fn = CL_criterion

# CL=kb.sum(-y_true[:,0]*kb.log(y_pred[:,0])/kb.log(2.)-y_true[:,1]*kb.log(y_pred[:,1])/kb.log(2.))



tfcounts = tf1.placeholder(dtype='float',shape = [None,2])


loss = -tf1.reduce_sum(tfcounts[:,0]*tf1.log(mdl.output[:,0])+tfcounts[:,1]*tf1.log(mdl.output[:,1]))


opti = tf1.train.AdamOptimizer()

train_op = opti.minimize(loss)



step = tf.Variable(0, dtype=tf.int64)
step_update = step.assign_add(1)
train_writer = tf.summary.create_file_writer(logdir+ 'train')
val_writer = tf.summary.create_file_writer(logdir+ 'val')
with train_writer.as_default():
  tr_loss_summ = tf.summary.scalar("train_loss", loss, step=step)
with val_writer.as_default():  
  val_loss_summ = tf.summary.scalar("val_loss", loss, step=step)
all_summary_ops = tf1.summary.all_v2_summary_ops()
train_writer_flush = train_writer.flush()
val_writer_flush = val_writer.flush()


sess = tf1.Session()

sess.run(tf1.global_variables_initializer())

sess.run([train_writer.init(),val_writer.init(), step.initializer])

train_inds = list(range(train_ds.n_ctxs))
val_inds = range(val_ds.n_ctxs)

best_val_loss = 100000000
for epoch in range(num_epochs):
    
    print('epoch:' + str(epoch))
    np.random.shuffle(train_inds)

    num_batches = np.ceil(train_ds.n_ctxs/batch_size).astype('int')
    
    
    for ibatch in range(num_batches):
    
        batch_inds = train_inds[ibatch*batch_size:(ibatch+1)*batch_size]
        trctxs = train_ds.ctxs[batch_inds,:]
        trcounts = train_ds.counts[batch_inds,:]
        
        sess.run([train_op],feed_dict = {mdl.input:trctxs,tfcounts:trcounts})
    
    tr_loss,_= sess.run([loss,tr_loss_summ],feed_dict = {mdl.input:trctxs,tfcounts:trcounts})   
    sess.run(step_update)
    sess.run(train_writer_flush)
    print('tr_loss:'+str(tr_loss))
    
    #%%# VALIDATION:
    batch_inds = random.sample(val_inds,batch_size)
    vctxs = val_ds.ctxs[batch_inds,:]
    vcounts = val_ds.counts[batch_inds,:]      
    
    val_loss,_ = sess.run([loss,val_loss_summ],feed_dict = {mdl.input:vctxs,tfcounts:vcounts})
    sess.run(val_writer_flush)
    print('val_loss:'+str(val_loss))

    if val_loss<best_val_loss:
        print('saving checkpoint..')
        np.save(checkpoint_path,sess.run(tf1.trainable_variables()))
        best_val_loss = val_loss





# with tf.compat.v1.Session(graph=g) as sess:
#   sess.run([writer.init(), step.initializer])

#   for i in range(100):
#     sess.run(all_summary_ops)
#     sess.run(step_update)
#     sess.run(writer_flush)
















