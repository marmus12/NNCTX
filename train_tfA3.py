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
from usefuls import setdiff,find_common_rows,find_diff_rows,find_diff_rows_with_ind,plt_imshow
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
# restore = True
# if restore:
    
num_epochs = 30000
batch_size = 10000
learning_rate = 0.001
# lambda_wr = 0
from models import tfModel10A as mymodel


minprob= 0.0001#0.01
ctx_type=100
#%%
val_data_dir = '/media/emre/Data/DATA/45A_a1s1_align0/'

train_data_dirs = ['/media/emre/Data/DATA/45A_ads6_ls9_align0/',
                   '/media/emre/Data/DATA/45A_ads6_ls9_align1/',
                   '/media/emre/Data/DATA/45A_ads6_ls9_align2/',
                   '/media/emre/Data/DATA/45A_ads6_ls9_align3/',
                   '/media/emre/Data/DATA/45A_ads6_ls9_align4/',
                   '/media/emre/Data/DATA/45A_ads6_ls9_align5/',
                   '/media/emre/Data/DATA/45A_ads6_ls9_align6/',
                   '/media/emre/Data/DATA/45A_ads6_ls9_align7/']



bs1 = batch_size//8
# val_data_dir = '/media/emre/Data/DATA/ricardo1_soldier1_100_minco_1/'
           #     '/home/emre/Documents/DATA/soldier_1_122/']
train_dss = []
train_indss = []
disc_inds = np.zeros((8,),'int')     
for ia in range(8):
    train_dss.append(ctx_dataset2(train_data_dirs[ia],ctx_type))
    tot_counts = np.sum(train_dss[ia].counts,1)
    disc_inds[ia] = np.argmax(tot_counts)
    train_indss.append( list(range(disc_inds[ia] ))+list(range(disc_inds[ia] +1,train_dss[ia].n_ctxs)))#list(range(train_ds.n_ctxs))



val_ds = ctx_dataset2(val_data_dir,ctx_type)


#%% DISCARD THE MOST FREQUENT CONTEXT FROM TRAINING SET; SINCE COUNTS ARE SO HIGH

###
vtot_counts = np.sum(val_ds.counts,1)
vdisc_ind = np.argmax(tot_counts)



val_inds = list(range(vdisc_ind))+list(range(vdisc_ind+1,val_ds.n_ctxs))#range(val_ds.n_ctxs)


#%%

mdl= mymodel(ctx_type)


curr_date = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = '/home/emre/Documents/train_logs/' + curr_date + '/'


os.mkdir(logdir)

checkpoint_path = logdir +'checkpoint.npy'

curr_file = inspect.getfile(inspect.currentframe()) # script filename (usually with path)
copyfile(curr_file,logdir + curr_date + "__" + curr_file.split("/")[-1])    
 



tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)



tfcounts = tf1.placeholder(dtype='float',shape = [batch_size,2])


losses = []
# loss = -tf1.reduce_sum(tfcounts[:,0]*tf1.log(mdl.output[:,0])+tfcounts[:,1]*tf1.log(mdl.output[:,1]))

for ia in range(8):
    a_start = ia*bs1
    a_end = (ia+1)*bs1
    losses.append(-tf1.reduce_sum( tfcounts[a_start:a_end,0]*tf1.log(mdl.outputs[a_start:a_end,2*ia]+minprob)
                                  + tfcounts[a_start:a_end,1]*tf1.log(mdl.outputs[a_start:a_end,2*ia+1]+minprob) ) / tf1.reduce_sum(tfcounts[a_start:a_end,1]) )

Tloss = tf1.reduce_sum(losses)

opti = tf1.train.AdamOptimizer(learning_rate=learning_rate)

step = tf.Variable(0, dtype=tf.int64)
step_update = step.assign_add(1)

# tr_loss_summs,train_writers,train_writer_flushs,train_ops = [],[],[],[]
# for ia in range(8):
    # train_ops.append( opti.minimize(losses[ia]))
    # train_writers.append( tf.summary.create_file_writer(logdir+ 'train'+str(ia)) )

    # with train_writers[ia].as_default():
    #   tr_loss_summs.append(tf.summary.scalar("train_loss"+str(ia), losses[ia], step=step))

    # train_writer_flushs.append( train_writers[ia].flush())

train_op = opti.minimize(Tloss)

train_writer = tf.summary.create_file_writer(logdir+ 'train')
with train_writer.as_default():
  tr_loss_summ = tf.summary.scalar("train_loss", Tloss, step=step)
train_writer_flush = train_writer.flush()


#%%
val_writer = tf.summary.create_file_writer(logdir+ 'val')  
with val_writer.as_default():  
    val_loss_summ = tf.summary.scalar("val_loss", losses[0], step=step)
      
all_summary_ops = tf1.summary.all_v2_summary_ops()
val_writer_flush = val_writer.flush()

#%%
sess = tf1.Session()
sess.run(tf1.global_variables_initializer())

# for ia in range(8):
#     sess.run([train_writers[ia].init(), step.initializer])
sess.run(train_writer.init())
sess.run(val_writer.init())



best_val_loss = 100000000
prev_tr_loss = 10000000

num_batchess = np.zeros((8,),int)
for ia in range(8):
    num_batchess[ia] = len(train_indss[ia])//bs1


trctxs = np.zeros((batch_size,ctx_type),dtype='bool')
trcounts = np.zeros((batch_size,2),dtype='int')
num_batches = np.min(num_batchess)
done=0
for epoch in range(num_epochs):
    
    for ia in range(8):
        np.random.shuffle(train_indss[ia])
    
    print('epoch:' + str(epoch))
    
    for ibatch in range(num_batches):    
        for ia in range(8):
            # print('alignment:'+ str(ia))
            batch_inds = train_indss[ia][ibatch*bs1:(ibatch+1)*bs1]
            a_start = ia*bs1
            a_end = (ia+1)*bs1
            trctxs[a_start:a_end] = train_dss[ia].ctxs[batch_inds,:]
            trcounts[a_start:a_end] = train_dss[ia].counts[batch_inds,:]
            
                   
        sess.run([train_op],feed_dict = {mdl.input:trctxs,tfcounts:trcounts})
        tr_loss = sess.run([Tloss,tr_loss_summ],feed_dict = {mdl.input:trctxs,tfcounts:trcounts})   
        sess.run(step_update)
        sess.run(train_writer_flush)
        print('tr_loss:'+str(tr_loss))

    #%%# VALIDATION:
    batch_inds = random.sample(val_inds,batch_size)
    vctxs = val_ds.ctxs[batch_inds,:]
    vcounts = val_ds.counts[batch_inds,:]      
    
    val_loss,_ = sess.run([losses[0],val_loss_summ],feed_dict = {mdl.input:vctxs,tfcounts:vcounts})
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
















