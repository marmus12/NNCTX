# NNCTX

Code for MMSP 2021 paper "Neural Network Modeling of Probabilities for Coding the Octree Representation of Point Clouds"
##########################################
Run main_enc_dec.py to test encoding and decoding with trained models (in trained_models/ folder).

To train your own models, 
First run collect_train_data.py (once for collecting training data, once for collecting validation data)
Then run train_tf.py by specifying train_data_dir & val_data_dir 
