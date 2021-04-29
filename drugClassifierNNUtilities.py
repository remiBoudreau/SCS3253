# Filename: drugClassifierNN
# Dependencies: drugClassifierNNUtilities, tensorflow, six.moves, pandas, matplotlib, numpy, random, sklearn, datetime
# Author: Jean-Michel Boudreau
# Date: April 29, 2019

'''
Contains all utility functions required by drugClassifierNN.py

'''

### Import dependencies ###
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from random import sample
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

'''
Retrieves entire ZINC12 database (as a tsv file) from URL 
'''

def fetch_zinc_data(zinc_url=
                    "http://zinc.docking.org/db/bysubset/6/6_prop.xls"):
    file_name = "zinc_db.xls"
    urllib.request.urlretrieve(zinc_url, file_name)

'''
Loads the first million rowss of the ZINC12 database and returns data as a 
pandas dataframe
'''
def load_zinc_data():
    file_name = "zinc_db.xls"
    # tsv too large to hold  in memory; get first 1000000 rows of tsv;   
    # this size is sufficient for creating drug and nondrug data subsets
    data = pd.read_csv(file_name, sep='\t', nrows=1000000)
    return data

'''
Recives ZINC data (with type as pandas dataframe) as input. Inspects the data
through various pandas dataframe functions.

'''
def inspect_data(data):
    data.head()
    data.info()
    data.describe()
    data.hist(bins=10, figsize=(20,15))
    plt.show() 

'''
Recieves ZINC data (with type as pandas dataframe) as input. Applies filter 
for "drug-likeness" according to Lipinsky's rule of 5. Returns vector of booleans
correspdonding to whether each element in the column meets the criteria.
'''
def Lipinsky_ro5(data):
    MWT1 = data['MWT']  <= 500 # molecular mass lower than 500 daltons
    LogP = data['LogP'] <= 5   # octonal-water partition coefficient not > 5
    HBD  = data['HBD']  <= 5   # no more than 5 hydrogen bond donors
    HBA  = data['HBA']  <= 10  # no more than 10 hydrogen bond acceptors
    return MWT1, LogP, HBD, HBA

'''
Recieves ZINC data (with type as pandas dataframe) as input. Filters for drug-
likeness according to Lipinsky's rule of 5 and seperates 'drug-like' molecules
from non drug-like molecules into seperate dataframes. Returns 2 seperate 
dataframes containing drug-like molecules and non drug-like molecules in that
order.
'''    
def Lipinsky_drug_classifier_v1(data):
    # Get booleans for each of Lipisnky's rule of 5's condition
    MWT1, LogP, HBD, HBA = Lipinsky_ro5(data)
    # if conditions are all met, instance of data copied to dataframe
    # containing drug-like molecules. Otherwise, instance is copied to 
    # dataframe containing non drug-like molecules.
    drug_data = data[MWT1 & LogP & HBD & HBA].drop(['Desolv_apolar', 
                                                    'Desolv_polar',
                                                    'Charge', 
                                                    'tPSA',
                                                    'NRB'], axis=1)
    non_drug_data = data[~(MWT1 & LogP & HBD & HBA)].drop(['Desolv_apolar', 
                                                    'Desolv_polar',
                                                    'Charge', 
                                                    'tPSA',
                                                    'NRB'], axis=1)
    return drug_data, non_drug_data

'''
Recieves ZINC data (with type as pandas dataframe) as input. Filters for drug-
likeness according to Lipinsky's rule of 5 alongside 3 additional conditions &
seperates 'drug-like' molecules from non drug-like molecules into seperate 
dataframes. Returns 2 seperate dataframes containing drug-like molecules and 
non drug-like molecules in that order.
'''
def Lipinsky_drug_classifier_v2(data):
    # Get booleans for each of Lipisnky's rule of 5's condition
    MWT1, LogP, HBD, HBA = Lipinsky_ro5(data)
    # Get booleans for 3 additional conditions listed on ZINC db for
    # drug-likeness
    MWT2 = data['MWT'] >= 150  
    NRB  = data['NRB'] <= 7    
    TPSA = data['tPSA'] < 150
    # if conditions are all met, instance of data copied to dataframe
    # containing drug-like molecules. Otherwise, instance is copied to 
    # dataframe containing non drug-like molecules.
    drug_data = data[MWT1 & MWT2 & LogP & NRB & TPSA & HBD & HBA]
    non_drug_data = data[~(MWT1 & MWT2 & LogP & NRB & TPSA & HBD & HBA)]
    return drug_data, non_drug_data

'''
Recieves the "drug"/"nondrug" data subsets (with type as pandas dataframe), 
number of molecuiles to be sampled and DNN model version as input. v1 
corresponds to the model with Lipinsky's rule of 5 and v2 corresponds to the
model that improves upon Lipinsky's rule of 5. Scales all features using 
sklearn's MinMaxScaler and randomly samples 50,000 drug-like and non-drug 
molecules from each subset, respectively. Furthermore, takes 10,000 from each 
each subset to use for testing and another 5,000 from each subset to use for 
validation purposes when training the DNN. Returns training and validation 
subsets when called and additionally returns testing subset when called with 
with version =/= 1

'''
def process_data(drug_data, non_drug_data, n_mol, version):    
    # Randomly sample n_mol/2 molecules for the drug/non drug data subsets and 
    # store as the variables passed as arguments
    n_samples = int(n_mol/2) 
    drug_data = drug_data.sample(n=n_samples, random_state=42)
    non_drug_data = non_drug_data.sample(n=n_samples, random_state=42)

    # Concatenate dataframes and normalize features
    all_data = pd.concat([drug_data, non_drug_data])
    scaler = MinMaxScaler()
    if version == 1:
        feature_list = ['MWT',
                       'LogP',
                       'HBD',
                       'HBA']
    else:
        feature_list = ['MWT',
                       'LogP',
                       'Desolv_apolar',
                       'Desolv_polar',
                       'HBD',
                       'HBA',
                       'tPSA',
                       'Charge',
                       'NRB']
    all_data[feature_list] = scaler.fit_transform(all_data[feature_list])
    
    # Seperate dataset for further processing
    drug_data = all_data.iloc[:50000, :]
    non_drug_data = all_data.iloc[50000:, :]
    
    # Label all data as either drug-like or not with 1 or 0 respectively
    drug_data['DRUG'] = 1
    non_drug_data['DRUG'] = 0
    
    # Randomly sample integers from 0 to n_mol/2 0.1*n_mol times for test set 
    # indices; Remaining indices are for training/validation sets. 
    # Gives 80/20 split for (training & validation)/test sets
    n_test = int(0.1*n_mol)
    molIn = range(n_samples)
    testIn = sample(molIn, n_test)
    trainIn = list(set(molIn) - set(testIn))
    
    # Create test/training subsets
    drug_data_test = drug_data.iloc[testIn]
    drug_data_train = drug_data.iloc[trainIn]
    non_drug_data_test = non_drug_data.iloc[testIn]
    non_drug_data_train = non_drug_data.iloc[trainIn]
    
    # Concatenate the training/testing subsets
    all_data_test = pd.concat([drug_data_test, non_drug_data_test])
    all_data_train = pd.concat([drug_data_train, non_drug_data_train])
    
    # Shuffle Training Data to avoid bias
    all_data_train = all_data_train.reindex(np.random.permutation(
            all_data_train.index))
    
    # Create numpy array of features for passing into DNN
    X_train = all_data_train[feature_list].values 
    X_test =  all_data_test[feature_list].values
    # # Create numpy array of labels for passing into DNN
    y_train = all_data_train['DRUG'].values
    y_test = all_data_test['DRUG'].values
    
    # Split the training data into validation and training datasets
    X_valid, X_train = X_train[:5000], X_train[5000:]
    y_valid, y_train = y_train[:5000], y_train[5000:]
    
    if version == 1:
        return X_train, X_valid, y_train, y_valid
    else: 
        return X_train, X_valid, X_test, y_train, y_valid, y_test
'''   
To make this notebook's output stable across runs
'''
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

'''
Shuffles elements of each batch prior to training with the DNN with the batch
'''
def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch
'''
Recieves number of features (n_inputs), number of nodes in each hidden layer
(n_hidden1, n_hidden2), number of outputs (n_outputs), model-type (version), 
training data (X-train, y_train), validation data (X_valid, y_valid), and test 
data (X_test, y_test). Trains and tests a DNN using all the above. Returns a 
vector containing the predicted classes ("drug-like" or not) of the test set. 
'''
def DNN(n_inputs, n_hidden1, n_hidden2, n_outputs, version, 
              X_train, X_valid, X_test, y_train, y_valid, y_test):    
    # Reset graph
    reset_graph()
    # Define batch size for batch
    batch_size = 50

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int32, shape=(None), name="y") 
    
    with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1",
                                  activation=tf.nn.elu)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2",
                                  activation=tf.nn.elu)
        logits = tf.layers.dense(hidden2, n_outputs, name="outputs")
        y_proba = tf.nn.softmax(logits)
    
    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")
    
    learning_rate = 0.01
    
    with tf.name_scope("train"):
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
    										   momentum=0.9, use_nesterov=True)
        training_op = optimizer.minimize(loss)
    
    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    n_epochs = 100
    
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
            print(epoch, "Batch accuracy:", acc_batch, "Validation accuracy:", acc_valid)
        
        save_path = "./lip_model_v" + str(version) + ".ckpt"
        save_path = saver.save(sess, save_path)

    with tf.Session() as sess:
        saver.restore(sess, save_path)
        Z = logits.eval(feed_dict={X: X_test},session=sess)
        y_pred = np.argmax(Z, axis=1)
        return y_pred
    