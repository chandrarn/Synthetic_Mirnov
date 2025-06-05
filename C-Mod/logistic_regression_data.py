#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 20:19:25 2025
    Initial Logistic regresion attempt
    read in standardized filename or directory, generate target n# set
    calculate standardized phase differences as ``data''
        - need more than one n# ``data'' case per n#:
            - otherwise training set will include unseen category (n)
    
    autoinclude comparison to real shot : separately
    - how to deal with m#
@author: rian
"""

# Load libraries 
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler  # for feature scaling
from sklearn.model_selection import train_test_split  # for train/test split

from scipy.signal import hilbert
import glob
import json
from sys import path; path.append('../signal_generation/')
from header_signal_generation import histfile


def run_regression(data_directory='../data_output/training_data/',):
    
    # Prep data
    # Prepare data
    X,y = load_processed_phase_Data(data_directory)
    
    
    n_samples, n_features = X.shape
    print(f'number of samples: {n_samples}, number of features: {n_features}')
    
    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    
    # scale data
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    #####################################################
    # convert to tensors
    X_train = torch.from_numpy(X_train.astype(np.float32))
    X_test = torch.from_numpy(X_test.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32))
    y_test = torch.from_numpy(y_test.astype(np.float32))
    
    # reshape y tensors
    y_train = y_train.view(y_train.shape[0], 2)
    y_test = y_test.view(y_test.shape[0], 2)
    

        
    model = LogisticRegression(n_features)
    
    ###############################################
    # Loss and optimizer 
    # Loss and optimizer
    learning_rate = 0.01
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    
    # Learning  loop
    # training loop
    num_epochs = 100
    
    for epoch in range(num_epochs):
        # forward pass and loss
        y_predicted = model(X_train)
        print(y_predicted)
        loss = criterion(y_predicted, y_train)
        
        # backward pass
        loss.backward()
        
        # updates
        optimizer.step()
        
        # zero gradients
        optimizer.zero_grad() #Unclear purpose/interpretation of "zero the gradients"
        
        if (epoch+1) % 10 == 0:
            print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
       
    
    ############################################
            
    with torch.no_grad():
        y_predicted = model(X_test)  # no need to call model.forward()
        y_predicted_cls = y_predicted.round()   # round off to nearest class
        acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])  # accuracy
        print(f'accuracy = {acc:.4f}')

######################################################
def load_processed_phase_Data(data_directory):

#    # Load in sensor locations [UNNECESSARY FOR ML SIDE]
    
    # Load in m/n information
    
    with open(data_directory+'Simulation_Params.json','r') as f:
        simulation_params = json.load(f)
        # Assume stores m,n,f,etc
        
    # Loop over files in folder matching save name pattern
    
    target = []
    features =[]
    for fName in __gen_F_names(data_directory):
        phases = []
        fName = fName[len(data_directory):]
        hist_file = histfile(data_directory+fName)
        for sig in hist_file:
            if sig == 'time':continue
            phases.append(np.unwrap(np.angle(hilbert(hist_file[sig]))))
        phases = np.array(phases)
        phases = np.mean(phases - phases[0],axis=1)[1:]
        features.append(phases)
        target.append([simulation_params[fName]['m'], simulation_params[fName]['n']])
        
        # Eventually:  will want avility to pull in any set of sensors
        
    # phase calculation
    
    # export data = [shots x sensors], target = [shots x binary n]

    return np.array(features), np.array(target)
def  __gen_F_names(data_dir):
    # # need to decide if we're keeping filament & surface together
    # m=params['m'];sensor_set=params['sensor_set']
    # n=params['n'];
    
    fNames = glob.glob(data_dir+'floops_*.hist')
    
    return fNames
################################################
# Define regression model
# Create model
# f = wx + b, sigmoid at the end
class LogisticRegression(nn.Module):

    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted
    
###################################################
if __name__ == '__main__':run_regression()