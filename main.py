# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 14:46:14 2018

@author: jonkr

This program attempts to train a Logistic Regression algorithm locally
with waveform data taken from the TI MSP430f2619 in order to classify 
different instruction level commands i.e. AND, ADD, XOR.
 
"""

import numpy as np
from ilpd_parser import ILPD_Parser
from ilpd_constructor import ILPD_Constructor
from os import listdir
from os.path import isfile, join
from sklearn.linear_model import LogisticRegression


##########################################
# INSTANSTIATE LOGISTIC REGRESSION OBJECT
LR = LogisticRegression()                
##########################################



ILPD = ILPD_Constructor()

ILPD.load_files()

print(np.shape(ILPD.data_array))
print(np.shape(ILPD.label_array))

print("training with array shapes: ", np.shape(ILPD.data_array[:,:]))

LR.fit(ILPD.data_array[:,:], ILPD.label_array)


##############################################################################

# AND SIGNAL FOR PREDICTION 

csvand = [f for f in listdir(r"C:\Users\jonkr\OneDrive\Documents\GitHub\ILPD\raw_data\1_and\\") if isfile(join(r"C:\Users\jonkr\OneDrive\Documents\GitHub\ILPD\raw_data\1_and\\", f))]

print(r"C:\Users\jonkr\OneDrive\Documents\GitHub\ILPD\raw_data\1_and\\" + csvand[0])

and_array = np.empty((125000,1))

Parse = ILPD_Parser(r"C:\Users\jonkr\OneDrive\Documents\GitHub\ILPD\raw_data\1_and\\" + csvand[0])

and_array[:,0] = np.array(Parse.parse())

print("test file loaded")

print("Loading of and file complete.")
 
and_array = np.transpose(and_array)
print(LR.predict(and_array))
print("This should be label: 1")
##############################################################################


##############################################################################

# ADD SIGNAL FOR PREDICTION
csvadd = [f for f in listdir(r"C:\Users\jonkr\OneDrive\Documents\GitHub\ILPD\raw_data\1_add\\") if isfile(join(r"C:\Users\jonkr\OneDrive\Documents\GitHub\ILPD\raw_data\1_add\\", f))]

print(r"C:\Users\jonkr\OneDrive\Documents\GitHub\ILPD\raw_data\1_add\\" + csvadd[0])

add_array = np.empty((125000,1))

Parse = ILPD_Parser(r"C:\Users\jonkr\OneDrive\Documents\GitHub\ILPD\raw_data\1_add\\" + csvadd[0])

add_array[:,0] = np.array(Parse.parse())

print("test file loaded")

print("Loading of add file complete.")
 
add_array = np.transpose(add_array)   

print(LR.predict(add_array))

print("This should be label: 2")
##############################################################################


##############################################################################

# XOR SIGNAL FOR PREDICTION

#csvxor = [f for f in listdir(r"C:\Users\jonkr\OneDrive\Documents\GitHub\ILPD\raw_data\100_sample_average\\") if isfile(join(r"C:\Users\jonkr\OneDrive\Documents\GitHub\ILPD\raw_data\1_xor\\", f))]

#print(r"C:\Users\jonkr\OneDrive\Documents\GitHub\ILPD\raw_data\1_xor\\" + csvxor[0])

xor_array = np.empty((125000,1))
and_array = np.empty((125000,1))

Parse = ILPD_Parser(r"C:\Users\jonkr\OneDrive\Documents\GitHub\ILPD\raw_data\100_sample_average\xor_ch1.csv")
Parse2 = ILPD_Parser(r"C:\Users\jonkr\OneDrive\Documents\GitHub\ILPD\raw_data\100_sample_average\and_ch1.csv")

xor_array[:,0] = np.array(Parse.parse())
and_array[:,0] = np.array(Parse2.parse())

print("test files loaded")
 
xor_array = np.transpose(xor_array)   
and_array = np.transpose(and_array)

print(LR.predict(xor_array))
print("This should be label: 3")

print(LR.predict(and_array))
print("This should be label: 1")

Parse.OScope_Plot()

Parse2.OScope_Plot()
