"""
@author = Adam Barker

"""
import numpy as np

from ilpd_parser import ILPD_Parser

import os

from os import listdir

from os.path import isfile, join

import matplotlib.pyplot as plt

class ILPD_Constructor():

    """takes parsed files and builds a matrix of test samples and targets for feeding into a DNN

    this file and ilpd_parser should be located in the ./ILPD folder otherwise you will need to

    change the path for loading the file names.  NOTE:  Filename convention should be

    '1_sample_TARGET_ch1_DATETIME' where TARGET is the operation the file is

    performing: add, and, or xor"""



    def __init__(self, path="", data_path="C:\Users\jonkr\OneDrive\Documents\E222"):

        self.path = path #where the input files are to be found

        self.data_path = data_path #where the output files will be saved


        return



    def load_files(self):

        """loads in the files in each directory and parses them"""



        #Load and parse AND files

        and_path = r"C:\Users\jonkr\OneDrive\Documents\GitHub\ILPD\raw_data\100_Single_Acquisition_and\\"

        #and_filenames = os.listdir("{}{}".format(self.path,and_path))
        
        and_filenames = [f for f in listdir(and_path) if isfile(join(and_path, f))]
        
        self.and_labels = np.zeros((1 , len(and_filenames)))

        self.and_labels[:] = 1 #One hot vector AND label  = [1 , 0 , 0]



        #Initialize array to load data samples: 125000 points in each sample X number of samples

        self.and_array = np.empty((125000, len(and_filenames)))

        for f in range(0, len(and_filenames)):

            print("{}{}{}".format(self.path, and_path, and_filenames[f]))
            Parse = ILPD_Parser("{}{}{}".format(self.path, and_path, and_filenames[f]))

            self.and_array[:,f] = np.array(Parse.parse())
            
            if f % 10 == 0 and f != 0:

                print("{} AND files loaded".format(f))

        print("Loading of AND files complete.")
        
        
        
        #Load and parse ADD files

        add_path = r"C:\Users\jonkr\OneDrive\Documents\GitHub\ILPD\raw_data\100_Single_Acquisition_add\\"

        #add_filenames = os.listdir("{}{}".format(self.path,add_path))
        
        add_filenames = [f for f in listdir(add_path) if isfile(join(add_path, f))]

        self.add_labels = np.zeros((1 , len(add_filenames)))

        self.add_labels[:] = 2 #One hot vector ADD label  = [0 , 1 , 0]



        #Initialize array to load data samples: 125000 points in each sample X number of samples

        self.add_array = np.empty((125000, len(add_filenames)))

        for f in range(0, len(add_filenames)):
            
            print("{}{}{}".format(self.path, add_path, add_filenames[f]))
            Parse = ILPD_Parser("{}{}{}".format(self.path, add_path, add_filenames[f]))

            self.add_array[:,f] = np.array(Parse.parse())

            if f % 10 == 0 and f != 0:

                print("{} ADD files loaded".format(f))

        print("Loading of ADD files complete.")


        #Load and parse XOR files

        xor_path = r"C:\Users\jonkr\OneDrive\Documents\GitHub\ILPD\raw_data\100_Single_Acquisition_xor\\"

        #xor_filenames = os.listdir("{}{}".format(self.path,xor_path))

        xor_filenames = [f for f in listdir(xor_path) if isfile(join(xor_path, f))]
         
        self.xor_labels = np.zeros((1 , len(xor_filenames)))

        self.xor_labels[:] = 3 #One hot vector XOR label  = [0 , 0 , 1]



        #Initialize array to load data samples: 125000 points in each sample X number of samples

        self.xor_array = np.empty((125000, len(xor_filenames)))

        for f in range(0, len(xor_filenames)):

            print("{}{}{}".format(self.path, xor_path, xor_filenames[f]))
            Parse = ILPD_Parser("{}{}{}".format(self.path, xor_path, xor_filenames[f]))

            self.xor_array[:,f] = np.array(Parse.parse())

            if f % 10 == 0 and f != 0 :

                print("{} XOR files loaded".format(f))

        print("Loading of XOR files complete.")


        #build the dataset by concat all arrays together

        self.data_array = np.transpose(np.concatenate((self.and_array, self.add_array, self.xor_array),axis=1))

        self.label_array = np.transpose(np.concatenate((self.and_labels, self.add_labels, self.xor_labels), axis=1))

        self.label_array = self.label_array.ravel()


        return



    def dataset_shuffle(self):

        """randomly shuffles the dataset (including labels)"""



        #numpy random shuffle only shuffles in the first dimension

        shuff_data_array = self.data_array.T

        shuff_label_array = self.label_array.T



        rng_state = np.random.get_state() #captures the random seed

        np.random.shuffle(shuff_data_array)

        np.random.set_state(rng_state) #uses the same random seed as previous so the shuffling matches

        np.random.shuffle(shuff_label_array)



        self.shuffled_data_array = shuff_data_array.T

        self.shuffled_label_array = shuff_label_array.T



        return



    def save_dataset(self, save_filename, type='shuffled_data_array' ):

        """saves the dataset to a file type """

        if type == 'shuffled_data_array':

            np.save(save_filename,self.shuffled_data_array)

            np.save(('{}_labels'.format(save_filename)),self.shuffled_label_array)



        elif type == 'data_array':

            np.save(save_filename,self.data_array)

            np.save(('{}_labels'.format(save_filename)),self.label_array)



        elif type == 'xor_array' or type == 'and_array' or type == 'add_array':

            fn = 'self.{}_array'.format(type[0:2])

            fn_labels = 'self.{}_labels'.format(type[0:2])

            np.save(save_filename, fn)

            np.save(('{}_labels'.format(save_filename)),fn_labels)

        return

