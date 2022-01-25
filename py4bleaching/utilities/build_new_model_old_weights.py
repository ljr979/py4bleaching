import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dl4tsc.utils.constants import ARCHIVE_NAMES, CLASSIFIERS, ITERATIONS
from dl4tsc.utils.utils import calculate_metrics
from loguru import logger
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from tensorflow import keras
#from src.python_photobleaching.analysis import clean_trajectories


from random import sample
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from matplotlib import pyplot
from numpy import where

    
def prepare_data_for_training(X_train, y_train, X_test, y_test,):
    # transform the labels from integers to one hot vectors
    enc = preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.values.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.values.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)

    if len(X_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        X_train = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

    return X_train, y_train, X_test, y_test


def fit_classifier(X_train, y_train, X_test, y_test, classifier_name, output_directory):
  
    classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory)

    classifier.fit(X_train, y_train, X_test, y_test, y_true)


def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=False):
    if classifier_name == 'fcn':
        from dl4tsc.classifiers import fcn
        return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mlp':
        from dl4tsc.classifiers import mlp
        return mlp.Classifier_MLP(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'resnet':
        from dl4tsc.classifiers import resnet
        return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcnn':
        from dl4tsc.classifiers import mcnn
        return mcnn.Classifier_MCNN(output_directory, verbose)
    if classifier_name == 'tlenet':
        from dl4tsc.classifiers import tlenet
        return tlenet.Classifier_TLENET(output_directory, verbose)
    if classifier_name == 'twiesn':
        from dl4tsc.classifiers import twiesn
        return twiesn.Classifier_TWIESN(output_directory, verbose)
    if classifier_name == 'encoder':
        from dl4tsc.classifiers import encoder
        return encoder.Classifier_ENCODER(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcdcnn':
        from dl4tsc.classifiers import mcdcnn
        return mcdcnn.Classifier_MCDCNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'cnn':  # Time-CNN
        from dl4tsc.classifiers import cnn
        return cnn.Classifier_CNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'inception':
        from dl4tsc.classifiers import inception
        return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose)


# to build new model without training, need to DEFINE the 'resnet' class neural network. did this by defining the class but created this classifier using the function 'create_classifier' which I used in my training script, but to do this also had to create shape of data and nb classes using 'long_trajectories' dataframe (because 'create_classifier' function imports the class 'resnet' and gives shape etc. based on a datafram  to get the shape)

#step01 cleanup trajectories (gathers trajectories and then smooshed them,outputs as a new file called compiled_for_new_model)
def clean_trajectories(input_folder, output_folder):
        
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    #want to find the folders in the dummy data 
    folder_list = [folder for folder in os.listdir(input_folder)]

    #now loop through those folders  and pull out the CSV files in each folder and put them in a dataframe with the extra columns (so need to figure out what to call location name and stuff)
    all_trajectory_data = []
    for folder in folder_list:
        #this line needs some sort of string formatting to fetch all the .csv files !!! only is taking the exact name of the folder and I can't figure it out. want to SPLIT on /*csv in all of my dummy data
        folder
        trajectories_files_list = [[f'{root}/{filename}' for filename in files if '.csv' in filename] for root, dirs, files in os.walk(f'{input_folder}{folder}')]
        #the below line is a confusing sentence that flattens the list from being multiple lists into one list
        trajectories_files_list = [item for sublist in trajectories_files_list for item in sublist]
        for filepath in trajectories_files_list:
            #changes the file path to replace weight back slashes that os.walk adds in, and makes hard to split on
            filepath = filepath.replace('\\', '/')
            file_details = re.split('/|\\\\', filepath)
            exp_condition = file_details[-3].split('_')[-1]
            coloc_type = file_details[-2]
            protein_type = file_details [-1].split('_')[0]

            raw_trajectories = pd.read_csv(f"{filepath}")
            # see pd.concat documentation for more info
            raw_trajectories.drop([col for col in raw_trajectories.columns.tolist() if ' ' in col], axis=1, inplace = True)
            raw_trajectories = raw_trajectories.T.reset_index().rename(columns = {'index': 'molecule_number'})
            raw_trajectories['treatment'] = exp_condition
            raw_trajectories['colocalisation'] = coloc_type
            raw_trajectories['protein'] = protein_type
            raw_trajectories['file_path'] = filepath
            # store DataFrame in list
            all_trajectory_data.append(raw_trajectories)
    smooshed_trajectories = pd.concat(all_trajectory_data)
    smooshed_trajectories['colocalisation'] = smooshed_trajectories['colocalisation'].str.capitalize()
    #now need to assign unique names to the molecules
    smooshed_trajectories['metadata'] = smooshed_trajectories['treatment'] + '_' + smooshed_trajectories['colocalisation'] + '_' + smooshed_trajectories['protein']
    #line below does exact same thing as above, but different way (in case u want to change later)
    #smooshed_trajectories ['metadata'] = [f'{treatment}_{coloc_type}_{protein_type}' for treatment, coloc_type, protein_type in smooshed_trajectories[['treatment', 'colocalisation', 'protein']].values]
    smooshed_trajectories['molecule_number'] = [f'{metadata}_{x}' for x, metadata in enumerate(smooshed_trajectories['metadata'])]
    trajectories = ['molecule_number'] + [col for col in smooshed_trajectories.columns.tolist() if type(col) == int]
    trajectories = smooshed_trajectories[trajectories].copy()
    #now need to output this file as the original data before renaming the molecules
    trajectories.to_csv(f'{output_folder}/{folder}_compiled_for_new_model.csv')
    return trajectories




def make_new_model(time_data, output_folder, robust_weights_path):

    #define the other inputs to be able to create the classifier
    nb_classes = 3
    input_shape = (len(time_data.T), 1)
    classifier_name = 'resnet'
    output_directory=output_folder

    #create the model architecture ('resnet' neural network) that is not built based on training
    classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory)

    #call the 'build_model' function within the class called 'resnet'
    model=classifier.build_model(input_shape= input_shape, nb_classes=nb_classes)
    #save this new architecture (no weights adjusted yet )
    model.save(output_folder + 'new_model_architecture.hdf5')

    #make sure the model we made using resnet architecture but trained using only short trajectoreis is loaded as such
    robust_weights_model = keras.models.load_model(robust_weights_path)

    #rename the model archietecture we just madewith resnet but did not train, as adjusted weights as we are about to adjust them using the weights from the short trajectory trained model
    adjusted_weights_model = model

    #set the weights of the new model to the weights we extract from the short only model
    adjusted_weights_model.set_weights(robust_weights_model.get_weights())

    #save this new model to the output folder to test. 
    adjusted_weights_model.save(output_folder + 'new_model_robust_weights.hdf5')
    return adjusted_weights_model, time_data


if __name__ == "__main__":

#this input path is to the FILE THAT IS THE SHAPE YOU WANT (i.e, the trajectories you've just extracted, so the trajectories folder in the experiment you're analysing)

    input_folder='Results/training_model/TEST_XAXIS_fUNCTION/labelled_for_training_July/long_only_trajectories.csv'

    #this is where you want the NEW model to be saved
    output_folder='Results/training_model/TEST_XAXIS_fUNCTION/labelled_for_training_July/LONG_ONLY_ADJUSTED_WEIGHTS/'

    #this is the path to the model you want to use the weights from. i.e. copy this older model from the online repository with all the models, into the workspace you're in, and then put that path into this definition
    robust_weights_path = 'Results/training_model/short_only_normalised_trajectories/trained_model/resnet_1/best_model.hdf5'

    #this function is defined at the top, but just cleans up the trajectories and puts them all together to find information about i.e. shape etc to feed to the new model
    trajectories=clean_trajectories(input_folder, output_folder)
    #gets just the time data from all trajectories 
    time_data = trajectories[[col for col in trajectories.columns.tolist() if col not in ['molecule_number']]].reset_index(drop=True)

    #this does the work, saves the new model and the model with the empty architecture (no weights) in your output folder
    adjusted_weights_model, time_data = make_new_model(time_data, output_folder, robust_weights_path)

#NOTE: need to make this into functions to clean it up, then add it into the py4bleaching section!!!! 
