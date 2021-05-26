
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
from src.python_photobleaching.analysis import clean_trajectories
#need to add clean trajectories import :) 
#---------------------------------------

def prepare_data_for_labelling(input_files, output_folder):
    smooshed_trajectories=[]
    for filepath in input_files:
        trajectories=pd.read_csv(f'{filepath}')
        trajectories.drop([col for col in trajectories.columns.tolist() if ' ' in col], axis=1, inplace = True)
        smooshed_trajectories.append(trajectories)
    smooshed_trajectories = pd.concat(smooshed_trajectories)
    #in case there are duplicate names we need to renumber (otherwise when classifying it it plots on top of one another)
    smooshed_trajectories[['treatment', 'colocalisation', 'protein', 'number']]=smooshed_trajectories['molecule_number'].str.split('_', expand=True)
    smooshed_trajectories['number']=[str(x) for x in range(len(smooshed_trajectories))]
    smooshed_trajectories['molecule_number']=smooshed_trajectories[['treatment', 'colocalisation', 'protein', 'number']].agg('_'.join, axis=1)
    smooshed_trajectories.drop(['treatment', 'colocalisation', 'protein', 'number'], axis=1, inplace=True)

    if not os.path.exists (f'{output_folder}labelling_molecules/'):
        os.makedirs(f'{output_folder}labelling_molecules/')

    smooshed_trajectories.to_csv(f'{output_folder}labelling_molecules/data_for_training.csv')


#Run streamlit NOW on smooshed_trajectories df. 
#need to define the labels dictionary in the 'main' to map the 0, 1 , 2 for discard, small, big

def map_labels(input_path, output_folder, labels):
    labelled_data=pd.read_csv(input_path)
    labelled_data.drop([col for col in labelled_data.columns if 'Unnamed' in col], axis=1, inplace=True)
    labelled_data['label'] = labelled_data['label'].map(labels)
    labelled_data.to_csv(f'{output_folder}labelled_data.csv')

#--------------------------------------------------------------
#this section here is the 'train_model script. this plots the data, splits up and preps train and test data for you, fits it with a classifier probability, then creates the actual classifier.
def plot_data_samples(dataframe, sample_numbers):
    ''' 
    Plot the time series data relating to the input list of sample numbers.

    sample_numbers: list of integers
        E.g. [1, 7, 22, 42]
    '''
    
    unique_labels = dataframe['label'].astype(int).unique()
    num_classes = len(unique_labels)
    if num_classes<=5:
        class_colors = dict(zip(unique_labels, ['palevioletred', 'crimson', 'purple', 'midnightblue', 'darkorange'][:num_classes]))
        palette = {sample: class_colors[class_number] for sample, class_number in dataframe.reset_index()[['index', 'label']].values}
    else:
        class_colors = sns.color_palette(n_colors=num_classes)

    for i in sample_numbers:
        logger.info(f'sample {i} is class {dataframe.loc[i, "label"].astype(int)}')

    for_plotting = pd.melt(dataframe.reset_index(), id_vars=['index', 'label'], var_name='time', value_vars=[col for col in dataframe.columns if col not in ['molecule_number', 'label']])
    for_plotting = for_plotting[for_plotting['index'].isin(sample_numbers)]
    fig, ax = plt.subplots()
    sns.lineplot(
        data=for_plotting,
        x='time',
        y='value',
        hue='index',
        palette=palette,
    )
    ax.set_ylabel('Data value')
    ax.set_xlabel('Timepoint')
    ticks = [x for x in range(for_plotting['time'].astype(int).max()) if x % 100 == 0 ]
    ax.set_xticks(ticks)
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    plt.savefig(f"{output_folder}samples.png")


def prepare_data_for_training(x_train, y_train, x_test, y_test,):
    # transform the labels from integers to one hot vectors
    enc = preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.values.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.values.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        x_train = x_train.values.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.values.reshape((x_test.shape[0], x_test.shape[1], 1))

    return x_train, y_train, x_test, y_test


def fit_classifier(x_train, y_train, x_test, y_test, classifier_name, output_directory):

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # transform the labels from integers to one hot vectors
    enc = preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.values.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.values.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        x_train = x_train.values.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.values.reshape((x_test.shape[0], x_test.shape[1], 1))

    input_shape = x_train.shape[1:]
    classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory)

    classifier.fit(x_train, y_train, x_test, y_test, y_true)


def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=False):
    if classifier_name == 'fcn':
        from classifiers import fcn
        return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mlp':
        from classifiers import mlp
        return mlp.Classifier_MLP(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'resnet':
        from classifiers import resnet
        return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcnn':
        from classifiers import mcnn
        return mcnn.Classifier_MCNN(output_directory, verbose)
    if classifier_name == 'tlenet':
        from classifiers import tlenet
        return tlenet.Classifier_TLENET(output_directory, verbose)
    if classifier_name == 'twiesn':
        from classifiers import twiesn
        return twiesn.Classifier_TWIESN(output_directory, verbose)
    if classifier_name == 'encoder':
        from classifiers import encoder
        return encoder.Classifier_ENCODER(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcdcnn':
        from classifiers import mcdcnn
        return mcdcnn.Classifier_MCDCNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'cnn':  # Time-CNN
        from classifiers import cnn
        return cnn.Classifier_CNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'inception':
        from classifiers import inception
        return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose)


def train_new_model(time_data, labels, output_folder, itrs=1, classifier_name='resnet'):

    # Split data into train and test portions
    X_train, X_test, y_train, y_test = train_test_split(time_data.T, labels)

    # ----------------------train model----------------------

    for itr in range(itrs):
        output_directory = f'{output_folder}{classifier_name}_{itr}/'

        logger.info(f'Method: {classifier_name} using {itr} iterations.')

        if os.path.exists(f'{output_directory}df_metrics.csv'):
            logger.info(f'{classifier_name} using {itr} iteration already done')
        else:
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

            fit_classifier(X_train, y_train, X_test, y_test, classifier_name, output_directory)

            logger.info('Training complete.')

            # evaluate best model on test data
            x_train, y_train, x_test, y_test = prepare_data_for_training(X_train, y_train, X_test, y_test,)
            model = keras.models.load_model(output_directory + 'best_model.hdf5')
            y_pred = model.predict(x_test)
            y_pred = np.argmax(y_pred, axis=1)
            y_true = np.argmax(y_test, axis=1)
            model_metrics = calculate_metrics(y_true, y_pred, 0.0)
            logger.info(f'Iteration {itr}: df metrics')
            [logger.info(f'{measure}: {round(val, 2)}') for measure, val in model_metrics.T.reset_index().values]



def prepare_data_to_predict(time_data):
    if len(time_data.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        time_data = time_data.values.reshape((time_data.shape[0], time_data.shape[1], 1))
    return time_data

def predict_labels(time_data, model_path):
    # evaluate best model on new dataset
    x_predict = prepare_data_to_predict(time_data)
    input_shape = x_predict.shape[1:]
    model = keras.models.load_model(model_path)
    y_pred = model.predict(x_predict)
    y_pred = np.argmax(y_pred, axis=1)
    # Add labels back to original dataframe
    time_data['label'] = y_pred
    return time_data



def plot_labels():

    data=pd.melt(time_data, id_vars='label', value_vars=[col for col in time_data.columns.tolist() if 'label' not in col], var_name='time')

    #plots the data in the colour for the label it was given
    data['time'] = data['time'].astype(int)
    sns.lineplot(data=data.groupby(['label', 'time']).mean().reset_index(), x='time', y='value', hue='label')
    sns.lineplot(data=data, x='time', y='value', hue='label')
    palette = {0.0: 'firebrick', 1.0: 'darkorange', 2.0: 'rebeccapurple', '0': 'firebrick', '1': 'darkorange', '3': 'rebeccapurple'}

def compare_labels(raw_data, time_data):
    #compares the original label column with the predicted label column?
    raw_data['predict_label'] = time_data['label']
    raw_data['diff'] = [0 if val == 0 else 1 for val in (raw_data['label'] - raw_data['predict_label'])]
    return raw_data


def plot_comparison(comparison, palette=False):
    if not palette:
        palette='muted'

    time_columns=[col for col in comparison.columns.tolist() if col not in ['label', 'molecule_number','predict_label', 'diff']]

    data=pd.melt(comparison, id_vars=['label'], value_vars=time_columns, var_name='time')
    data['time']=data['time'].astype(int)
    for molecule, df in comparison[comparison['diff'] == 1].groupby('molecule_number'):
        
        original_label = df['label'].tolist()[0]
        predict_label = df['predict_label'].tolist()[0]

        fig, ax = plt.subplots()
        sns.lineplot(data=data.groupby(['label', 'time']).mean().reset_index(), x='time', y='value', hue='label', palette=palette)
        sns.lineplot(x=np.arange(0, len(time_columns)), y=df[time_columns].values[0], color=palette[original_label], linestyle='--')
        plt.title(f'Molecule {molecule}: original label {original_label}, predicted {predict_label}')
        plt.show()

#--------------------


def pipeline(input_path,output_folder, labels):

    output_folders = ['trained_model','validation']
    for folder in output_folders:
        if not os.path.exists(f'{output_folder}{folder}/'):
            os.makedirs(f'{output_folder}{folder}/')

    #-----------------------------
    #train model section 
    map_labels(input_path, f'{output_folder}labelling_molecules/', labels)

    raw_data = pd.read_csv(f'{output_folder}labelling_molecules/labelled_data.csv')
    raw_data.drop([col for col in raw_data.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)
    raw_data['label'] = raw_data['label'].fillna(0)

    # prepare time series data
    time_data = raw_data[[col for col in raw_data.columns.tolist() if col not in ['molecule_number', 'label']]].T.reset_index(drop=True)

    # prepare label data
    labels = raw_data['label'].copy().astype(int)

    # train model
    #train_new_model(time_data, labels, f'{output_folder}trained_model/', itrs=1, classifier_name='resnet')

    #read in raw data and time data
    raw_data = pd.read_csv(f'{output_folder}labelling_molecules/labelled_data.csv')
    raw_data.drop([col for col in raw_data.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)
    raw_data['label'] = raw_data['label'].fillna(0)

    # prepare time series data
    time_data = raw_data[[col for col in raw_data.columns.tolist() if col not in ['molecule_number', 'label']]]
    time_data = predict_labels(time_data, f'{output_folder}trained_model/resnet_1/best_model.hdf5')
    time_data.groupby('label').count()
    comparison = compare_labels(raw_data, time_data)
    palette = {0.0: 'firebrick', 1.0: 'darkorange', 2.0: 'rebeccapurple', '0': 'firebrick', '1': 'darkorange', '3': 'rebeccapurple'}
    plot_comparison(comparison, palette=palette)

    

if __name__ == "__main__":
    input_files = [
        'Results/training_model/20201222_Exp_15_647data/clean_data/cleaned_data.csv',
        'Results/training_model/20210113_Exp_17_647data/clean_data/cleaned_data.csv'
        ]
    output_folder = 'Results/training_model/20210525_647datacompiled/'


    prepare_data_for_labelling(input_files, output_folder)

    #now do streamlit at this point and come back to run pipeline

    #input path for the labelled molecules after streamlit
    #input_path = f'Results/training_model/20210525_647datacompiled/labelling_molecules/labelled_molecules.csv'
    input_path='Results/analysis_pipeline/20201209_Exp_14_test/predict_labels/predicted_labels.csv'
    #change this dictionary depending on what I label them in streamlit
    labels={0:0, 1:1, 2:2}

    pipeline(input_path, output_folder, labels)

