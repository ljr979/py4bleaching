    """This script can perform all functions required to train a new resnet model for trajectory classification if needed.

    """
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
from py4bleaching.analysis import clean_trajectories
from random import sample

def prepare_data_for_labelling(input_files, output_folder, streamlit=False):
    """This function prepares the trajectories data you want to train your model on, to manually label the training dataset in an app called 'streamlit' where they get labelled and saved to then train the model later on.

    Args:
        input_files (list): a list of paths to 'cleaned_data.csv' files (output by 'clean_trajectories' function)
        output_folder (string): path to folder you want to save data to
        streamlit (bool, optional): if you have not performed the streamlit labelling already, this is False, and as such normalises the data on the Y axis, and splits it up into batches of 100 trajectories to label in streamlit and saves them as such. Defaults to False.
    """
    if not os.path.exists (f'{output_folder}labelling_molecules/'):
        os.makedirs(f'{output_folder}labelling_molecules/')


    if streamlit:
        input_files=[filename for filename in f'{output_folder}labelling_molecules/' if 'labelled_data' in filename]
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

    if not streamlit:
        #normalisation section where we normalise to max value in each trajectory
        normalised_trajectories = smooshed_trajectories.copy().set_index('molecule_number')
        normalised_trajectories = (normalised_trajectories.T/normalised_trajectories.T.max()).T

        #split up into smaller chunks easier for streamlit stuff 
        n = 100  #chunk row size
        [normalised_trajectories[i:i+n].to_csv(f'{output_folder}labelling_molecules/data_for_training_{i}.csv') for i in range(0,normalised_trajectories.shape[0],n)]
    else:
        #here we are just trying to make all of the data we are training on has 1000 time points so that if I have longer time series, the model we have trained can operate on that shape of data
        time_columns = [int(col) for col in smooshed_trajectories.columns.tolist() if col not in ['molecule_number', 'label']]
        if max(time_columns) != 1000:
            new_columns = [str(timepoint) for timepoint in range(max(time_columns)+1, 1000)]
            smooshed_trajectories[new_columns] = np.nan
        smooshed_trajectories.to_csv(f'{output_folder}labelling_molecules/smooshed_labels.csv')


#Run streamlit NOW on smooshed_trajectories df. 
#need to define the labels dictionary in the 'main' to map the 0, 1 , 2 for discard, small, big
def map_labels(input_path, output_folder, labels):
    """this maps the names you label each trajectory (classifying them) in streamlit, to be 0, 1, or 2 for the model, as they need to be trained as numbers not words. Saves for training.

    Args:
        input_path (string): path to the labelled, collated trajectories
        output_folder (string): path to the folder you want this to be saved in
        labels (dictionary): dictionary to define what you labelled each as in streamlit, to 0 1 or 2.
    """
    labelled_data=pd.read_csv(input_path)
    labelled_data.drop([col for col in labelled_data.columns if 'Unnamed' in col], axis=1, inplace=True)
    labelled_data['label'] = labelled_data['label'].map(labels)
    labelled_data.rename(columns={'sample_name':'molecule_number'}, inplace=True)
    labelled_data.to_csv(f'{output_folder}labelled_data.csv')

#--------------------------------------------------------------
#this section here is the 'training model script. this plots the data, splits up and preps train and test data for you, fits it with a classifier probability, then creates the actual classifier.
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


def prepare_data_for_training(X_train, y_train, X_test, y_test,):
    """functions to prepare the data for the model, it needs one hot vectors so changes the train and test into that datatype

    Args:
        X_train (array): all are arrays from train_test_split from scikitlearn
        y_train (array): 
        X_test (array): 
        y_test (array): 

    Returns:
        multivariate one dimensional arrays: one hot encoded arrays split up for training and testing.
    """
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
    """fits the classifier (we use resnet but can adjust it to another if necessary) to the split we made previously.

    Args:
        X_train (array): data split into train and test from the trajectories
        y_train (array): data split into train and test from the trajectories
        X_test (array): data split into train and test from the trajectories
        y_test (array): data split into train and test from the trajectories
        classifier_name (string): name of the classifier we want to use for our model architecture
        output_directory (string): where to save the model classifier
    """
    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

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

    input_shape = X_train.shape[1:]
    classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory)

    classifier.fit(X_train, y_train, X_test, y_test, y_true)


def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=False):
    """creates the classifier we want to use for our model

    Args:
        classifier_name (str): name of classifier to choose
        input_shape (tuple of integers): shape of the X train data
        nb_classes (int): number of classes (how many classifications- we use 3)
        output_directory (string): path to save
        verbose (bool, optional):  Defaults to False.

    Returns:
        _type_: _description_
    """
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


def train_new_model(time_data, labels, output_folder, itrs=1, classifier_name='resnet'):
    """train a new model and save to classify your photobleaching data trajectories

    Args:
        time_data (df): dataframe containing all timeseries data (no molecule number)
        labels (list): list of the label names we used to classify the trajectories
        output_folder (string): path to save 
        itrs (int, optional): number of interations in training the new model. Defaults to 1.
        classifier_name (str, optional): name of classifier. Defaults to 'resnet'.
    """
    # Split data into train and test portions
    X_train, X_test, y_train, y_test = train_test_split(time_data, labels)

    # ----------------------train model----------------------

    for itr in range(itrs):
        output_directory = f'{output_folder}{classifier_name}_{itr+1}/'

        logger.info(f'Method: {classifier_name} using {itr+1} iterations.')

        if os.path.exists(f'{output_directory}df_metrics.csv'):
            logger.info(f'{classifier_name} using {itr+1} iteration already done')
        else:
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

            fit_classifier(X_train, y_train, X_test, y_test, classifier_name, output_directory)

            logger.info('Training complete.')

            # evaluate best model on test data
            X_train, y_train, X_test, y_test = prepare_data_for_training(X_train, y_train, X_test, y_test,)
            model = keras.models.load_model(output_directory + 'best_model.hdf5')
            y_pred = model.predict(X_test)
            y_pred = np.argmax(y_pred, axis=1)
            y_true = np.argmax(y_test, axis=1)
            model_metrics = calculate_metrics(y_true, y_pred, 0.0)
            logger.info(f'Iteration {itr+1}: df metrics')
            [logger.info(f'{measure}: {round(val, 2)}') for measure, val in model_metrics.T.reset_index().values]


def prepare_data_to_predict(time_data):
    """reshape time data for model

    Args:
        time_data (df): trajectory data no mol name

    Returns:
        df: time data reshaped
    """
    if len(time_data.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        time_data = time_data.values.reshape((time_data.shape[0], time_data.shape[1], 1))
    return time_data

def predict_labels(time_data, model_path):
    """predicts labels on new data

    Args:
        time_data (df): reshaped time data (fluorescence trajectories)
        model_path (str): path to the model you just created

    Returns:
        df: time data with another column containing the predicted label (predicted by his new model)
    """
    # evaluate best model on new dataset
    x_predict = prepare_data_to_predict(time_data)
    input_shape = x_predict.shape[1:]
    model = keras.models.load_model(model_path)
    y_pred = model.predict(x_predict)
    y_pred = np.argmax(y_pred, axis=1)
    # Add labels back to original dataframe
    time_data['label'] = y_pred
    return time_data



def plot_labels(time_data):
    """plots mean fluorescence for each label

    Args:
        time_data (df): trajectories with label
    """
    data=pd.melt(time_data, id_vars='label', value_vars=[col for col in time_data.columns.tolist() if 'label' not in col], var_name='time')

    #plots the data in the colour for the label it was given
    data['time'] = data['time'].astype(int)
    sns.lineplot(data=data.groupby(['label', 'time']).mean().reset_index(), x='time', y='value', hue='label')
    sns.lineplot(data=data, x='time', y='value', hue='label')
    palette = {0.0: 'firebrick', 1.0: 'darkorange', 2.0: 'rebeccapurple', '0': 'firebrick', '1': 'darkorange', '3': 'rebeccapurple'}

def compare_labels(raw_data, time_data):
    """function compares the predicted vs labelled (manually) classification

    Args:
        raw_data (df): df containing our labels
        time_data (df): df containing predicted labels by model

    Returns:
        df  : raw data containing the original vs predicted label
    """
    #compares the original label column with the predicted label column?
    raw_data['predict_label'] = time_data['label']
    raw_data['diff'] = [0 if val == 0 else 1 for val in (raw_data['label'] - raw_data['predict_label'])]
    return raw_data


def plot_comparison(comparison, palette=False):
    """function to plot the label 'truth' (by us) versus the label 'predicted' (by the model) to test its' accuracy / if we agree with its' choice (in this plottingwe can see if we actually agree with its choice more than our original choice, or if it's wildly wrong.)

    Args:
        comparison (df): the 'raw data' df with the original and predicted labels
        palette (bool, optional): dictionary which maps the label to the colour we want it to plot as. Defaults to False.
    """
    if not palette:
        palette='muted'

    time_columns=[col for col in comparison.columns.tolist() if col not in ['label', 'molecule_number','predict_label', 'diff']]

    data=pd.melt(comparison, id_vars=['label'], value_vars=time_columns, var_name='time')
    data['time']=data['time'].astype(int)
    data_subset = sample([col for col in data.columns.tolist()], 20)
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
    """the workflow to train the new model and check it's ability to predict the labels.

    Args:
        input_path (str): the path to the 'smooshed labels' (trajectories with our manually labelled shapes.)
        output_folder (str): output folder for the model (top level that we are working in)
        labels (dict): dictionary defining the labels (name of label that we manually used, mapping to 0, 1 or 2 for the model)
    """
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
    time_data = raw_data[[col for col in raw_data.columns.tolist() if col not in ['molecule_number', 'label']]].reset_index(drop=True)

    # prepare label data
    labels = raw_data['label'].copy().astype(int)

    # train model
    train_new_model(time_data, labels, f'{output_folder}trained_model/', itrs=1, classifier_name='resnet')

    #read in raw data and time data
    raw_data = pd.read_csv(f'{output_folder}labelling_molecules/labelled_data.csv')
    raw_data.drop([col for col in raw_data.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)
    raw_data['label'] = raw_data['label'].fillna(0)

    # prepare time series data
    time_data = raw_data[[col for col in raw_data.columns.tolist() if col not in ['molecule_number', 'label']]]
    time_data = predict_labels(time_data, f'{output_folder}trained_model/resnet_1/last_model.hdf5')
    time_data.groupby('label').count()
    comparison = compare_labels(raw_data, time_data)
    palette = {0.0: 'firebrick', 1.0: 'darkorange', 2.0: 'rebeccapurple', '0': 'firebrick', '1': 'darkorange', '3': 'rebeccapurple'}
    plot_comparison(comparison, palette=palette)

    

if __name__ == "__main__":

    output_folder = 'Results/training_model/'
        
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    #location of files containing trajectories that are completely unprocessed that you want to train data on (just need to be cleaned up)
    input_folder=''
    clean_trajectories(input_folder, output_folder, matched=False)
    #now get this saved clean trajectories so you can use them in streamlit
    input_files = [[f'{root}/{filename}' for filename in files if 'cleaned_data.csv' in filename] for root, dirs, files in os.walk(f'{input_folder}')]
    #the below line is a confusing sentence that flattens the list from being multiple lists into one list
    input_files = [item for sublist in input_files for item in sublist]
    prepare_data_for_labelling(input_files, output_folder, streamlit=False)

    #now do streamlit at this point and come back to run pipeline (now have a bunch of normalised trajectory files with same unique names cause it's easier for streamlit, and just smoosh them back together for training model)

    #giving empty list for the input files because in the previous running of this function we define the input files as a bunch of files that have 'labelled_data' in them, which is what we get from strealit output
    prepare_data_for_labelling(input_files=[], output_folder=output_folder, streamlit=True)
    #input path for the labelled molecules after streamlit
    input_path= f'{output_folder}labelling_molecules/smooshed_labels.csv'

    #change this dictionary depending on what I label them in streamlit
    labels={'big':0, 'small':1, 'discard':2}

    pipeline(input_path, output_folder, labels)
