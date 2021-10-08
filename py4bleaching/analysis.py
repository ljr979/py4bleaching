import os
import re
from random import sample

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sdt import changepoint
from tensorflow import keras
from py4bleaching.utilities.collect_model import download_model
#step01 cleanup trajectories 
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

    #now need to output this file as the original data before renaming the molecules
    smooshed_trajectories.to_csv(f'{output_folder}/{folder}_initial_compiled_data.csv')

    #now need to assign unique names to the molecules
    smooshed_trajectories['metadata'] = smooshed_trajectories['treatment'] + '_' + smooshed_trajectories['colocalisation'] + '_' + smooshed_trajectories['protein']
    #line below does exact same thing as above, but different way (in case u want to change later)
    #smooshed_trajectories ['metadata'] = [f'{treatment}_{coloc_type}_{protein_type}' for treatment, coloc_type, protein_type in smooshed_trajectories[['treatment', 'colocalisation', 'protein']].values]
    smooshed_trajectories['molecule_number'] = [f'{metadata}_{x}' for x, metadata in enumerate(smooshed_trajectories['metadata'])]
    timeseries_data = ['molecule_number'] + [col for col in smooshed_trajectories.columns.tolist() if type(col) == int]
    timeseries_data = smooshed_trajectories[timeseries_data].copy()
    #this line saves the UNIQUE MOLECULE NUMBERS with the RAW TRAJECTORIES (not normalised or anything)
    timeseries_data.to_csv(f'{output_folder}cleaned_data.csv')

    #now to normalise on the y axis so that we can feed normalised data to the model, which will always have a normalised Y axis
    normalised_trajectories = timeseries_data.copy().set_index('molecule_number')
    normalised_trajectories = (normalised_trajectories.T/normalised_trajectories.T.max()).T.reset_index()

    #split up into smaller chunks easier for streamlit stuff 
    normalised_trajectories.to_csv(f'{output_folder}normalised_clean_data.csv') 

#step02 predict labels section 
def prepare_data_to_predict(time_data):
            #here we are just trying to make all of the data we are training on has 1000 time points so that if I have longer time series, the model we have trained can operate on that shape of data
    time_columns = [int(col) for col in time_data.columns.tolist() if col not in ['molecule_number', 'label']]

    #if max(time_columns) != 1000:
        #new_columns = [str(timepoint) for timepoint in range(max(time_columns)+1, 1000)]
        #time_data[new_columns] = np.nan
    return time_data.values.reshape((time_data.shape[0], time_data.shape[1], 1))

def predict_labels(time_data, model_path):
    # evaluate best model on new dataset
    x_predict = prepare_data_to_predict(time_data)
    input_shape = x_predict.shape[1:]

    #feeds data into the model
    model = keras.models.load_model(model_path)
    #gives probability that at that time point, there is a particular label
    y_pred = model.predict(x_predict)
    #decides if it is that label based on the max probability and then assigns to y_pred for adding to a new label column
    y_pred = np.argmax(y_pred, axis=1)

    # Add labels back to original dataframe
    time_data['label'] = y_pred

    return time_data


#step03 fitting changepoints section

#defining function to plot and check probability threshold for detecting changepoints
def find_changepoints(df, output_folder, molecule_names=False, intensity_column='fluorescence', probability_threshold=0.5, visualise=False):
    """this function has the purpose of using the 'Bayesian offline' changepoint module to find the changepoints in the trajectories of all the 'small molecules' in the trajectories that have been picked by the model and processed with labels. This module outputs the probability of a change point (a step in fluorescence) at every single point in the trajectory. we then use this data to determine what changepoints are real, and then find the step size. 

    Args:
        df (dataframe): this is the dataframe which contains all the trajectories with unique names from the 'clean data' script.
        output_folder ([type]): this is the folder where we put the graphs to check that the changpoint threshold is picking everything up that we want, and matches them to the molecule name. 
        molecule_names (list): list of unique names to test, default is all molecules when 'False' 
        intensity_column (str, optional): [the name of the column containing the fluorescence intensity values from the trajectory (not the changepoint lines)]. Defaults to 'fluorescence'.
        probability_threshold (float, optional): [the threshold for the probability of a changepoint occurring, anything above this value is officially a 'changepoint' and thus we get an output of changepoint indices instead of probabilities which they are previously]. Defaults to 0.2.
        visualise (bool, optional): [if TRUE it visualises the probability at every point on the left in a graph, with the probability threshold shown to see how many changepoints are above that. on the right it will plot the trajectory with the changpoint data plotted over the top to see how well it fits]. Defaults to False.

    Returns:
        [dataframe]: [dataframe with the changepoints and the trajectories for every molecule]
    """
    if not molecule_names:
        molecule_names=[col for col in df.columns if not col =='time']
    processed_dfs=[]
    for molecule in molecule_names:
        molecule
        molecule_df = df[['time', molecule]].rename(columns={molecule:'fluorescence'})
        molecule_df['time']= molecule_df['time'].astype(int)
        molecule_array = np.array(molecule_df[intensity_column])
        
        det = changepoint.BayesOffline("const", "gauss")
        molecule_df['changepoint_probs'] = det.find_changepoints(molecule_array)

        change_points = molecule_df[molecule_df['changepoint_probs'] >= probability_threshold].index.tolist()
        
         # Assign step labels according to changepoints above probability_threshold
        molecule_df = label_steps(molecule_df, probability_threshold=probability_threshold)

        # add a column to contain the molecule name for later grouping
        molecule_df['molecule_name'] = molecule

        if visualise:
            fig, axes = plt.subplots(1, 2, figsize=(20,5))
            sns.lineplot(data=molecule_df, x='time', y='changepoint_probs', ax=axes[0])
            axes[0].axhline(probability_threshold, color='green', linestyle='dotted')
            sns.lineplot(data=molecule_df, x='time', y=intensity_column, ax=axes[1])
            for changes in change_points:
                axes[1].axvline(changes, color='red', linestyle='--')
            plt.title(f'molecule {molecule}')
            plt.savefig(f'{output_folder}{molecule}_{probability_threshold}.png')
            plt.show()
        
        processed_dfs.append(molecule_df)

    return pd.concat(processed_dfs)

def label_steps(molecule_df, probability_threshold=0.5):
    """[goes through the dataframe containing the changepoint steps, and then makes a new column with labels at every point at each 'level' of the changepoint steps so that we can average all the fluorescence at each of these step levels]

    Args:
        molecule_df ([data frame]): [the dataframe with the changepoints and the fluorescence ]
        probability_threshold (float, optional): [decided according to plotting changepoint to decide the best value for picking up changepoints]. Defaults to 0.2.

    Returns:
        [dataframe]: [df containing the trajectories, the changepoints, and each point at each step labelled]
    """
    x = 0
    step_labels = []
    for prob in molecule_df['changepoint_probs']:
        if prob >= probability_threshold:
            x += 1
        step_labels.append(x)
    molecule_df['step_label'] = step_labels

    return molecule_df

def plot_fitted_example(processed_molecules, molecule_name, output_folder=False):
    """Plots line plot for processed molecule data, including fluorescence, changepoints and fitted steps.

    Args:
        processed_molecules (df): DataFrame containing fluorescence, step_label, changepoint_probs, step_fluorescence
        molecule_name (int): number of the molecule to preview
        output_folder (bool, optional): If str output_folder is passed, save plot to fitted_molecules folder withint output_folder. Defaults to False.
    """
    
    if not os.path.exists(f'{output_folder}fitted_molecules/'):
        os.makedirs(f'{output_folder}fitted_molecules/')

    molecule_df = processed_molecules[processed_molecules['molecule_name'] == molecule_name]
    
    fig, ax = plt.subplots()
    for changes in molecule_df[molecule_df['changepoint_probs'] >= probability_threshold]['time'].tolist():
        ax.axvline(changes, color='red', linestyle='--')
    sns.lineplot(data=molecule_df, x='time', y='fluorescence', color='black')
    sns.lineplot(data=molecule_df, x='time', y='step_fluorescence', hue='step_label', palette='bright')
    
    if output_folder:
        plt.savefig(f'{output_folder}fitted_molecules/{molecule_name}.png')
    plt.show()

def find_photobleaching_steps(processed_molecules):
     # calcucate the mean fluorescence value at each step before it steps down to the next level
    step_fluorescence = processed_molecules.groupby(['molecule_name', 'step_label']).median()['fluorescence'].reset_index().rename(columns={'fluorescence': 'step_fluorescence'})

    processed_molecules = pd.merge(processed_molecules, step_fluorescence, on=['molecule_name', 'step_label'], how='outer')
        # calculate stepsize
    step_sizes=[]
    for molecule, df in step_fluorescence.groupby('molecule_name'):
        steps = df['step_fluorescence'].tolist()
        step_size = [np.nan] + [steps[x] - steps[x+1]  for x in range(len(steps)-1)]
        df['step_size'] = step_size
        step_sizes.append(df)
    step_sizes=pd.concat(step_sizes)
    #next line filters out the steps back up in intensity (these are negative values) and leaves you with all the steps down
    step_sizes = step_sizes[step_sizes['step_size'] > 0]
    return processed_molecules, step_sizes


def calculating_step_sizes(step_sizes, output_folder):
    # collect list of matched molecule names and steps for each condition type: last steps with highest step label, and then single steps with highest step label in molecules that only have one step
    last_steps = [(molecule, df.sort_values('step_label').iloc[-1]['step_label']) for molecule, df in step_sizes.groupby('molecule_name')]
    single_steps = [(molecule, df.sort_values('step_label').iloc[-1]['step_label']) for molecule, df in step_sizes.groupby('molecule_name') if len(df) == 2]

    #now assigns a 1 to the molecule name in whichever condition it meets: ie. all small molecules are every value as we are already looking at 'small molecules' data frame. so every value =1. Then looks for the molecule names and step labels that are in the 'last steps' defined above, and the step label and if it's in thaat list then assign 1 etc. (grabs actual step size value at this point by the .values part)
    step_sizes['all_small'] = 1
    step_sizes['last_step'] = [1 if (molecule, step) in last_steps else 0 for molecule, step in step_sizes[['molecule_name', 'step_label']].values]
    step_sizes['single_step'] = [1 if (molecule, step) in single_steps else 0 for molecule, step in step_sizes[['molecule_name', 'step_label']].values]

    median_steps={}
    for molecule_type in ['all_small', 'last_step', 'single_step']:
        filtered_df = step_sizes[step_sizes[molecule_type]==1].copy()
        #calculate the average / median step size for each step type
        median_steps[molecule_type]=filtered_df['step_size'].median()

        fig,ax = plt.subplots() 
        sns.distplot(filtered_df['step_size'])
        ax.axvline(filtered_df['step_size'].median(), linestyle='--', color='red')
        ax.annotate(f'median = {round(filtered_df["step_size"].median(), 2)}', xy = (0.8, 0.8), xycoords = 'figure fraction')
        plt.title(molecule_type)
        plt.savefig(f'{output_folder}{molecule_type}_step_type_dist.png')
        plt.show()
        
    median_steps=pd.DataFrame(median_steps, index=['step_size']).T.reset_index().rename(columns={'index':'step_type'})
    return step_sizes, median_steps

#step04 calculating molecules sizes and plotting distributions   
def calculating_stoichiometries(clean_data, step_sizes):
    # filter out throwaway trajectories
    usable_trajectories = clean_data[clean_data['label'] != 2.0].copy()
    #usable_trajectories = clean_data.copy()
    # collect (max) fluorescence values for each trajectory
    timepoint_columns = [col for col in usable_trajectories.columns.tolist() if col not in ['molecule_number', 'label']]

    molecule_counts = []
    for molecule, df in usable_trajectories.groupby('molecule_number'):
        max_fluorescence_value = np.max(sorted(df[timepoint_columns].values[0], reverse=True))
        # Calculate average number of molecules by mean fluorescence / step size
        molecule_counts.append(pd.DataFrame([molecule, max_fluorescence_value]).T)

    molecule_counts = pd.concat(molecule_counts)
    molecule_counts.columns = ['molecule_number', 'max_fluorescence']
    for size_type, size in step_sizes.items():
        size_type=size_type.replace('_stepsize', '')
        molecule_counts[f'{size_type}_mol_count'] = molecule_counts['max_fluorescence'] / size
        
    molecule_counts[[col for col in molecule_counts.columns if 'molecule_number' not in col]]=molecule_counts[[col for col in molecule_counts.columns if 'molecule_number' not in col]].astype(float)
    molecule_counts[['treatment', 'colocalisation', 'protein', 'molecule_number']] = molecule_counts['molecule_number'].str.split('_', expand = True)
    return molecule_counts

def plotting_molecule_size(step_sizes, molecule_counts, output_folder):
    #split and plot for different experiment treatments
    # Visualise distribution of molecules
    for size_type, size in step_sizes.items():
        size_type=size_type.replace('_stepsize', '')
        #make the dataframe manually binned for the histogram, counts how many molecules counted in each bin, then takes the max so you can set the y limit accordingly (scales all histograms to max value) 
        max_hist_count = pd.cut(molecule_counts[f'{size_type}_mol_count'], bins=np.arange(0, 50)).reset_index().groupby(f'{size_type}_mol_count').count().max().tolist()[0]
        x_limits = molecule_counts.max()
        
        for group, df in molecule_counts.groupby(['colocalisation', 'treatment']): 
            fig, ax = plt.subplots()
            sns.histplot(df[f'{size_type}_mol_count'], binwidth=1, color='#20731f')
            ax.annotate(f"median={round(df[f'{size_type}_mol_count'].median(), 2)}", xy=(0.7, 0.8), xycoords='figure fraction')
            ax.axvline(df[f'{size_type}_mol_count'].median(), linestyle='--', color='red')
            plt.title(f'{group} {size_type}')
            plt.xlim(0,df.max())
            plt.ylim(0, max_hist_count)
            plt.xlabel('Molecule count')
            plt.savefig(f'{output_folder}/histogram_{group}_{size_type}.png')
            plt.show()


def sanity_checks(clean_data):
    test_Df = pd.melt(clean_data, id_vars= ['label', 'molecule_number'], value_vars=[f'{x}' for x in range(600)], var_name='timepoint', value_name='intensity' )

    test_Df[['treatment', 'colocalisation', 'protein', 'molecule_number']] = test_Df['molecule_number'].str.split('_', expand = True)

    test_Df['timepoint']=test_Df['timepoint'].astype(int)
    sns.lineplot(data=test_Df, x='timepoint', y='intensity', hue='treatment')
    plt.show()

    for treatment, df in test_Df.groupby('treatment'):     
        sns.lineplot(data=df, x='timepoint', y='intensity', hue='label')
        plt.title(treatment)
        plt.show()




def pipeline(input_folder,output_folder, probability_threshold, model_name):

    output_folders = ['clean_data', 'predict_labels', 'fitting_changepoints','calculate_molecule_size', 'model_for_prediction']
    
    for folder in output_folders:
        if not os.path.exists(f'{output_folder}{folder}/'):
            os.makedirs(f'{output_folder}{folder}/')
    #this bit says if the model isnt in the output folder then download it
    if not os.path.exists(f'{output_folder}model_for_prediction/{model_name}.hdf5'):
        download_model(model_name, f'{output_folder}model_for_prediction/')

    model_path = f'{output_folder}model_for_prediction/{model_name}.hdf5'

    clean_trajectories(input_folder, f'{output_folder}clean_data/')
    #----------------------------------------------------
    #now predict labels
    # Read in dataset
    raw_data = pd.read_csv(f'{output_folder}clean_data/cleaned_data.csv')
    raw_data.drop([col for col in raw_data.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)
    normalised_data = pd.read_csv(f'{output_folder}clean_data/normalised_clean_data.csv')
    normalised_data.drop([col for col in normalised_data.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)
    # prepare time series data (leaves only time so that the model knows how to read it)
    time_data = normalised_data[[col for col in normalised_data.columns.tolist() if col not in ['molecule_number', 'label']]].reset_index(drop=True)
   
    #this is the actual prediction part
    time_data = predict_labels(time_data, model_path)

    #adds molecule numbers from original dataframe back onto the labelled time data (normalised here)
    time_data['molecule_number'] = normalised_data['molecule_number']

    # Save to csv (normalised)
    time_data.to_csv(f'{output_folder}/predict_labels/normalised_with_labels.csv')

    #now need to also map the labels from the time_data above, onto the raw data using a dictionary to match up the label to the molecule number (call this predicted_labels.csv)
    labels_dict = dict(time_data[['molecule_number','label']].values)

    #map these labels onto raw data
    raw_data['label'] = raw_data['molecule_number'].map(labels_dict)
    raw_data.to_csv(f'{output_folder}/predict_labels/predicted_labels.csv')


    #--------------------------------------------------------

    #reading in labelled data and cleaning it up a bit for the function. This is specific to each experiment (potentially for a while and helps to look at the data, one day shouldn't be so different)
    clean_data = pd.read_csv(f'{output_folder}predict_labels/predicted_labels.csv')
    clean_data = clean_data.drop([col for col in clean_data.columns.tolist() if 'Unnamed: 0' in col], axis=1)
    small_molecules = clean_data[clean_data['label'] == 1].copy().set_index('molecule_number')
    small_molecules= small_molecules [[col for col in small_molecules.columns.tolist() if col not in ['molecule_number', 'label']]].copy().T.reset_index().rename(columns={'index':'time'})
    
    #creates a subset of the data randomly to visualise the thresholds for small molecules
    test_list = [col for col in small_molecules.columns.tolist() if col is not 'time']
    molecules_to_test = sample(test_list, 5)
    #molecules_to_test = sample([col for col in small_molecules.columns.tolist() if col is not 'time'], 5)

    #performs the function and visualise the sample. once probability is checked can move forward
    test_probs = find_changepoints(small_molecules, f'{output_folder}fitting_changepoints/', molecules_to_test, intensity_column='fluorescence', probability_threshold=probability_threshold, visualise=True)


    # actual changepoint finding for all molecules
    processed_molecules = find_changepoints(small_molecules, f'{output_folder}fitting_changepoints/', molecule_names=False, intensity_column='fluorescence', probability_threshold=probability_threshold, visualise=False)

    #this calls the function that finds the median fluorescence at each changepoint, calls this a step fluorescence, and then also calculates the step size. Passes back the step fluorescence in processed molecules and step sizes filtered to remove the steps back up in fluorescence (step_sizes)
    processed_molecules, step_sizes = find_photobleaching_steps(processed_molecules)

    #if you want to plot the changepoints, uncomment line below 
    # for molecule in processed_molecules['molecule_name'].unique().tolist():
    #     molecule
    #     plot_fitted_example(processed_molecules, molecule_name= molecule, f'{output_folder}fitting_changepoints/'=False)

    step_sizes, median_steps= calculating_step_sizes(step_sizes,f'{output_folder}fitting_changepoints/')
    step_sizes.to_csv(f'{output_folder}fitting_changepoints/step_sizes.csv')
    median_steps.to_csv(f'{output_folder}fitting_changepoints/median_steps.csv')

    #----------------------------------------------------------------

        #reading in step sizes 
    step_sizes = pd.read_csv(f'{output_folder}fitting_changepoints/median_steps.csv')
    step_sizes = step_sizes.drop([col for col in step_sizes.columns.tolist() if 'Unnamed: 0' in col], axis=1)
    step_sizes=dict(step_sizes[['step_type', 'step_size']].values)

    #reading in trajectory data
    clean_data = pd.read_csv(f'{output_folder}predict_labels/predicted_labels.csv')
    clean_data = clean_data.drop([col for col in clean_data.columns.tolist() if 'Unnamed: 0' in col], axis=1)

    molecule_counts=calculating_stoichiometries(clean_data,step_sizes)
    plotting_molecule_size(step_sizes,molecule_counts,output_folder=f'{output_folder}calculate_molecule_size/')
    sanity_checks(clean_data)

    # save the data! YAY!
    molecule_counts.to_csv(f'{output_folder}calculate_molecule_size/molecule_counts.csv')
    molecule_counts.groupby(['treatment', 'colocalisation', 'protein']).median().reset_index().to_csv(f'{output_folder}calculate_molecule_size/molecule_counts_median.csv')


if __name__ == "__main__":

    input_folder = 'imagejresults/Experiment32_hsp27_01/'
    output_folder = 'Results/model_2_test/Hsp27_all/'

    #change this according to the model that you'd like to use (from the repo with all the models)
    model_name = 'Model_2'
  
    pipeline(input_folder, output_folder, probability_threshold=0.5, model_name=model_name)