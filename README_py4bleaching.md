# py4bleaching
Installable module for calculating molecule size/stoichiometries from photobleaching trajectories

The below lines creates a submodule in git 

cd utilities/pb/
git submodule add https://github.com/ljr979/py4bleaching py4bleaching

This module is called upon by running ```trajectory_analysis.py``` scripts, and outputs multiple folders

An example can be found in ```data/2_example_python_output/two_colour/1_trajectory_analysis/client/py4bleaching/```. 
Each section is briefly described below. 


| *Folder*    | *Files* | *Description*   |
|-------------|---------------------------------|---------------|
|calculate molecule size|```molecule_counts_median.csv```, and ```molecule_counts.csv```|These files contain median num subunits per molecule, and the count for every molecule in the dataset. Also find histogram for each timepoint and treatment, based on each 'step type' to decide which is most appropriate |
|clean data|```initial_compiled_data```, ```cleaned_data```, and ```normalised_clean_data```|first version of compiled trajectories, then the cleaned dataset used moving forward (with organised mol name), and the cleaned data df normalised to the max fluorescence of each trajectory|
|fitting changepoints|```median_steps```, ```step_sizes```, and png files showing the distribution of step sizes for each step type. Also, an example fitting of changepoints to check the probability threshold.|median size of photobleaching steps for each step type, and also all of the combined step sizes the median was derived from|
|model for prediction|```.hdf5``` files ```Model_name```, ```model_init```, ```model_architecture```, ```new model name```|These examples are for using the 'build new model' x_norm key word, i.e. these files are output for those situations where you make a whole new model to match your new data x limits. model name is the 'robust' model, model_architecture is the architecture (without weights) for the new model, and the 'new model name' is the new model with weights from the old model. Other model types will just save the model you have chosen here|
|predict labels|```normalised_with_labels.csv```, ```predicted_labels.csv```|the data that is y-normalised for each trajectory, wit hthe model-predicted labels included (with unique mol names). Predicted labels is the file which has these labels mapped back onto the RAW (not normalised) data for calculating mol size and step size|

# Training a resnet model
The script titled ```training.py``` can be run on trajectories that have been manually labelled such that a new model is generated and can be called upon by the analysis section of py4bleaching. These models should be added to the repository ```py4bleaching_models``` which can be found [**here**](https://github.com/ljr979/py4bleaching_models).
The workflow for this is as follows:
1. gather trajectories by photobleaching molecules using TIRF microscopy
2. run training script up to the point where you are directed to **'label molecules'**. This is the point at which you need to visualise your trajectories and assign a label to them manually. These should indicate they ```(0)``` do not have clear photobleaching steps ```(1)``` do have clear photobleaching steps ```(2)``` are not appropriate for further analysis (will be disregarded)
3. run the remainder of ```training.py``` on the trajectories file with the labels in it, to produce the model
4. The model and information on the type of data is was trained on should be added into py4bleaching_models repository. 

# A  note on model selection

Below is a brief description to help to decide which classification model is appropriate for your data. Reminder that these models are trained to recognize the SHAPE of your data, in regard to clear photobleaching steps or not, and will filter out any data that bleaches instantly or does not bleach at all. If this is not appropriate for your data, you may need to train your own model. Using the below x_norm values will transform your data such that it is appropriate for the desired, corresponding model

| *model_name*    | *x_norm* | *Description*   |
|-------------|-------------|---------------|
|```Model_1``` |'False'|This model was not trained on y-normalised data, or accounting for x-axis length variation. Trained on AF488 photobleaching trajectories, 600 frames long.|
|```Model_2```|'False'|y-normalised fluorescence data from multiple fluorophores. Performs very well on a variety of fluorophores, but not well on variable x-axis lengths. Trained on 600-length|
|```Model_3```|'noise_50'|Trained on y-normalised fluorescence values, which have been extended from multiple x-axis lengths to 1000 frames with empty frames. these have been filled with gaussian noise using the mean and s.d. of the last 50 frames prior to the end|
|```Model_4```|'build_new_weights'|takes the weights from your **favourite** robust model (we used Model_2), and creates a new architecture from your newly obtained data, then fills it with the reliable weights. Uses this model to predict on variable x-axis lengths. |