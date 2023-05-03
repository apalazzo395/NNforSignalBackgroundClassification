Code to produce the (parametric) Deep Neural Network ((P)DNN) for the HDBS analysis.
The necessary steps are described below. Each one of them corresponds to a python script in this project. Moreover, two .txt file and a python file imported by all the others are included and described below. A log file is saved at the end of each step.

# Configuration_TAG.txt
This text file contains the configuration parameters:
- ntuplePath: path to the flat ntuples produced by the CxAODReader
- dfPath: path to the pandas data frames into which the flat ntuples are converted by the saveToPkl.py file. At each step different subdirectories will be automatically created (if not already existing) where the results will be saved. The subdirectories will be created according to the type of data (jet collection, signal, background, analysis, channel, preselectionCuts) and neural network (DNN or PDNN) considered.
- inputFiles: list of names of the input .root files that will be converted into pandas dataframes (without absolute path and extension)
- backgrounds: list of the different backgrounds in the input files
- signals: list of the different signals in the input files
- rootBranchSubSample: list of the variables (root branches) that will actually be converted into pandas dataframes
- variablesToSave: list of the variables to save in the dataframe created at the end of step 2, along with other variables created in that step. Different lists are specified according to the type of analysis and signal considered
- variablesToDerive: list of the variables to derive starting from other variables included in the original dataframe
- InputFeatures(Merged/Resolved): list of the variables that will be fed to the (P)DNN in the merged/resolved regime

# DSIDtoMass.txt
This text file contains the map used to convert DSID variables into mass values.

# Functions.py
This file contains useful functions that will be called by the other .py scripts. It also contains the default value of the flags that can be specified then running the script (in the ReadArgParser function).

# Step 1) saveToPkl.py
This script takes the input .root files and converts them into .pkl files. Only branches listed in rootBranchSubSample in the configuration file are stored in the output .pkl files.

It requires two flags:
- jetCollection (-j): the name of the jet collection used to produce the .root files (e.g. 'TCC', 'UFO_PFLOW') - mandatory
- tag (-t): the CxAOD tag, the default one is considered if not specified 

The output directory structure is 
```
outputDir = dfPath + tag + '/' + jetCollection
```

The output files have the same name as the input ntuples with `.root` replaced by `_DF.pkl`.

# Step 2) buildDataset.py
This script takes the .pkl files created in the previous step and selects in each of them only the events in the channel/analysis specified and of the signal/background type requested. A new signal/background flag ('isSignal') is associated to each one of them (1/0). A new mass value ('mass') is associated to each signal event according to the map stored in DSIDtoMass.txt. The 'mass' value of background events is randomly chosen among those listed in DSIDtoMass.txt. The derived variables specified in the config file are computed. The shuffled dataframe resulting from these manipulation in saved.

The following input flags can be specified: 
- jetCollection (-j)
- analysis (-a): type of analysis ('merged' or 'resolved') -- mandatory
- channel (-c): the channel considered ('ggF' or 'VBF') -- mandatory
- signal (-s): the signal to selected -- mandatory
- background (-b): the type of background to select. Multiple backgrounds can be selected and must be specified in quotation marks separated by a space (e.g. 'Zjets Diboson'). If 'all' (default), all backgrounds in the backgrounds list in the configuration file will be selected
- preselection cuts (-p): string that will be translated to python commad to filter the inital dataframes according to it (e.g. 'lep1_pt > 0 and lep1_eta > 0')

The output directory structure is 
```
outputDir = dfPath + tag + '/' + jetCollection + '/' + analysis + '/' + channel + '/' + preselectionCuts + '/' + signal + '/' + background
```

The output file name is 
```
outputFile = 'MixData_' + tag + '_' + jetCollection + '_' + analysis + '_' + channel + '_' + preselectionCuts + '_' + signal + '_' + background
```

# Step 3) splitDataset.py
This script splits the dataframe produced at the previous step into one train and one test sample, according to the fraction specified. The median and the interquartile range of all possible input features to the neural network are computed and saved in an output file (`variables.json`). The interquartile range is defined as the difference between the 75 quartile and the 25 one. The train weight is computed on the train set. It first equalizes the different number of signal events with the same mass inside the signal sample and then the number of total signal to the number of background events. These weights will be used to train the neural network. The train and test sample created are saved. 

The following input flags must be specified:
- jetCollection (-j)
- analysis (-a) -- mandatory
- channel (-c) -- mandatory
- preselectionCuts (-p)
- background (-b)
- signal (-s) -- mandatory
- training fraction (-t): relative size of the training sample, between 0 and 1

The output directory is the same as the one in the previous step:
```
outputDir = dfPath + tag + '/' + jetCollection + '/' + analysis + '/' + channel + '/' + preselectionCuts + '/' + signal + '/' + background
```

The output files names are 
```
outputFile1 = 'data_train_' + tag + '_' + jetCollection + '_' + analysis + '_' + channel + '_' + preselectionCuts + '_' + signal + '_' + background + '_' + validationFraction + 't.pkl'
outputFile2 = 'data_test_' + tag + '_' + jetCollection + '_' + analysis + '_' + channel + '_' + preselectionCuts + '_' + signal + '_' + background + '_' + validationFraction + 't.pkl'
```

# Step 4) buildDNN.py / buildPDNN.py
These scripts run the (Parametric) Deep Neural Network. The train set saved at the end of the previous step is loaded; the input features specified in the config file are scaled according to the variables.json file previously created; in this way the distribution of each of their variables is approximately a gaussian with mean = 0 and sigma = 1 (if possible). These variables are then fed to the neural network.

The following input flags can be specified:
- jetCollection (-j) 
- analysis (-a) -- mandatory
- channel (-c) -- mandatory
- nodes (-n): number of nodes of the (P)DNN
- layers (-l): number of layers of the (P)DNN
- epochs (-e): number of epochs for the training 
- validation (-v): fraction of the training data that will actually be used for validation
- dropout (-d): fraction of the neurons that will be dropped during the training
- training fraction (-t): relative size of the training sample, between 0 and 1
- doTrain (--doTrain): if 1 the training is performed, if 0 the model stored in the output directory is loaded without any training
- doTest (--doTest): if 1 the test is performed, if 0 the script is stopped after the training

The output directory structure is 
```
outputDir = dfPath + tag + '/' + jetCollection + '/' + analysis + '/' + channel + '/' + preselectionCuts + '/' + signal + '/' + background + '/' + (p)DNN
```

Several output files are saved:
- `architecture.json`: it contains the architecture of the trained model. If `--doTrain 0` this architecture will be loaded
- `weights.h5`: it contains the weights of the trained model. If `--doTrain 0` this architecture will be loaded
- `variables.json`: a copy of the file containing the scaling factors that has been loaded
- loss, architecture plots
- a new subdirectory is created for each mass hypothesis. It contains plots with scores, confusion matrix and background rejections.


# Step 5) computeSignificance.py
This script compares the significance obtained using the invariant mass to the one obtained using the neural network output. 

It takes as input the file created at the end of step 2, selects the events in the requested region and the computes the significance.

The following flags can be specified:
- jetCollection (-j) 
- signal (-s) -- mandatory
- regime (-r): regime in which the significance must be computed. Different regimes can be specified in quotation marks separated with a space (e.g.: 'Pass_Res_GGF_ZZ_2btag_SR Pass_Res_GGF_ZZ_01btag_SR'); the significance will be computed and saved for each regime separately and their quadratic sum is also computed -- mandatory

The output directory structure is 
```
outputDir = dfPath + tag + '/' + jetCollection + '/' + preselectionCuts + '/' + regime + '/' + signal
```

The output files are
```
invariant mass histogram = 'InvariantMass_' + regime + '_' + mass + '_' + tag + '_' + jetCollection + '_' + preselectionCuts + '_' + signal + '_' + background + '.png'
scores histogram = 'Scores_' + regime + '_' + mass + '_' + tag + '_' + jetCollection + '_' + preselectionCuts + '_' + signal + '_' + background + '.png'
significance comparison = 'Significance_' + tag + '_' + jetCollection + '_' + preselectionCuts + '_' + signal + '_' + background + regime + '.png'
```

# Old step 5) 
At this point you might want to use your (p)DNN into the CxAODReader. In order to do so, you first have to convert the model into a format that can be read by the reader. The interface between the neural network and the reader is the lwtnn package: https://github.com/lwtnn/lwtnn. This latter comes together with a bunch of conversion scripts. IMPORTANT NOTICE: the CxAODReader also has a duplicate of the lwtnn package. However, as of 2021-05-27, the lwtnn used by the reader is a deprecated version. For converting the model, please clone the latest version of lwtnn from its git repository. Detailed instructions on how to use the conversion tools are found in the README thereof. In summary, as we use a sequential model, we need to run the following command:

  $ python lwtnn/converters/keras2json.py architecture.json variables.json weights.h5 > neural_net.json