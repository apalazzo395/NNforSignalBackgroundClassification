### This script takes as input the pkl files resulting from the conversion of the ntuples produced by the Reader, 
### selects the events according to the regime (merged/resolved) and channel (ggF/VBF) requested, add some useful information to each event (origin, isSignal, mass)
### and saves the resulting dataframe. 
### A cut on the mass value is performed according to the regime selected.
### Histograms and a correlation matrix of the relevat variables can be saved.

from Functions import ReadArgParser, ReadConfig, checkCreateDir, ShufflingData, SelectEvents, CutMasses, DrawVariablesHisto, DrawCorrelationMatrix, computeDerivedVariables
import pandas as pd
import numpy as np
import random
import os
import matplotlib
import matplotlib.pyplot as plt
import math
from termcolor import colored, cprint
#print(sys.executable, ' '.join(map(shlex.quote, sys.argv)))

overwriteDataFrame = False
#print(str(sys))
### Reading the command line
tag, analysis, channel, preselectionCuts, signal, background, drawPlots = ReadArgParser()

### Reading from config file
ntuplePath, InputFeatures, dfPath, variablesToSave, variablesToDerive, backgroundsList = ReadConfig(tag, analysis, signal)

### Creating output directories and logFile
tmpOutputDir = dfPath + analysis + '/' + channel + '/' + preselectionCuts# + '/ggFandVBF'# + '/ggFVBF'
print(format('First output directory: ' + tmpOutputDir), checkCreateDir(tmpOutputDir))
tmpFileCommonName = tag + '_' + analysis + '_' + channel + '_' + preselectionCuts

outputDir = tmpOutputDir + '/' + signal + '/' + background
print(format('Second output directory: ' + outputDir), checkCreateDir(outputDir))
fileCommonName = tmpFileCommonName + '_' + signal + '_' + background
logFileName = outputDir + '/logFile_buildDataset_' + fileCommonName + '.txt'

logFile = open(logFileName, 'w')
logFile.write('Input files path: ' + dfPath + 'CxAOD tag: ' + tag + '\nAnalysis: ' + analysis + '\nChannel: ' + channel + '\nPreselection cuts: ' + preselectionCuts + '\nSignal: ' + signal + '\nBackground: ' + background)

### Loading DSID-mass map and storing it into a dictionary
DictDSID = {}
DSIDfileName = 'DSIDtoMass.txt'
with open(DSIDfileName) as DSIDfile:
    cprint('Reading DSID - mass correspondance from ' + DSIDfileName, 'green')
    lines = DSIDfile.readlines()
    for line in lines:
        DictDSID[int(line.split(':')[0])] = int(line.split(':')[1])

### Creating the list of the origins selected (signal + background)
if background == 'all':
    inputOrigins = backgroundsList.copy()
    backgroundLegend = backgroundsList.copy()
    logFile.write(' (' + str(backgroundsList) + ')')
else:
    inputOrigins = list(background.split('_'))
inputOrigins.append(signal)

### Creating empty signal and background dataframe 
dataFrameSignal = []
dataFrameBkg = []

for target in inputOrigins:
    fileName = tmpOutputDir + '/' + target + '_' + tmpFileCommonName + '.pkl'

    ### Loading dataframe if found and overwrite flag is false 
    if not overwriteDataFrame:
        if os.path.isfile(fileName):
            if target == signal:
                cprint('Found signal dataframe: loading ' + fileName, 'green')
                dataFrameSignal = pd.read_pickle(fileName)
            else:
                cprint('Found background dataframe: loading ' + fileName, 'green')
                dataFrameBkg.append(pd.read_pickle(fileName))

    ### Creating dataframe if not found or overwrite flag is true
    if overwriteDataFrame or not os.path.isfile(fileName):
        ### Defining local dataframe (we might have found only one among many dataframes)
        partialDataFrameBkg = []
        for file in os.listdir(dfPath):
            ### Loading input file
            if file.startswith(target) and file.endswith('.pkl'):
                cprint('Loading ' + dfPath + file, 'green')
                inputDf = pd.read_pickle(dfPath + file)
                ### Selecting events according to merged/resolved regime and ggF/VBF channel
                inputDf = SelectEvents(inputDf, channel, analysis, preselectionCuts, signal)
                ### Creating new column in the dataframe with the origin
                inputDf = inputDf.assign(origin = target)

                ### Filling signal/background dataframes
                if target == signal:
                    dataFrameSignal.append(inputDf)
                else:
                    partialDataFrameBkg.append(inputDf)

        ### Concatening and saving signal and background dataframes
        if target == signal:
            dataFrameSignal = pd.concat(dataFrameSignal, ignore_index = True)
            dataFrameSignal.to_pickle(fileName)

        elif target != signal:
            partialDataFrameBkg = pd.concat(partialDataFrameBkg, ignore_index = True)
            partialDataFrameBkg.to_pickle(fileName)
            ### Appending the local background dataframe to the final one
            dataFrameBkg.append(partialDataFrameBkg)

        cprint('Saved ' + fileName, 'green')

### Concatening the global background dataframe
dataFrameBkg = pd.concat(dataFrameBkg, ignore_index = True)

### Creating a new isSignal column with values 1 (0) for signal (background) events
dataFrameSignal = dataFrameSignal.assign(isSignal = 1)
dataFrameBkg = dataFrameBkg.assign(isSignal = 0)
'''
### Removing isolated event with high lep1_m
dataFrameBkg = dataFrameBkg.query('lep1_m < 0.15')
'''
### Converting DSID to mass in the signal dataframe
massesSignal = dataFrameSignal['DSID'].copy()
DSIDsignal = np.array(list(set(list(dataFrameSignal['DSID']))))

for DSID in DSIDsignal:
    massesSignal = np.where(massesSignal == DSID, DictDSID[DSID], massesSignal)
dataFrameSignal = dataFrameSignal.assign(mass = massesSignal)

### Cutting signal events according to their mass and the type of analysis
dataFrameSignal = CutMasses(dataFrameSignal, analysis)
massesSignalList = list(set(list(dataFrameSignal['mass'])))
cprint('Masses in the signal sample: ' + str(np.sort(np.array(massesSignalList))) + ' GeV (' + str(len(massesSignalList)) + ')', 'blue')
logFile.write('\nMasses in the signal sample: ' + str(np.sort(np.array(massesSignalList))) + ' GeV (' + str(len(massesSignalList)) + ')')

'''
### Assigning a random mass to background events according to the signal mass distribution 
massDict = dict(dataFrameSignal['mass'].value_counts(normalize = True))
massesBkg = random.choices(list(massDict.keys()), weights = list(massDict.values()), k = len(dataFrameBkg))
dataFrameBkg = dataFrameBkg.assign(mass = massesBkg)
'''

### Assigning a random mass to background events 
massesBkg = np.random.choice(massesSignalList, dataFrameBkg.shape[0])
dataFrameBkg = dataFrameBkg.assign(mass = massesBkg)

### Concatening signal and background dataframes
dataFrame = pd.concat([dataFrameSignal, dataFrameBkg], ignore_index = True)


passDict = {'merged': {'ggF': ['Pass_VV2Lep_MergHP_GGF_ZZ_2btag_SR', 'Pass_VV2Lep_MergLP_GGF_ZZ_2btag_SR', 'Pass_VV2Lep_MergHP_GGF_ZZ_01btag_SR', 'Pass_VV2Lep_MergLP_GGF_ZZ_01btag_SR', 'Pass_VV2Lep_MergHP_GGF_ZZ_2btag_ZCR', 'Pass_VV2Lep_MergLP_GGF_ZZ_2btag_ZCR', 'Pass_VV2Lep_MergHP_GGF_ZZ_01btag_ZCR', 'Pass_VV2Lep_MergLP_GGF_ZZ_01btag_ZCR', 'Pass_VV2Lep_MergHP_GGF_WZ_SR', 'Pass_VV2Lep_MergLP_GGF_WZ_SR', 'Pass_VV2Lep_MergHP_GGF_WZ_ZCR', 'Pass_VV2Lep_MergLP_GGF_WZ_ZCR'],
                       'VBF': ['Pass_VV2Lep_MergLP_VBF_ZZ_SR', 'Pass_VV2Lep_MergHP_VBF_ZZ_SR', 'Pass_VV2Lep_MergHP_VBF_ZZ_ZCR', 'Pass_VV2Lep_MergLP_VBF_ZZ_ZCR', 'Pass_VV2Lep_MergHP_VBF_WZ_SR', 'Pass_VV2Lep_MergLP_VBF_WZ_SR', 'Pass_VV2Lep_MergHP_VBF_WZ_ZCR', 'Pass_VV2Lep_MergLP_VBF_WZ_ZCR']
                   },
            'resolved': {'ggF': ['Pass_VV2Lep_Res_GGF_ZZ_2btag_SR', 'Pass_VV2Lep_Res_GGF_ZZ_01btag_SR', 'Pass_VV2Lep_Res_GGF_ZZ_2btag_ZCR', 'Pass_VV2Lep_Res_GGF_ZZ_01btag_ZCR', 'Pass_VV2Lep_Res_GGF_WZ_SR', 'Pass_VV2Lep_Res_GGF_WZ_ZCR'],
                         'VBF': ['Pass_VV2Lep_Res_VBF_ZZ_SR', 'Pass_VV2Lep_Res_VBF_ZZ_ZCR', 'Pass_VV2Lep_Res_VBF_WZ_SR', 'Pass_VV2Lep_Res_VBF_WZ_ZCR']
                     }
}
'''
passDict = {'merged':{'ggF': {'Radion': ['Pass_VV2Lep_MergHP_GGF_ZZ_2btag_SR', 'Pass_VV2Lep_MergLP_GGF_ZZ_2btag_SR', 'Pass_VV2Lep_MergHP_GGF_ZZ_01btag_SR', 'Pass_VV2Lep_MergLP_GGF_ZZ_01btag_SR', 'Pass_VV2Lep_MergHP_GGF_ZZ_2btag_ZCR', 'Pass_VV2Lep_MergLP_GGF_ZZ_2btag_ZCR', 'Pass_VV2Lep_MergHP_GGF_ZZ_01btag_ZCR', 'Pass_VV2Lep_MergLP_GGF_ZZ_01btag_ZCR'],
                            'RSG': ['Pass_VV2Lep_MergHP_GGF_ZZ_2btag_SR', 'Pass_VV2Lep_MergLP_GGF_ZZ_2btag_SR', 'Pass_VV2Lep_MergHP_GGF_ZZ_01btag_SR', 'Pass_VV2Lep_MergLP_GGF_ZZ_01btag_SR', 'Pass_VV2Lep_MergHP_GGF_ZZ_2btag_ZCR', 'Pass_VV2Lep_MergLP_GGF_ZZ_2btag_ZCR', 'Pass_VV2Lep_MergHP_GGF_ZZ_01btag_ZCR', 'Pass_VV2Lep_MergLP_GGF_ZZ_01btag_ZCR'],
                            'HVTWZ': ['Pass_VV2Lep_MergHP_GGF_WZ_SR', 'Pass_VV2Lep_MergLP_GGF_WZ_SR', 'Pass_VV2Lep_MergHP_GGF_WZ_ZCR', 'Pass_VV2Lep_MergLP_GGF_WZ_ZCR']},
                    'VBF': {'VBFRadion': ['Pass_VV2Lep_MergLP_VBF_ZZ_SR', 'Pass_VV2Lep_MergHP_VBF_ZZ_SR', 'Pass_VV2Lep_MergHP_VBF_ZZ_ZCR', 'Pass_VV2Lep_MergLP_VBF_ZZ_ZCR'],
                            'VBFRSG': ['Pass_VV2Lep_MergLP_VBF_ZZ_SR', 'Pass_VV2Lep_MergHP_VBF_ZZ_SR', 'Pass_VV2Lep_MergHP_VBF_ZZ_ZCR', 'Pass_VV2Lep_MergLP_VBF_ZZ_ZCR'],
                            'VBFHVTWZ': ['Pass_VV2Lep_MergHP_VBF_WZ_SR', 'Pass_VV2Lep_MergLP_VBF_WZ_SR', 'Pass_VV2Lep_MergHP_VBF_WZ_ZCR', 'Pass_VV2Lep_MergLP_VBF_WZ_ZCR']}
                },
            'resolved': {'ggF': {'Radion': ['Pass_VV2Lep_Res_GGF_ZZ_2btag_SR', 'Pass_VV2Lep_Res_GGF_ZZ_01btag_SR', 'Pass_VV2Lep_Res_GGF_ZZ_2btag_ZCR', 'Pass_VV2Lep_Res_GGF_ZZ_01btag_ZCR'],
                                 'RSG': ['Pass_VV2Lep_Res_GGF_ZZ_2btag_SR', 'Pass_VV2Lep_Res_GGF_ZZ_01btag_SR', 'Pass_VV2Lep_Res_GGF_ZZ_2btag_ZCR', 'Pass_VV2Lep_Res_GGF_ZZ_01btag_ZCR'],
                                 'HVTWZ': ['Pass_VV2Lep_Res_GGF_WZ_SR', 'Pass_VV2Lep_Res_GGF_WZ_ZCR']},
                         'VBF': {'VBFRadion': ['Pass_VV2Lep_Res_VBF_ZZ_SR', 'Pass_VV2Lep_Res_VBF_ZZ_ZCR'],
                                 'VBFRSG': ['Pass_VV2Lep_Res_VBF_ZZ_SR', 'Pass_VV2Lep_Res_VBF_ZZ_ZCR'],
                                 'VBFHVTWZ': ['Pass_VV2Lep_Res_VBF_WZ_SR', 'Pass_VV2Lep_Res_VBF_WZ_ZCR']}
                     }
}
'''
### Saving number of events for each origin
for origin in inputOrigins:
    logFile.write('\nNumber of ' + origin + ' events: ' + str(dataFrame[dataFrame['origin'] == origin].shape[0]) + ' (raw), ' + str(sum(dataFrame[dataFrame['origin'] == origin]['weight'])) +' (with MC weights)')
    cprint('Number of ' + origin + ' events: ' + str(dataFrame[dataFrame['origin'] == origin].shape[0]) + ' (raw), ' + str(sum(dataFrame[dataFrame['origin'] == origin]['weight'])) +' (with MC weights)', 'blue')
    dataFrameOrigin = dataFrame[dataFrame['origin'] == origin]
    for passVar in passDict[analysis][channel]: #passDict[analysis][channel][signal]:
        dataSetSingleRegion = dataFrameOrigin[dataFrameOrigin[passVar] == True]
        cprint('------- Number of events in region ' + passVar + ': ' + str(dataSetSingleRegion.shape[0]) + ' (raw), ' + str(sum(dataSetSingleRegion['weight'])) + ' (with MC weights)', 'blue')
        logFile.write('\n------ Number of events in region ' + passVar + ': ' + str(dataSetSingleRegion.shape[0]) + ' (raw), ' + str(sum(dataSetSingleRegion['weight'])) + ' (with MC weights)')

### Computing derived variables
dataFrame = computeDerivedVariables(variablesToDerive, dataFrame, signal, analysis)

### Selecting in the dataframe only the variables relevant for the next steps
dataFrame = dataFrame[variablesToSave + variablesToDerive]

### Removing events with high absoulte MC weights
meanWeight = dataFrame['weight'].mean()
stdWeight = dataFrame['weight'].std()
selectionString = 'abs(MCweight - meanMCweight) <= 5 * std(MCweight)'
cprint('Applying cut \'' + selectionString + '\'', 'white', 'on_green')
selection = 'abs(weight - ' + str(meanWeight) + ') <= 5 * ' + str(stdWeight)
dataFrame = dataFrame.query(selection)

### Shuffling the dataframe
dataFrame = ShufflingData(dataFrame)

### Saving number of events for each origin after cutting on weight
logFile.write('\nNumber of events after applying cut \'' + selectionString + '\':')
cprint('Number of events after applying the cut', 'blue')
for origin in inputOrigins:
    logFile.write('\nNumber of ' + origin + ' events: ' + str(dataFrame[dataFrame['origin'] == origin].shape[0]) + ' (raw), ' + str(sum(dataFrame[dataFrame['origin'] == origin]['weight'])) +' (with MC weights)')
    cprint('Number of ' + origin + ' events: ' + str(dataFrame[dataFrame['origin'] == origin].shape[0]) + ' (raw), ' + str(sum(dataFrame[dataFrame['origin'] == origin]['weight'])) +' (with MC weights)', 'blue')

### Saving the combined dataframe
outputFileName = '/MixData_' + fileCommonName + '.pkl'
#dataFrame.to_pickle(outputDir + outputFileName)
cprint('Saved ' + outputDir + outputFileName, 'green')
logFile.write('\nSaved combined (signal and background) dataframe in ' + outputDir + outputFileName)

### Closing the log file
logFile.close()
cprint('Saved ' + logFileName, 'green')

### Drawing histogram of variables
if drawPlots:
    histoOutputDir = outputDir + '/trainTestHistograms'
    checkCreateDir(histoOutputDir)
    DrawVariablesHisto(dataFrame, histoOutputDir, fileCommonName, analysis, channel, signal, backgroundLegend, preselectionCuts, backgroundsList, False)
    #DrawCorrelationMatrix(dataFrame, InputFeatures, outputDir, fileCommonName, analysis, channel, signal, backgroundLegend)
