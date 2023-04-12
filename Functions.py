### Assigning script names to variables
fileNameSaveToPkl = 'saveToPkl.py'
fileNameBuildDataSet = 'buildDataset.py'
fileNameComputeSignificance = 'computeSignificance.py'#'computeSignificanceScores.py' #'computeSignificance.py'#New.py' ##2
fileNameSplitDataSet = 'splitDataset.py'
fileNameBuildDNN = 'buildDNN.py'
#fileNameBuildPDNN = 'buildPDNNtuningHyp.py'
fileNameBuildPDNN = 'buildPDNN.py'#'buildPDNNscores.py'#'buildPDNN.py'
fileName6 = 'tuningHyperparameters.py'
fileNamePlots = 'tests/drawPlots.py'
fileNameCreateScoresBranch = 'addScoreBranch.py'#'createScoresBranch.py'

### Reading the command line
from argparse import ArgumentParser
import sys
from termcolor import colored, cprint

def ReadArgParser():
    parser = ArgumentParser(add_help = False)
    parser.add_argument('-a', '--Analysis', help = 'Type of analysis: \'merged\' or \'resolved\'', type = str)
    parser.add_argument('-c', '--Channel', help = 'Channel: \'ggF\' or \'VBF\'', type = str)
    parser.add_argument('-s', '--Signal', help = 'Signal: \'VBFHVTWZ\', \'Radion\', \'RSG\', \'VBFRSG\', \'HVTWZ\' or \'VBFRadion\'', type = str)
    #parser.add_argument('-j', '--JetCollection', help = 'Jet collection: \'TCC\', \'UFO_PFLOW\'', type = str, default = 'UFO_PFLOW')
    parser.add_argument('-b', '--Background', help = 'Background: \'Zjets\', \'Wjets\', \'stop\', \'Diboson\', \'ttbar\' or \'all\' (in quotation mark separated by a space)', type = str, default = 'all')
    parser.add_argument('-t', '--TrainingFraction', help = 'Relative size of the training sample, between 0 and 1', default = 0.8)
    parser.add_argument('-p', '--PreselectionCuts', help = 'Preselection cut', type = str)
    parser.add_argument('-h', '--hpOptimization', help = 'If 1 hyperparameters optimization will be performed', default = 0)
    parser.add_argument('-n', '--Nodes', help = 'Number of nodes of the (p)DNN, should always be >= nColumns and strictly positive', default = 48)#128) #32
    parser.add_argument('-l', '--Layers', help = 'Number of hidden layers of the (p)DNN', default = 2)#4) #2
    parser.add_argument('-e', '--Epochs', help = 'Number of maximum epochs for the training', default = 200) #150
    parser.add_argument('-v', '--Validation', help = 'Fraction of the training data that will actually be used for validation', default = 0.2)
    parser.add_argument('-d', '--Dropout', help = 'Fraction of the neurons to drop during the training', default = 0.2)
    parser.add_argument('-m', '--Mass', help = 'Masses for the (P)DNN train/test (GeV, in quotation mark separated by a space)', default = 'all')
    parser.add_argument('--doTrain', help = 'If 1 the training will be performed, if 0 it won\'t', default = 1)
    parser.add_argument('--doTest', help = 'If 1 the test will be performed, if 0 it won\'t', default = 1)
    parser.add_argument('--loop', help = 'How many times the code will be executed', default = 1) #20
    parser.add_argument('--tag', help = 'CxAOD tag', default = 'r33-24')
    parser.add_argument('--drawPlots', help = 'If 1 all plots will be saved', default = 0)
    parser.add_argument('-r', '--regime', help = '')
    parser.add_argument('-f', '--FeatureToPlot', help = 'Feature to plot to compute significance', default = 'score')
    parser.add_argument('--trainSet', help = 'trainSet')
    parser.add_argument('--doStudyLRpatience', default = 0)
    parser.add_argument('--config', default = 0)

    args = parser.parse_args()

    analysis = args.Analysis
    if args.Analysis is None and fileNameSaveToPkl not in sys.argv[0] and fileNameComputeSignificance not in sys.argv[0] and 'computeSignificance_2.py' not in sys.argv[0]:
        parser.error('Requested type of analysis (either \'merged\' or \'resolved\')')
    elif args.Analysis and analysis != 'resolved' and analysis != 'merged':
        parser.error('Analysis can be either \'merged\' or \'resolved\'')
    channel = args.Channel
    if args.Channel is None and fileNameSaveToPkl not in sys.argv[0] and fileNameComputeSignificance not in sys.argv[0] and 'computeSignificance_2.py' not in sys.argv[0]:
        parser.error('Requested channel (either \'ggF\' or \'VBF\')')
    elif args.Channel and channel != 'ggF' and channel != 'VBF':
        parser.error('Channel can be either \'ggF\' or \'VBF\'')
    signal = args.Signal
    if args.Signal is None and fileNameSaveToPkl not in sys.argv[0]:
        parser.error('Requested type of signal (\'Radion\', \'RSG\', \'HVTWZ\')')
    if args.Signal and signal != 'Radion' and signal != 'RSG' and signal != 'HVTWZ':
        parser.error('Signal can be only \'Radion\', \'RSG\' or \'HVTWZ\'')
    if args.Channel and channel == 'VBF':
        signal = channel + signal
    '''
    jetCollection = args.JetCollection
    if args.JetCollection is None:
        parser.error('Requested jet collection (\'TCC\' or \'UFO_PFLOW\')')
    elif args.JetCollection != 'TCC' and args.JetCollection != 'UFO_PFLOW':
        parser.error('Jet collection can be \'TCC\', \'UFO_PFLOW\'')
    '''
    background = args.Background.split()
    for bkg in background:
        if (bkg !=  'Zjets' and bkg != 'Wjets' and bkg != 'stop' and bkg != 'Diboson' and bkg != 'ttbar' and bkg != 'all'):
            parser.error('Background can be \'Zjets\', \'Wjets\', \'stop\', \'Diboson\', \'ttbar\' or \'all\'')
    backgroundString = 'all'
    if args.Background != 'all':
        backgroundString = '_'.join([str(item) for item in background])
    trainingFraction = float(args.TrainingFraction)
    if args.TrainingFraction and (trainingFraction < 0. or trainingFraction > 1.):
        parser.error('Training fraction must be between 0 and 1')
    preselectionCuts = args.PreselectionCuts
    if args.PreselectionCuts is None:
        preselectionCuts = 'none'
    hpOptimization = bool(int(args.hpOptimization))
    numberOfNodes = int(args.Nodes)
    if args.Nodes and numberOfNodes < 1:
        parser.error('Number of nodes must be integer and strictly positive')
    numberOfLayers = int(args.Layers)
    if args.Layers and numberOfLayers < 1:
        parser.error('Number of layers must be integer and strictly positive')
    numberOfEpochs = int(args.Epochs)
    if args.Epochs and numberOfEpochs < 1:
        parser.error('Number of maximum epochs must be integer and strictly positive')
    validationFraction = float(args.Validation)
    if args.Validation and (validationFraction < 0. or validationFraction > 1.):
        parser.error('Validation fraction must be between 0 and 1')
    dropout = float(args.Dropout)
    if args.Dropout and (dropout < 0. or dropout > 1.):
        parser.error('Dropout must be between 0 and 1')
    mass = args.Mass.split()
    doTrain = bool(int(args.doTrain))
    if args.doTrain and (doTrain != 0 and doTrain != 1):
        parser.error('doTrain can only be 1 (to perform the training) or 0')
    doTest = bool(int(args.doTest))
    if args.doTest and (doTest != 0 and doTest != 1):
        parser.error('doTest can only be 1 (to perform the test) or 0')
    loop = int(args.loop)
    trainSet = args.trainSet
    tag = args.tag
    drawPlots = args.drawPlots
    regime = args.regime#.split()
    if args.regime:
        regime = regime.split()
    feature = args.FeatureToPlot
    doStudyLRpatience = bool(int(args.doStudyLRpatience))
    configurationFile = args.config

    if fileNameSaveToPkl in sys.argv[0]:
        print('           tag = ' + tag)
        #print('jet collection = ' + jetCollection)
        return tag

    if fileNameBuildDataSet in sys.argv[0]:
        print('                    tag = ' + tag)
        #print('         jet collection = ' + jetCollection)
        print('          background(s) = ' + str(backgroundString))
        print('              drawPlots = ' + str(drawPlots))
        print('                 signal = ' + str(signal))
        return tag, analysis, channel, preselectionCuts, signal, backgroundString, drawPlots

    if fileNameSplitDataSet in sys.argv[0] or fileNamePlots in sys.argv[0]:
        print('              tag = ' + tag)
        #print('   jet collection = ' + jetCollection)
        print('       background = ' + str(backgroundString))
        print('training fraction = ' + str(trainingFraction))
        print('        drawPlots = ' + str(drawPlots))
        return tag, analysis, channel, preselectionCuts, backgroundString, signal, trainingFraction, drawPlots

    if fileNameBuildDNN in sys.argv[0] or fileNameBuildPDNN in sys.argv[0]:# or sys.argv[0] == fileName6):
        print('               background(s) = ' + str(backgroundString))
        print('               test mass(es) = ' + str(mass))
        print('           training fraction = ' + str(trainingFraction))
        print('hyperparameters optimization = ' + str(hpOptimization))
        print('                     doTrain = ' + str(doTrain))
        print('                      doTest = ' + str(doTest))
        print('         validation fraction = ' + str(validationFraction))
        print('    number of maximum epochs = ' + str(numberOfEpochs))
        print('                    trainSet = ' + str(trainSet))
        print('          configuration file = ' + str(configurationFile))
        if not hpOptimization:
            print('             number of nodes = ' + str(numberOfNodes))
            print('     number of hidden layers = ' + str(numberOfLayers))
            print('                     dropout = ' + str(dropout))
        return tag, analysis, channel, preselectionCuts, backgroundString, trainingFraction, signal, numberOfNodes, numberOfLayers, numberOfEpochs, validationFraction, dropout, mass, doTrain, doTest, loop, hpOptimization, drawPlots, trainSet, doStudyLRpatience, configurationFile

    if fileNameComputeSignificance in sys.argv[0] or 'computeSignificance_2.py' in sys.argv[0]:
        return tag, regime, preselectionCuts, signal, backgroundString

    if fileNameCreateScoresBranch in sys.argv[0]:
        print('          background(s) = ' + str(backgroundString))
        print('                 signal = ' + str(signal))
        return tag, analysis, channel, preselectionCuts, signal, backgroundString

### Reading from the configuration file
import configparser, ast
import shutil

def ReadConfigSaveToPkl(tag):#, jetCollection):
    #configurationFile = 'Configuration_' + jetCollection + '_' + tag + '.ini'
    configurationFile = 'Configuration_' + tag + '.ini'
    print('Reading configuration file: ' + configurationFile)
    config = configparser.ConfigParser()
    config.read(configurationFile)
    ntuplePath = config.get('config', 'ntuplePath')
    inputFiles = ast.literal_eval(config.get('config', 'inputFiles'))
    dfPath = config.get('config', 'dfPath')
    dfPath += tag + '/'# + jetCollection + '/'
    rootBranchSubSample = ast.literal_eval(config.get('config', 'rootBranchSubSample'))
    #print (format('Output directory: ' + dfPath), checkCreateDir(dfPath))
    #shutil.copyfile(configurationFile, dfPath + configurationFile)
    return ntuplePath, inputFiles, dfPath, rootBranchSubSample

def ReadConfig(tag, analysis, signal):
#def ReadConfig(tag, analysis, signal, configFile):
    configurationFile = 'Configuration_' + tag + '.ini'
    #configurationFile = configFile
    print('Reading configuration file: ' + configurationFile)
    config = configparser.ConfigParser()
    config.read(configurationFile)
    inputFiles = ast.literal_eval(config.get('config', 'inputFiles'))
    ntuplePath = config.get('config', 'ntuplePath')
    rootBranchSubSample = ast.literal_eval(config.get('config', 'rootBranchSubSample'))
    signalsList = ast.literal_eval(config.get('config', 'signals'))
    backgroundsList = ast.literal_eval(config.get('config', 'backgrounds'))
    dfPath = config.get('config', 'dfPath')
    #dfPath += tag + '/' + jetCollection + '/'
    dfPath += tag + '/'
    if analysis == 'merged':
        InputFeatures = ast.literal_eval(config.get('config', 'inputFeaturesMerged'))
        variablesToDerive = ast.literal_eval(config.get('config', 'variablesToDeriveMerged'))
        variablesToSave = ast.literal_eval(config.get('config', 'variablesToSaveMerged'))
    elif analysis == 'resolved':
        #if signal == 'Radion' or signal == 'RSG':
        if 'Radion' in signal or 'RSG' in signal:
            InputFeatures = ast.literal_eval(config.get('config', 'inputFeaturesResolvedRadionRSG'))
            variablesToDerive = ast.literal_eval(config.get('config', 'variablesToDeriveResolvedRadionRSG'))
            variablesToSave = ast.literal_eval(config.get('config', 'variablesToSaveResolvedRadionRSG'))
        else:
            InputFeatures = ast.literal_eval(config.get('config', 'inputFeaturesResolvedHVT'))
            variablesToDerive = ast.literal_eval(config.get('config', 'variablesToDeriveResolvedHVT'))
            variablesToSave = ast.literal_eval(config.get('config', 'variablesToSaveResolvedHVT'))
    if fileNameBuildDataSet in sys.argv[0]:
        return ntuplePath, InputFeatures, dfPath, variablesToSave, variablesToDerive, backgroundsList
    if fileNameComputeSignificance in sys.argv[0] or fileNameCreateScoresBranch in sys.argv[0] or 'computeSignificance_2.py' in sys.argv[0]:
        return inputFiles, rootBranchSubSample, dfPath, variablesToSave, backgroundsList
    if fileNamePlots in sys.argv[0]:
        return dfPath, InputFeatures
    if fileNameSplitDataSet in sys.argv[0]:
        return dfPath, signalsList, backgroundsList
    if fileNameBuildDNN in sys.argv[0] or fileNameBuildPDNN in sys.argv[0] or fileName6 in sys.argv[0]:
        return ntuplePath, dfPath, InputFeatures

### Checking if the output directory exists. If not, creating it
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'} ---> '3' to suppress INFO, WARNING, and ERROR messages in Tensorflow

def checkCreateDir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)
        return ' (created)'
    else:
        return ' (already there)'

### Functions to enable or disable 'print' calls
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
def enablePrint():
    sys.stdout = sys.__stdout__

### Loading input data
import pandas as pd
import numpy as np

#def LoadData(dfPath, tag, jetCollection, signal, analysis, channel, background, trainingFraction, preselectionCuts, InputFeatures, iLoop):
def LoadData(dfPath, tag, signal, analysis, channel, background, trainingFraction, preselectionCuts, InputFeatures, iLoop):
    fileCommonName = tag + '_' + analysis + '_' + channel + '_' + preselectionCuts + '_' + str(signal) + '_' + background + '_' + str(trainingFraction) + 't'
    print('Loading ' + dfPath + 'data_train_' + fileCommonName + '.pkl')
    dataTrain = pd.read_pickle(dfPath + '/data_train_' + fileCommonName + '.pkl')
    #dataTrain = dataTrain.query('Pass_VV2Lep_MergHP_GGF_WZ_SR == True or Pass_VV2Lep_MergLP_GGF_WZ_SR == True or Pass_VV2Lep_MergHP_VBF_WZ_SR == True or Pass_VV2Lep_MergLP_VBF_WZ_SR == True')
    #print('# of events in dataTrain: ' + str(len(dataTrain)))
    '''
    passCols = ['Pass_VV2Lep_MergHP_GGF_ZZ_Tag_ZCR', 'Pass_VV2Lep_MergHP_GGF_ZZ_Untag_ZCR', 'Pass_VV2Lep_MergLP_GGF_ZZ_Tag_ZCR', 'Pass_VV2Lep_MergLP_GGF_ZZ_Untag_ZCR', 'Pass_VV2Lep_MergHP_GGF_WZ_ZCR', 'Pass_VV2Lep_MergLP_GGF_WZ_ZCR', 'Pass_VV2Lep_MergHP_VBF_WZ_ZCR', 'Pass_VV2Lep_MergHP_VBF_ZZ_ZCR', 'Pass_VV2Lep_MergLP_VBF_WZ_ZCR', 'Pass_VV2Lep_MergLP_VBF_ZZ_ZCR']
    for col in passCols:
        dataTrain[col].replace(to_replace = [False, True], value = [0, 1], inplace = True)
        sumCol = dataTrain[col].sum()
        print(col + ' -> ' + str(sumCol))
    exit()
    '''
    #X_train = np.array(dataTrain[InputFeatures].values).astype(np.float32)
    #y_train = np.array(dataTrain['isSignal'].values).astype(np.float32)
    #w_train = dataTrain['train_weight'].values
    print('Loading ' + dfPath + 'data_test_' + fileCommonName + '.pkl')
    dataTest = pd.read_pickle(dfPath + '/data_test_' + fileCommonName + '.pkl')
    #X_test = np.array(dataTest[InputFeatures].values).astype(np.float32)
    #y_test = np.array(dataTest['isSignal'].values).astype(np.float32)
    #w_test = dataTest['train_weight'].values

    return dataTrain, dataTest#, X_train, X_test, y_train, y_test, w_train, w_test


### Writing in the log file
def WriteLogFile(tag, ntuplePath, InputFeatures, dfPath, hpOptimization, doTrain, doTest, validationFraction, batchSize):
    logString = 'CxAOD tag: ' + tag + '\nntuple path: ' + ntuplePath + '\nInputFeatures: ' + str(InputFeatures) + '\ndfPath: ' + dfPath + '\nHyperparameters optimization: ' + str(hpOptimization) + '\ndoTrain: ' + str(doTrain) + '\ndoTest: ' + str(doTest) + '\nValidation fraction: ' + str(validationFraction) + '\nBatch size: ' + str(batchSize)# + '\nPatience value: ' + str(patienceValue)# + '\nNumber of train events: ' + str(len(data_train)) + ' (' + str(len(data_train_signal)) + ' signal and ' + str(len(data_train_bkg)) + ' background)' + '\nNumber of test events: ' + str(len(data_test)) + ' (' + str(len(data_test_signal)) + ' signal and ' + str(len(data_test_bkg)) + ' background)'
    return logString

def SelectEvents(dataFrame, channel, analysis, preselectionCuts, signal):
    ### Selecting events according to type of analysis and channel
    selectionMergedGGF = 'Pass_VV2Lep_MergHP_GGF_ZZ_2btag_SR == True or Pass_VV2Lep_MergHP_GGF_ZZ_01btag_SR == True or Pass_VV2Lep_MergHP_GGF_WZ_SR == True or Pass_VV2Lep_MergLP_GGF_ZZ_2btag_SR == True or Pass_VV2Lep_MergLP_GGF_ZZ_01btag_SR == True or Pass_VV2Lep_MergLP_GGF_WZ_SR == True or Pass_VV2Lep_MergHP_GGF_ZZ_2btag_ZCR == True or Pass_VV2Lep_MergHP_GGF_ZZ_01btag_ZCR == True or Pass_VV2Lep_MergHP_GGF_WZ_ZCR == True or Pass_VV2Lep_MergLP_GGF_ZZ_2btag_ZCR == True or Pass_VV2Lep_MergLP_GGF_ZZ_01btag_ZCR == True or Pass_VV2Lep_MergLP_GGF_WZ_ZCR == True'# or Pass_VV2Lep_MergHP_GGF_ZZ_01btag_TCR == True or Pass_VV2Lep_MergHP_GGF_ZZ_2btag_TCR == True or Pass_VV2Lep_MergHP_GGF_WZ_TCR == True or Pass_VV2Lep_MergLP_GGF_ZZ_01btag_TCR == True or Pass_VV2Lep_MergLP_GGF_ZZ_2btag_TCR == True or Pass_VV2Lep_MergLP_GGF_WZ_TCR == True'
    selectionMergedGGFZZLPuntagSR = 'Pass_VV2Lep_MergLP_GGF_ZZ_01btag_SR == True and Pass_VV2Lep_MergHP_GGF_ZZ_2btag_SR == False and Pass_VV2Lep_MergHP_GGF_ZZ_01btag_SR == False and Pass_VV2Lep_MergHP_GGF_WZ_SR == False and Pass_VV2Lep_MergLP_GGF_ZZ_2btag_SR == False and Pass_VV2Lep_MergHP_GGF_ZZ_2btag_ZCR == False and Pass_VV2Lep_MergHP_GGF_WZ_ZCR == False and Pass_VV2Lep_MergHP_GGF_ZZ_01btag_ZCR == False and Pass_VV2Lep_MergLP_GGF_ZZ_2btag_ZCR == False and Pass_VV2Lep_MergLP_GGF_ZZ_01btag_ZCR == False and Pass_VV2Lep_MergLP_GGF_WZ_SR == False and Pass_VV2Lep_MergLP_GGF_WZ_ZCR == False'
    selectionMergedVBF = 'Pass_VV2Lep_MergHP_VBF_WZ_SR == True or Pass_VV2Lep_MergHP_VBF_ZZ_SR == True or Pass_VV2Lep_MergLP_VBF_WZ_SR == True or Pass_VV2Lep_MergLP_VBF_ZZ_SR == True or Pass_VV2Lep_MergHP_VBF_WZ_ZCR == True or Pass_VV2Lep_MergHP_VBF_ZZ_ZCR == True or Pass_VV2Lep_MergLP_VBF_WZ_ZCR == True or Pass_VV2Lep_MergLP_VBF_ZZ_ZCR == True'# or Pass_VV2Lep_MergHP_VBF_WZ_TCR == True or Pass_VV2Lep_MergHP_VBF_ZZ_TCR == True or Pass_VV2Lep_MergLP_VBF_WZ_TCR == True or Pass_VV2Lep_MergLP_VBF_ZZ_TCR == True'
    #selectionMergedVBF = 'Pass_VV2Lep_MergHP_VBF_ZZ_SR == True or Pass_VV2Lep_MergLP_VBF_ZZ_SR == True or Pass_VV2Lep_MergHP_VBF_ZZ_ZCR == True or Pass_VV2Lep_MergLP_VBF_ZZ_ZCR == True'# or Pass_VV2Lep_MergHP_VBF_WZ_TCR == True or Pass_VV2Lep_MergHP_VBF_ZZ_TCR == True or Pass_VV2Lep_MergLP_VBF_WZ_TCR == True or Pass_VV2Lep_MergLP_VBF_ZZ_TCR == True'
    selectionResolvedGGF = 'Pass_VV2Lep_Res_GGF_WZ_SR == True or Pass_VV2Lep_Res_GGF_ZZ_2btag_SR == True or Pass_VV2Lep_Res_GGF_ZZ_01btag_SR == True or Pass_VV2Lep_Res_GGF_WZ_ZCR == True or Pass_VV2Lep_Res_GGF_ZZ_2btag_ZCR == True or Pass_VV2Lep_Res_GGF_ZZ_01btag_ZCR == True'# or Pass_VV2Lep_Res_GGF_WZ_TCR == True or Pass_VV2Lep_Res_GGF_ZZ_2btag_TCR == True or Pass_VV2Lep_Res_GGF_ZZ_01btag_TCR == True'
    selectionResolvedVBF = 'Pass_VV2Lep_Res_VBF_WZ_SR == True or Pass_VV2Lep_Res_VBF_ZZ_SR == True or Pass_VV2Lep_Res_VBF_WZ_ZCR == True or Pass_VV2Lep_Res_VBF_ZZ_ZCR == True'# or Pass_VV2Lep_Res_VBF_WZ_TCR == True or Pass_VV2Lep_Res_VBF_ZZ_TCR == True'
    selectionResolved = 'Pass_VV2Lep_Res_GGF_WZ_SR == True or Pass_VV2Lep_Res_GGF_ZZ_2btag_SR == True or Pass_VV2Lep_Res_GGF_ZZ_01btag_SR == True or Pass_VV2Lep_Res_GGF_WZ_ZCR == True or Pass_VV2Lep_Res_GGF_ZZ_2btag_ZCR == True or Pass_VV2Lep_Res_GGF_ZZ_01btag_ZCR == True or Pass_VV2Lep_Res_VBF_WZ_SR == True or Pass_VV2Lep_Res_VBF_ZZ_SR == True or Pass_VV2Lep_Res_VBF_WZ_ZCR == True or Pass_VV2Lep_Res_VBF_ZZ_ZCR == True'

    if channel == 'ggF':
        dataFrame = dataFrame.query('Pass_isVBFVV == False')
        if analysis == 'merged':
            selection = selectionMergedGGF
            #selection = selectionMergedGGFZZLPuntagSR            
        elif analysis == 'resolved':
            selection = selectionResolvedGGF
            #selection = selectionResolved

    elif channel == 'VBF':
        dataFrame = dataFrame.query('Pass_isVBFVV == True')
        if analysis == 'merged':
            selection = selectionMergedVBF
        elif analysis == 'resolved':
            selection = selectionResolvedVBF
    dataFrame = dataFrame.query(selection)

    ### Applying preselection cuts (if any)
    if preselectionCuts != 'none':
        #dataFrame = dataFrame.query(preselectionCuts)
        if preselectionCuts == 'looseEventsSelection':
            if 'HVT' in signal:
                print('Loose events selection for HVT')
                dataFrame = dataFrame.query('Pass_VV2Lep_SFLeptons == True and Pass_VV2Lep_Trigger == True and Pass_VV2Lep_FatJet == True and fatjet_pt > 200 and lep1_pt > 30 and lep2_pt > 30 and Pass_VV2Lep_WTaggerSubStructCutLP == True')
            else:
                print('Loose events selection for Radion and RSG')
                dataFrame = dataFrame.query('Pass_VV2Lep_SFLeptons == True and Pass_VV2Lep_Trigger == True and Pass_VV2Lep_FatJet == True and fatjet_pt > 200 and lep1_pt > 30 and lep2_pt > 30 and Pass_VV2Lep_ZTaggerSubStructCutLP == True')

    return dataFrame

def SelectRegime(dataFrame, preselectionCuts, regime, channel, signal):
    ### Selecting events according to the regime
    if regime == 'allMergedGGF':
        if 'HVT' not in signal:
            dataFrame = dataFrame.query('Pass_VV2Lep_MergHP_GGF_ZZ_2btag_SR == True or Pass_VV2Lep_MergLP_GGF_ZZ_2btag_SR == True or Pass_VV2Lep_MergHP_GGF_ZZ_01btag_SR == True or Pass_VV2Lep_MergLP_GGF_ZZ_01btag_SR == True')
        else:
            dataFrame = dataFrame.query('Pass_VV2Lep_MergHP_GGF_WZ_SR == True or Pass_VV2Lep_MergLP_GGF_WZ_SR == True')
    if regime == 'allMergedVBF':
        if 'HVT' not in signal:
            dataFrame = dataFrame.query('Pass_VV2Lep_MergHP_VBF_ZZ_SR == True or Pass_VV2Lep_MergLP_VBF_ZZ_SR == True')
        else:
            dataFrame = dataFrame.query('Pass_VV2Lep_MergLP_VBF_WZ_SR == True or Pass_VV2Lep_MergHP_VBF_WZ_SR == True')

    elif regime == 'allResolvedGGF':
        if 'HVTWZ' not in signal:
            dataFrame = dataFrame.query('Pass_VV2Lep_Res_GGF_ZZ_01btag_SR == True or Pass_VV2Lep_Res_GGF_ZZ_2btag_SR == True')
        else:
            print('ERROR')
    '''
    if regime == 'allMergedZCRs':
        dataFrame = dataFrame.query('Pass_isVBFVV == ' + isVBF)
        selectionMergedGGF = 'Pass_VV2Lep_MergHP_GGF_ZZ_2btag_SR == True or Pass_VV2Lep_MergHP_GGF_ZZ_01btag_SR == True or Pass_VV2Lep_MergHP_GGF_WZ_SR == True or Pass_VV2Lep_MergLP_GGF_ZZ_2btag_SR == True or Pass_VV2Lep_MergLP_GGF_ZZ_01btag_SR == True or Pass_VV2Lep_MergLP_GGF_WZ_SR == True or Pass_VV2Lep_MergHP_GGF_ZZ_2btag_ZCR == True or Pass_VV2Lep_MergHP_GGF_ZZ_01btag_ZCR == True or Pass_VV2Lep_MergHP_GGF_WZ_ZCR == True or Pass_VV2Lep_MergLP_GGF_ZZ_2btag_ZCR == True or Pass_VV2Lep_MergLP_GGF_ZZ_01btag_ZCR == True or Pass_VV2Lep_MergLP_GGF_WZ_ZCR == True'# or Pass_VV2Lep_MergHP_GGF_ZZ_01btag_TCR == True or Pass_VV2Lep_MergHP_GGF_ZZ_2btag_TCR == True or Pass_VV2Lep_MergHP_GGF_WZ_TCR == True or Pass_VV2Lep_MergLP_GGF_ZZ_01btag_TCR == True or Pass_VV2Lep_MergLP_GGF_ZZ_2btag_TCR == True or Pass_VV2Lep_MergLP_GGF_WZ_TCR == True'
        dataFrame = dataFrame.query(selectionMergedGGF)
    '''

    ### Applying preselection cuts (if any)
    if preselectionCuts != 'none':
        dataFrame = dataFrame.query(preselectionCuts)

    return dataFrame

### Selecting signal events according to their mass and type of analysis
def CutMasses(dataFrame, analysis):
    if analysis == 'merged':
        dataFrame = dataFrame.query('mass >= 500')
    elif analysis == 'resolved':
        dataFrame = dataFrame.query('mass <= 1500')
    return dataFrame

### Shuffling dataframe
import sklearn.utils

def ShufflingData(dataFrame):
    dataFrame = sklearn.utils.shuffle(dataFrame, random_state = 123)
    #dataFrame = dataFrame.reset_index(drop = True)
    return dataFrame

### Drawing histograms of each variables in the dataframe divided by class
import seaborn
import matplotlib
import matplotlib.pyplot as plt
plt.ioff()
plt.rcParams["figure.figsize"] = [7,7]
plt.rcParams.update({'font.size': 16})

def integral(y,x,bins):
    x_min=x
    s=0
    for i in np.where(bins>x)[0][:-1]:
        s=s+y[i]*(bins[i+1]-bins[i])
    return s

import ROOT
from ROOT import TCanvas, TH1F, TLatex, gPad, gStyle, TLegend, THStack

def ATLASLabel(x,y,text,color=1):
    """
    An ATLAS label
    """
    l = TLatex()
    l.SetNDC()
    l.SetTextFont(72)
    l.SetTextColor(color)
    l.SetTextSize(0.06)
    delx = 0.115*600*gPad.GetWh()/(330*gPad.GetWw())
    l.DrawLatex(x,y,"ATLAS")
    if text:
        p = TLatex()
        p.SetNDC()
        p.SetTextSize(0.04)
        p.SetTextFont(42)
        p.SetTextColor(color)
        p.DrawLatex(x+delx,y,text)

def CustomLabel(x,y,text, color = 1):
    """
    A custom label
    """
    p = TLatex()
    p.SetNDC()
    #p.SetTextSize(0.06)
    p.SetTextSize(0.04)
    p.SetTextFont(42)
    p.SetTextColor(color)
    p.DrawLatex(x,y,text)

def DrawVariablesHisto(dataFrame, outputDir, outputFileCommonName, analysis, channel, signal, background, preselectionCuts, backgroundsList, scaled = False): ### serve background?

    dataFrameSignal = dataFrame[dataFrame['origin'] == signal]
    dataFrameBkg = dataFrame[dataFrame['origin'] != signal]
    dataFrameBkgDict = {}
    for background in backgroundsList:
        dataFrameBkgDict[background] = dataFrame[dataFrame['origin'] == background]
    if 'VBF' in signal:
        signalLabel = signal.replace('VBF', '')
        dataFrame['origin'].replace(to_replace = [signal], value = [signalLabel], inplace = True)
    else:
        signalLabel = signal
    featureLogX = ['fatjet_D2', 'fatjet_m', 'fatjet_pt', 'lep1_pt', 'lep2_pt', 'Zcand_pt']
    legendText = 'analysis: ' + analysis + '\nchannel: ' + channel + '\nsignal: ' + signal + '\nbackground: ' + ', '.join(background)
    if (preselectionCuts != 'none'):
        legendText += '\npreselection cuts: ' + preselectionCuts

    labelsDict = {'rnnVJ1_pt': r'RNN jet_{1} p_{T}', 'rnnVJ2_pt': r'RNN jet_{2} p_{T}', 'rnnVJ1_eta': r'RNN jet_{1} #eta', 'rnnVJ2_eta': r'RNN jet_{2} #eta', 'rnnVJ1_phi': r'RNN jet_{1} #phi', 'rnnVJ2_phi': r'RNN jet_{2} #phi', 'rnnVJ1_e': r'RNN jet_{1} E [GeV]', 'rnnVJ2_e': r'RNN jet_{2} E [GeV]', 'lep1_pt': r'lep_{1} p_{T} [GeV]', 'lep2_pt': r'lep_{2} p_{T} [GeV]', 'lep1_phi': r'lep_{1} #phi', 'lep2_phi': r'lep_{2} #phi', 'lep1_eta': r'lep_{1} #eta', 'lep2_eta': r'lep_{2} #eta', 'NJets': '# of jets', 'NLargeRJets': r'# of large-#it{R} jets', 'fatjet_pt': r'large-#it{R} jet p_{T} [GeV]', 'fatjet_eta': r'large-#it{R} jet #eta', 'fatjet_phi': r'large-#it{R} jet #phi', 'fatjet_pt': r'large-#it{R} jet m [GeV]', 'fatjet_D2': r'large-#it{R} jet D2', 'sigVJ1_pt': r'signal jet_{1} p_{T} [GeV]', 'sigVJ1_eta': r'signal jet_{1} #eta', 'sigVJ1_phi': r'signal jet_{1} #phi', 'sigVJ2_pt': r'signal jet_{2} p_{T} [GeV]', 'sigVJ2_eta': r'signal jet_{2} #eta', 'sigVJ2_phi': r'signal jet_{2} #phi', 'lep1_m': r'lep_{1} m [GeV]', 'lep2_m': r'lep_{2} m [GeV]', 'Zcand_pt': r'Z_{cand} p_{T} [GeV]', 'Zcand_m': r'Z_{cand} m [GeV]'}

    binsDict = {'lep1_m': np.linspace(-0.03, 0.15, 6), 'lep1_pt': np.linspace(0, 2000, 51), 'lep1_eta': np.linspace(-3, 3, 51), 'lep1_phi': np.linspace(-3.5, 3.5, 51), 'lep2_m': np.linspace(-0.03, 0.15, 6), 'lep2_pt': np.linspace(0, 1200, 51), 'lep2_eta': np.linspace(-3, 3, 51), 'lep2_phi': np.linspace(-3.5, 3.5, 51), 'fatjet_m': np.linspace(0, 500, 51), 'fatjet_pt': np.linspace(0, 3000, 51), 'fatjet_eta': np.linspace(-3, 3, 51), 'fatjet_phi': np.linspace(-3.5, 3.5, 51), 'fatjet_D2': np.linspace(-0.5, 5.5, 51), 'Zcand_m': np.linspace(60, 130, 51), 'Zcand_pt': np.linspace(-20, 4000, 51), 'Zdijet_m': np.linspace(60, 140, 51), 'Zdijet_pt': 'auto', 'Zdijet_eta': np.linspace(-4, 4, 51), 'Zdijet_phi': np.linspace(-3.5, 3.5, 51), 'sigZJ1_m': np.linspace(0, 120, 51), 'sigZJ1_pt': np.linspace(0, 1000, 51), 'sigZJ1_eta': np.linspace(-3, 3, 51), 'sigZJ1_phi': np.linspace(-3.5, 3.5, 51), 'sigZJ2_m': np.linspace(0, 40, 51), 'sigZJ2_pt': np.linspace(0, 300, 51), 'sigZJ2_eta': np.linspace(-3, 3, 51), 'sigZJ2_phi': np.linspace(-3.5, 3.5, 51), 'DNNScore_W': np.linspace(0, 1, 51), 'DNNScore_Z': np.linspace(0, 1, 51), 'DNNScore_h': np.linspace(0, 1, 51), 'DNNScore_t': np.linspace(0, 1, 51), 'DNNScore_qg': np.linspace(0, 1, 51), 'delta_phi_lep12': np.linspace(-0.5, 3.5, 51), 'delta_phi_jetlep': np.linspace(-0.5, 3.5, 51), 'delta_phi_lepjet': np.linspace(-0.5, 3.5, 51), 'X_VV_merged': np.linspace(-50, 7000, 51)} # 'mass': 'auto', 'origin': 'auto',

    originsLabelsDict = {'Radion': 'ggF Radion', 'HVTWZ': 'DY HVT WZ', 'RSG': 'ggF RSG', 'Zjets': r'#it{Z} + jets', 'ttbar': r'#it{t#bar{t}}', 'Diboson': 'Diboson', 'Wjets': r'#it{W} + jets', 'stop': 'single top'}
    colorsDict = {'Zjets': ROOT.kGreen - 9, 'ttbar': ROOT.kBlue - 9, 'Diboson': ROOT.kGray, 'Wjets': ROOT.kMagenta + 1, 'stop': ROOT.kOrange}
    dictHisto = {}
    for feature in dataFrame.columns:
        if 'Pass' in feature or feature == 'train_weight' or feature == 'origin' or feature == 'weight' or feature == 'isSignal': ### fare quello per origin!:
            continue
        print(feature)
        statType = 'density'#'probability'
        #hueType = dataFrame['isSignal']
        hueType = dataFrameBkg['origin']#dataFrame['isSignal']
        legendBool = True
        if feature == 'origin' or feature == 'weight':
            statType = 'count'
            hueType = dataFrame['origin']
            legendBool = False

        dictHisto[feature] = {}

        Canvas = TCanvas(feature, feature, 800, 600)
        Canvas.cd()

        Stack = THStack()
        Legend = TLegend(.55, .75 - 4 * 0.025, .77, .85)
        Legend.SetBorderSize(0)
        Legend.SetTextFont(42)
        Legend.SetTextSize(0.04) # 0.045
        Legend.SetFillStyle(0)
        totalIntegral = 0
        minBin = min(min(dataFrameBkg[feature]), min(dataFrameSignal[feature]))
        maxBin = max(max(dataFrameBkg[feature]), max(dataFrameSignal[feature]))

        for background in backgroundsList:
            if feature in binsDict.keys():
                histo = TH1F('hist_' + feature + '_' + background, "", len(binsDict[feature]) + 1, binsDict[feature][0], binsDict[feature][len(binsDict[feature]) - 1])
            else:
                print('not in bin dictttttttttttttt')
                histo = TH1F('hist_' + feature + '_' + background, "", 50, minBin, maxBin)
            for value, weight in zip(dataFrameBkgDict[background][feature], dataFrameBkgDict[background]['weight']):
                histo.Fill(value, weight)
            dictHisto[feature][background] = histo
            totalIntegral += histo.Integral()

        for iBkg in reversed(range(len(backgroundsList))):
            bkg = backgroundsList[iBkg]
            dictHisto[feature][bkg].Scale(1 / totalIntegral)
            dictHisto[feature][bkg].SetLineWidth(2)
            #dictHisto[feature][bkg].SetLineColor(colorsDict[bkg])
            dictHisto[feature][bkg].SetFillColor(colorsDict[bkg])
            dictHisto[feature][bkg].SetLineWidth(2)
            Legend.AddEntry(dictHisto[feature][bkg], originsLabelsDict[bkg], 'f')
            Stack.Add(dictHisto[feature][bkg])

        Stack.Draw("HIST")

        if feature in binsDict.keys():
            histoSignal = TH1F('hist_' + feature + '_' + signal, "", len(binsDict[feature]) + 1, binsDict[feature][0], binsDict[feature][len(binsDict[feature]) - 1])
        else:
            print('not in bin dictttttttttttttt')
            histoSignal = TH1F('hist_' + feature + '_' + signal, "", 50, minBin, maxBin)
        for value, weight in zip(dataFrameSignal[feature], dataFrameSignal['weight']):
            histoSignal.Fill(value, weight)

        histoSignal.Scale(1 / histoSignal.Integral())
        histoSignal.SetLineColor(ROOT.kRed)
        histoSignal.SetLineWidth(2)
        Legend.AddEntry(histoSignal, originsLabelsDict[signal], 'l')
        histoSignal.Draw("HIST,SAME")

        Stack.SetMaximum(Stack.GetMaximum() + 2)

        if feature == 'weight' or feature == 'origin':
            Stack.GetYaxis().SetTitle('Counts')
        else:
            Stack.GetYaxis().SetTitle('Normalized yield')
        if feature in labelsDict:
            if scaled == True:
                Stack.GetXaxis().SetTitle('Scaled ' + labelsDict[feature])
            else:
                Stack.GetXaxis().SetTitle(labelsDict[feature])
        else:
            print('Not in dictttt')
            if scaled == True and feature != 'origin':
                Stack.GetXaxis().SetTitle('Scaled ' + feature)
            if scaled == False or feature == 'origin':
                Stack.GetXaxis().SetTitle(feature)
        Stack.SetTitle('SRs + ZCRs, ' + signalLabel + ', ' + analysis + ', ' + channel)

        ATLASLabel(0.15, 0.82, "Internal")
        CustomLabel(0.15, 0.74, "#sqrt{s} = 13 TeV, #int Ldt = 139 fb^{-1}")
        Legend.Draw("SAME")
        gStyle.SetOptStat(0)

        Canvas.SetLogy()
        pltName = outputDir + '/Histo_' + feature + '_' + outputFileCommonName + '.png'
        Canvas.SaveAs(pltName)
        cprint('Saved ' + pltName, 'green')
        Canvas.Close()


    dataFrame['isSignal'].replace(to_replace = ['Background', 'Signal'], value = [0, 1], inplace = True)
    if 'VBF' in signal:
        dataFrame['origin'].replace(to_replace = [signalLabel], value = [signal], inplace = True)
    plt.close()
    return

#def DrawVariablesHisto(dataFrame, InputFeatures, outputDir, outputFileCommonName, jetCollection, analysis, channel, signal, background, preselectionCuts, scaled = False):
#def DrawVariablesHisto(dataFrame, InputFeatures, outputDir, outputFileCommonName, analysis, channel, signal, background, preselectionCuts, scaled = False):
def DrawVariablesHistoOld(dataFrame, outputDir, outputFileCommonName, analysis, channel, signal, background, preselectionCuts, scaled = False):
    '''
    ### Replacing '0' with 'Background' and '1' with 'Signal' in the 'isSignal' column
    dataFrame['isSignal'].replace(to_replace = [0, 1], value = ['Background', 'Signal'], inplace = True)
    dataFrameSignal = dataFrame[dataFrame['isSignal'] == 'Signal']
    dataFrameBkg = dataFrame[dataFrame['isSignal'] == 'Background']
    '''
    dataFrameSignal = dataFrame[dataFrame['origin'] == signal]
    dataFrameBkg = dataFrame[dataFrame['origin'] != signal]
    #print(list(set(list(dataFrameBkg['origin']))))
    if 'VBF' in signal:
        signalLabel = signal.replace('VBF', '')
        dataFrame['origin'].replace(to_replace = [signal], value = [signalLabel], inplace = True)
    else:
        signalLabel = signal
    featureLogX = ['fatjet_D2', 'fatjet_m', 'fatjet_pt', 'lep1_pt', 'lep2_pt', 'Zcand_pt']
    legendText = 'analysis: ' + analysis + '\nchannel: ' + channel + '\nsignal: ' + signal + '\nbackground: ' + ', '.join(background)
    if (preselectionCuts != 'none'):
        legendText += '\npreselection cuts: ' + preselectionCuts
    for feature in dataFrame.columns:
        '''
        if feature not in InputFeatures and feature != 'origin' and feature != 'weight' and feature != 'DNNScore_t':
            continue
        '''
        if 'Pass' in feature or feature == 'train_weight':
            continue
        print(feature)
        statType = 'density'#'probability'
        #hueType = dataFrame['isSignal']
        hueType = dataFrameBkg['origin']#dataFrame['isSignal']
        legendBool = True
        if feature == 'origin' or feature == 'weight':
            statType = 'count'
            hueType = dataFrame['origin']
            legendBool = False
        #binsDict = {'lep1_m': np.linspace(0, 0.12, 4), 'lep1_pt': np.linspace(0, 2000, 51), 'lep1_eta': np.linspace(-3, 3, 21), 'lep1_phi': np.linspace(-3.5, 3.5, 21), 'lep2_m': np.linspace(0, 0.12, 4), 'lep2_pt': np.linspace(0, 2000, 51), 'lep2_eta': np.linspace(-3, 3, 21), 'lep2_phi': np.linspace(-3.5, 3.5, 21), 'fatjet_m': np.linspace(0, 500, 51), 'fatjet_pt': np.linspace(0, 3000, 51), 'fatjet_eta': np.linspace(-3, 3, 21), 'fatjet_phi': np.linspace(-3.5, 3.5, 21), 'fatjet_D2': np.linspace(0, 15, 51), 'Zcand_m': np.linspace(60, 140, 21), 'Zcand_pt': np.linspace(0, 7000, 31), 'mass': 'auto', 'origin': 'auto', 'Zdijet_m': np.linspace(60, 140, 21), 'Zdijet_pt': 'auto', 'Zdijet_eta': np.linspace(-4, 4, 11), 'Zdijet_phi': np.linspace(-3.5, 3.5, 11), 'sigVJ1_m': np.linspace(0, 120, 21), 'sigVJ1_pt': np.linspace(0, 1000, 11), 'sigVJ1_eta': np.linspace(-3, 3, 21), 'sigVJ1_phi': np.linspace(-3.5, 3.5, 21), 'sigVJ2_m': np.linspace(0, 40, 16), 'sigVJ2_pt': np.linspace(0, 300, 11), 'sigVJ2_eta': np.linspace(-3, 3, 21), 'sigVJ2_phi': np.linspace(-3.5, 3.5, 21), 'DNNScore_W': np.linspace(0, 1, 21), 'DNNScore_Z': np.linspace(0, 1, 21), 'DNNScore_h': np.linspace(0, 1, 21), 'DNNScore_t': np.linspace(0, 1, 21), 'DNNScore_qg': np.linspace(0, 1, 21)} ## for Radion merged ggF
        binsDict = {}
        minBin = min(min(dataFrameBkg[feature]), min(dataFrameSignal[feature]))
        maxBin = max(max(dataFrameBkg[feature]), max(dataFrameSignal[feature]))
        Bins = np.linspace(minBin, maxBin, 16)
        if feature not in binsDict:
            binsDict[feature] = 'auto'
        if feature == 'weight':
            #ax = seaborn.histplot(data = dataFrame['weight'], x = dataFrame['weight'], hue = dataFrame['isSignal'], common_norm = False, stat = statType, legend = True) 
            ax = seaborn.histplot(data = dataFrame['weight'], x = dataFrame['weight'], hue = dataFrame['isSignal'], bins = np.linspace(min(dataFrame['weight']), max(dataFrame['weight']), 21), common_norm = False, stat = statType, legend = True) 
        if feature == 'origin':
            ax = seaborn.histplot(data = dataFrame[feature], x = dataFrame[feature], hue = dataFrame['origin'], common_norm = False, stat = statType, legend = True)
        else:
            '''
            if scaled == False and feature in binsDict:
                ax = seaborn.histplot(data = dataFrameBkg[feature], x = dataFrameBkg[feature], weights = dataFrameBkg['weight'], bins = np.array(binsDict[feature]), hue = hueType, legend = legendBool, multiple = 'stack', stat = 'probability', common_norm = True) #stat = statType
            if scaled == False and feature not in binsDict:
                ax = seaborn.histplot(data = dataFrame[feature], x = dataFrame[feature], weights = dataFrame['weight'], hue = hueType, common_norm = False, stat = statType, legend = legendBool)#, multiple = 'stack')
            elif scaled == True:
                ax = seaborn.histplot(data = dataFrame[feature], x = dataFrame[feature], weights = dataFrame['weight'], hue = hueType, common_norm = False, stat = statType, legend = legendBool)#, multiple = 'stack')
            '''
            ax = seaborn.histplot(data = dataFrame[feature], x = dataFrame[feature], weights = dataFrame['weight'], hue = hueType, bins = Bins, common_norm = False, stat = statType, legend = legendBool)#, multiple = 'stack')
            #seaborn.histplot(data = dataFrameSignal[feature], x = dataFrameSignal[feature], weights = dataFrameSignal['weight'], bins = np.array(binsDict[feature]), element = 'step', fill = False, stat = 'probability', color = 'red')#, lw = 2, color = 'blue', label = signalLabel)#, density = True)
            seaborn.histplot(data = dataFrameSignal[feature], x = dataFrameSignal[feature], weights = dataFrameSignal['weight'], bins = Bins, element = 'step', fill = False, stat = 'probability', color = 'red')#, lw = 2, color = 'blue', label = signalLabel)#, density = True)
        #elif fileNameSplitDataSet in sys.argv[0]:
        ''' histogram of train_weight without bins is slow 
            if feature == 'train_weight':
                ax = seaborn.histplot(data = dataFrame['weight'], x = dataFrame['weight'], hue = dataFrame['isSignal'], common_norm = False, stat = 'count', legend = True)
        '''
        #    contents, bins, _ = plt.hist(dataFrame[feature], weights = dataFrame['train_weight'], bins = 100, label = legendText)
        labelDict = {'lep1_e': r'lep$_1$ e [GeV]', 'lep1_m': r'lep$_1$ m [GeV]', 'lep1_pt': r'lep$_1$ p$_T$ [GeV]', 'lep1_eta': r'lep$_1$ $\eta$', 'lep1_phi': r'lep$_1$ $\phi$', 'lep2_e': r'lep$_2$ e [GeV]', 'lep2_m': r'lep$_2$ m [GeV]', 'lep2_pt': r'lep$_2$ p$_t$ [GeV]', 'lep2_eta': r'lep$_2$ $\eta$', 'lep2_phi': r'lep$_2$ $\phi$', 'fatjet_m': 'fat jet m [GeV]', 'fatjet_pt': r'fat jet p$_t$ [GeV]', 'fatjet_eta': r'fat jet $\eta$', 'fatjet_phi': r'fat jet $\phi$', 'fatjet_D2': r'fat jet D$_2$', 'Zcand_m': 'Z$_{cand}$ m [GeV]', 'Zcand_pt': r'Z$_{cand}$ p$_t$ [GeV]', 'X_VV_merged_m': 'X_boosted m [GeV]', 'X_ZZ_resolved_m': 'X_resolved_ZZ m [GeV]', 'X_WZ_resolved_m': 'X_resolved_WZ m [GeV]', 'mass': 'mass [GeV]', 'weight': 'weight', 'isSignal': 'isSignal', 'origin': 'origin', 'Wdijet_m': 'Wdijet m [GeV]', 'Wdijet_pt': 'Wdijet p$_T$ [GeV]', 'Wdijet_eta': 'Wdijet $\eta$', 'Wdijet_phi': 'Wdijet $\phi$', 'Zdijet_m': 'Z$_{dijet}$ m [GeV]', 'Zdijet_pt': 'Z$_{dijet}$ p$_T$ [GeV]', 'Zdijet_eta': 'Z$_{dijet}$ $\eta$', 'Zdijet_phi': 'Z$_{dijet}$ $\phi$', 'sigVJ1_m': 'sigVJ1 m', 'sigVJ1_pt': 'sigVJ1 p$_T$', 'sigVJ1_eta': 'sigVJ1 $\eta$', 'sigVJ1_phi': 'sigVJ1 $\phi$', 'sigVJ2_m': 'sigVJ2 m', 'sigVJ2_pt': 'sigVJ2 p$_T$', 'sigVJ2_eta': 'sigVJ2 $\eta$', 'sigVJ2_phi': 'sigVJ2 $\phi$', 'sigVJ1_m': 'sigVJ1 m', 'sigVJ1_pt': 'sigVJ1 p$_T$', 'sigVJ1_eta': 'sigVJ1 $\eta$', 'sigVJ1_phi': 'sigVJ1 $\phi$', 'sigVJ2_m': 'sigVJ2 m', 'sigVJ2_pt': 'sigVJ2 p$_T$', 'sigVJ2_eta': 'sigVJ2 $\eta$', 'sigVJ2_phi': 'sigVJ2 $\phi$', 'lep1_px': r'lep$_1$ p$_x$ [GeV]', 'lep1_py': r'lep$_1$ p$_y$ [GeV]', 'lep1_pz': r'lep$_1$ p$_z$ [GeV]', 'lep2_px': r'lep$_2$ p$_x$ [GeV]', 'lep2_py': r'lep$_2$ p$_y$ [GeV]', 'lep2_pz': r'lep$_2$ p$_z$ [GeV]', 'fatjet_e': 'fat jet e [GeV]', 'fatjet_px': r'fat jet p$_x$ [GeV]', 'fatjet_py': r'fat jet p$_y$', 'fatjet_pz': r'fat jet p$_z$'}
        '''
        if feature in featureLogX:
            ax.set_xscale('log')
        '''
        #plt.figtext(0.35, 0.7, legendText, wrap = True, horizontalalignment = 'left')
        #plt.figtext(0.77, 0.45, legendText, wrap = True, horizontalalignment = 'left')
        #plt.subplots_adjust(left = 0.1, right = 0.75)
        if feature in labelDict:
            if scaled == True:
                plt.xlabel('Scaled ' + labelDict[feature])
            else:
                plt.xlabel(labelDict[feature])
        else:
            if scaled == True and feature != 'origin':
                plt.xlabel('Scaled ' + feature)
            if scaled == False or feature == 'origin':
                plt.xlabel(feature)
        '''
        if fileNameSplitDataSet in sys.argv[0]:
            plt.ylabel('Weighted counts')
            plt.legend(handlelength = 0, handletextpad = 0, prop={'size': 15})
        if fileNameBuildDataSet in sys.argv[0]:
            plt.ylabel('Weighted probability (MC weights)')
            if feature == 'weight':
                plt.ylabel('Counts')
        '''
        plt.ylabel('Weighted probability (MC weights)')                                                                                                           
        if feature == 'weight' or feature == 'origin':
            plt.ylabel('Counts')
        #plt.legend(handlelength = 0, handletextpad = 0, prop={'size': 15})
        if feature == 'origin':
            plt.yscale('log')
        plt.title('SRs + ZCRs, ' + signalLabel + ', ' + analysis + ', ' + channel)
        pltName = '/Histo_' + feature + '_' + outputFileCommonName + '.png'
        plt.tight_layout()
        plt.savefig(outputDir + pltName)
        print('Saved ' + outputDir + pltName)
        plt.clf()
    dataFrame['isSignal'].replace(to_replace = ['Background', 'Signal'], value = [0, 1], inplace = True)
    if 'VBF' in signal:
        dataFrame['origin'].replace(to_replace = [signalLabel], value = [signal], inplace = True)
    plt.close()
    #plt.subplots_adjust(left = 0.15, right = 0.95)
    return

### Computing train weight
#def ComputeTrainWeights(dataSetSignal, dataSetBackground, massesSignalList, outputDir, fileCommonName, jetCollection, analysis, channel, signal, background, preselectionCuts, drawPlots):
def ComputeTrainWeights(dataSetSignal, dataSetBackground, massesSignalList, outputDir, fileCommonName, analysis, channel, signal, background, preselectionCuts, drawPlots):
    numbersDict = {}
    logString = ''
    for signalMass in massesSignalList:
        ### Number of signals with each mass value
        numbersDict[signalMass] = dataSetSignal[dataSetSignal['mass'] == signalMass].shape[0]

    ### Minimum number of signals with the same mass
    #minNumber = min(numbersDict.values())
    minNumber = sum(numbersDict.values()) / len(numbersDict)
    ### New column in the signal dataframe for train_weight
    dataSetSignal = dataSetSignal.assign(train_weight = 0)
    for signalMass in massesSignalList:
        dataSetSignalMass = dataSetSignal[dataSetSignal['mass'] == signalMass]
        ### Sum of MC weights for signal with the same mass
        signalSampleWeight = dataSetSignalMass['weight'].sum()
        ### Scale factor to equalize signal samples with different mass
        scaleFactorDict = minNumber / signalSampleWeight
        ### Train weight = MC weight * scale factor
        dataSetSignal['train_weight'] = np.where(dataSetSignal['mass'] == signalMass, dataSetSignal['weight'] * scaleFactorDict, dataSetSignal['train_weight'])
        if drawPlots:
            ### Filling the bar plot
            plt.bar(signal + ' ' + str(signalMass) + ' GeV', dataSetSignal[dataSetSignal['mass'] == signalMass]['train_weight'].sum(), color = 'blue')
    if drawPlots:
        plt.bar('all ' + signal, dataSetSignal['train_weight'].sum(), color = 'orange')
    '''
    ### All signal weight
    signalWeight = dataSetSignal['train_weight'].sum()
    '''
    ### Background MC weight
    bkgWeight = dataSetBackground['weight'].sum()
    ### Scale factor to equalize signal/background
    scaleFactor = minNumber *  len(massesSignalList)/ bkgWeight
    ### Creating new column in the background dataframe with the train weight
    dataSetBackground = dataSetBackground.assign(train_weight = dataSetBackground['weight'] * scaleFactor)

    if drawPlots:
        plt.bar('background', dataSetBackground['train_weight'].sum(), color = 'green')
        plt.ylabel('Weighted counts')
        plt.xticks(rotation = 'vertical')
        legendText = 'analysis: ' + analysis + '\nchannel: ' + channel + '\nsignal: ' + signal + '\nbackground: ' + ', '.join(background)
        if (preselectionCuts != 'none'):
            legendText += '\npreselection cuts: ' + preselectionCuts
        plt.figtext(0.25, 0.7, legendText, wrap = True, horizontalalignment = 'left')
        plt.tight_layout()
        pltName = outputDir + '/WeightedEvents_' + fileCommonName + '.png'
        plt.savefig(pltName)
        print('Saved ' + pltName)
        logString = '\nSaved ' + pltName
        plt.clf()
        plt.close()
    return dataSetSignal, dataSetBackground, logString
    

### Computing weighted median and IQR range
def ScalingFeatures(dataTrain, dataTest, InputFeatures, outputDir):
    dataTrainCopy = dataTrain.copy()
    sumTrainWeights = np.array(dataTrainCopy['train_weight']).sum()
    halfTrainWeights = sumTrainWeights / 2
    perc1 = sumTrainWeights * 0.25
    perc2 = sumTrainWeights * 0.75
    variablesFileName = outputDir + '/variables.json'
    variablesFile = open(variablesFileName, 'w')
    variablesFile.write("{\n")
    variablesFile.write("  \"inputs\": [\n")
    for feature in InputFeatures:
        if 'DNN' in feature:
            continue
        cumulativeSum = 0
        print('Scaling ' + feature)
        dataTrainCopy = dataTrainCopy.sort_values(by = [feature])
        for index in range(len(dataTrainCopy)):
            cumulativeSum += dataTrainCopy['train_weight'].iloc[index]
            if cumulativeSum <= perc1:
                perc1index = index
            else:
                break
        for index in range(perc1index, len(dataTrainCopy)):
            cumulativeSum += dataTrainCopy['train_weight'].iloc[index]
            if cumulativeSum <= halfTrainWeights:
                medianIndex = index
            else:
                break
        for index in range(medianIndex, len(dataTrainCopy)):
            cumulativeSum += dataTrainCopy['train_weight'].iloc[index]
            if cumulativeSum <= perc2:
                perc2index = index
            else:
                break
        quartileLeft = dataTrainCopy[feature].iloc[perc1index]
        median = dataTrainCopy[feature].iloc[medianIndex]
        quartileRight = dataTrainCopy[feature].iloc[perc2index]
        iqr = quartileRight - quartileLeft ### InterQuartile Range
        dataTrain[feature] = (dataTrain[feature] - median) / iqr
        dataTest[feature] = (dataTest[feature] - median) / iqr
        variablesFile.write("    {\n")
        variablesFile.write("      \"name\": \"%s\",\n" % feature)
        variablesFile.write("      \"offset\": %lf,\n" % median) # EJS 2021-05-27: I have compelling reasons to believe this should be -mu
        variablesFile.write("      \"scale\": %lf\n" % iqr) # EJS 2021-05-27: I have compelling reasons to believe this should be 1/sigma                            
        variablesFile.write("    }")
        if feature != InputFeatures[len(InputFeatures) - 1]:
            variablesFile.write(",\n")
        else:
            variablesFile.write("\n")
    variablesFile.write("  ],\n")
    variablesFile.write("  \"class_labels\": [\"BinaryClassificationOutputName\"]\n")
    variablesFile.write("}\n")
    print('Saved variables offsets and scales in ' + variablesFileName)
    logString = '\nSaved variables offset and scales in ' + variablesFileName
    return dataTrain, dataTest, logString

### Computing weighted median and IQR range
def ComputeScaleFactors(dataTrain, outputDir):
    dataTrainCopy = dataTrain.copy()
    variablesNotToScale = ['Pass', 'DNN', 'origin', 'isSignal', 'weight', 'train_weight', 'unscaledMass']
    variablesToScale = []
    for variable in dataTrain.columns:
        addVariable = True
        for variableNotToScale in variablesNotToScale:
            if variableNotToScale in variable:
                addVariable = False
                print('Not scaling ' + variable)
                break
        if addVariable == True:
            variablesToScale.append(variable)
    sumTrainWeights = np.array(dataTrainCopy['train_weight']).sum()
    halfTrainWeights = sumTrainWeights / 2
    perc1 = sumTrainWeights * 0.25
    perc2 = sumTrainWeights * 0.75
    variablesFileName = outputDir + '/variables.json'
    variablesFile = open(variablesFileName, 'w')
    variablesFile.write("{\n")
    variablesFile.write("  \"inputs\": [\n")
    for feature in variablesToScale:
        cumulativeSum = 0
        print('Computing scale factors (median and IQR) for ' + feature)
        dataTrainCopy = dataTrainCopy.sort_values(by = [feature])
        for index in range(len(dataTrainCopy)):
            cumulativeSum += dataTrainCopy['train_weight'].iloc[index]
            if cumulativeSum <= perc1:
                perc1index = index
            else:
                break
        for index in range(perc1index, len(dataTrainCopy)):
            cumulativeSum += dataTrainCopy['train_weight'].iloc[index]
            if cumulativeSum <= halfTrainWeights:
                medianIndex = index
            else:
                break
        for index in range(medianIndex, len(dataTrainCopy)):
            cumulativeSum += dataTrainCopy['train_weight'].iloc[index]
            if cumulativeSum <= perc2:
                perc2index = index
            else:
                break
        quartileLeft = dataTrainCopy[feature].iloc[perc1index]
        median = dataTrainCopy[feature].iloc[medianIndex]
        quartileRight = dataTrainCopy[feature].iloc[perc2index]
        iqr = quartileRight - quartileLeft ### InterQuartile Range (IQR)
        variablesFile.write("    {\n")
        variablesFile.write("      \"name\": \"%s\",\n" % feature)
        variablesFile.write("      \"offset\": %lf,\n" % median) # EJS 2021-05-27: I have compelling reasons to believe this should be -mu
        variablesFile.write("      \"scale\": %lf\n" % iqr) # EJS 2021-05-27: I have compelling reasons to believe this should be 1/sigma                            
        variablesFile.write("    }")
        if feature != variablesToScale[len(variablesToScale) - 1]:
            variablesFile.write(",\n")
        else:
            variablesFile.write("\n")
    variablesFile.write("  ],\n")
    variablesFile.write("  \"class_labels\": [\"BinaryClassificationOutputName\"]\n")
    variablesFile.write("}\n")
    print('Saved variables offsets (median) and scales (interquartile range, IQR) in ' + variablesFileName)
    logString = '\nSaved variables offset (median) and scales (interquartile range, IQR) in ' + variablesFileName
    return logString

### Building the (P)DNN
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.layers.core import Dense, Activation
from sklearn.metrics import log_loss
import tensorflow as tf
#from keras.optimizers import SGD

def BuildNN(N_input, nodesNumber, layersNumber, dropout, studyLearningRate):
    activationFunction = 'swish'#'relu'#'swish'
    model = Sequential()
    model.add(Dense(units = nodesNumber, input_dim = N_input, activation = activationFunction))
    #model.add(Dense(units = 88, input_dim = N_input, activation = 'relu'))
    #model.add(Activation('relu'))
    if dropout > 0:
        model.add(Dropout(dropout))
    for i in range(0, layersNumber):
        model.add(Dense(nodesNumber, activation = activationFunction)) ##swish
    #    model.add(Activation('relu'))
        model.add(Dropout(dropout))
    model.add(Dense(1, activation = 'sigmoid'))
    #model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
    Loss = 'binary_crossentropy'
    Metrics = ['accuracy']
    learningRate = 0.01#0.005#0.01#0.001#0.0003 #0.001
    '''
    if studyLearningRate:
        Optimizer = SGD(lr = 0.1)
    else:
        #Optimizer = SGD(lr = 0.1)
        #Optimizer = tf.keras.optimizers.Adam(learning_rate = learningRate) #Adam
    '''
    #Optimizer = tf.keras.optimizers.Adam(learning_rate = learningRate) #Adam, Nadam
    Optimizer = tf.keras.optimizers.Nadam(learning_rate = learningRate) #Adam, Nadam
    return model, Loss, Metrics, learningRate, Optimizer, activationFunction

def scheduler(epoch, lr):
    if epoch <= 5:
        return 0.003
    elif epoch < 30:
        return 0.001
    else:
        return 0.0001

def SaveArchAndWeights(model, outputDir):
    arch = model.to_json()
    outputArch = outputDir + '/architecture.json'
    with open(outputArch, 'w') as arch_file:
        arch_file.write(arch)
    print('Saved NN architecture in ' + outputArch)
    outputWeights = outputDir + '/weights.h5'
    model.save_weights(outputWeights)
    print('Saved NN weights in ' + outputWeights)

def SaveVariables(outputDir, X_input):
    outputVar = outputDir + '/variables.json'
    with open(outputVar, 'w') as var_file:
        var_file.write("{\n")
        var_file.write("  \"inputs\": [\n")
        for col in X_input.columns:
            offset = -1. * float(X_input.mean(axis = 0)[col])
            scale = 1. / float(X_input.std(axis = 0)[col])
            var_file.write("    {\n")
            var_file.write("      \"name\": \"%s\",\n" % col)
            var_file.write("      \"offset\": %lf,\n" % offset) # EJS 2021-05-27: I have compelling reasons to believe this should be -mu
            var_file.write("      \"scale\": %lf\n" % scale) # EJS 2021-05-27: I have compelling reasons to believe this should be 1/sigma
            var_file.write("    }")
            '''
            if (col < X_input.shape[1]-1):
                var_file.write(",\n")
            else:
                var_file.write("\n")
            '''
            if col != X_input.columns[len(X_input.columns) - 1]:
                var_file.write(",\n")
            else:
                var_file.write("\n")
        var_file.write("  ],\n")
        var_file.write("  \"class_labels\": [\"BinaryClassificationOutputName\"]\n")
        var_file.write("}\n")
    print('Saved variables in ' + outputVar)

def SaveFeatureScaling(outputDir, X_input):
    outputFeatureScaling = outputDir + '/FeatureScaling.dat'
    with open(outputFeatureScaling, 'w') as scaling_file: # EJS 2021-05-27: check which file name is hardcoded in the CxAODReader
        scaling_file.write("[")
        scaling_file.write(', '.join(str(i) for i in X_input.columns))
        scaling_file.write("]\n")
        scaling_file.write("Mean\n")
        scaling_file.write("[")
        scaling_file.write(' '.join(str(float(i)) for i in X_input.mean(axis=0)))
        scaling_file.write("]\n")
        scaling_file.write("minusMean\n")
        scaling_file.write("[")
        scaling_file.write(' '.join(str(-float(i)) for i in X_input.mean(axis=0)))
        scaling_file.write("]\n")
        scaling_file.write("Var\n")
        scaling_file.write("[")
        scaling_file.write(' '.join(str(float(i)) for i in X_input.var(axis=0)))
        scaling_file.write("]\n")
        scaling_file.write("sqrtVar\n")
        scaling_file.write("[")
        scaling_file.write(' '.join(str(float(i)) for i in X_input.std(axis=0)))
        scaling_file.write("]\n")
        scaling_file.write("OneOverStd\n")
        scaling_file.write("[")
        scaling_file.write(' '.join(str(1./float(i)) for i in X_input.std(axis=0)))
        scaling_file.write("]\n")
    print('Saved features scaling in ' + outputFeatureScaling)

def SaveModel(model, outputDir, NN):
    SaveArchAndWeights(model, outputDir)
    variablesFileName = '/variables.json'
    #previousDir = outputDir.replace('loose' + NN, '') ###########aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    #previousDir = outputDir.replace(NN, '') ###########aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    variablesDir = outputDir.rsplit('/', 1)[0]
    #variablesDir += variablesFileName
    #print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA ' + variablesDir + variablesFileName)
    #shutil.copyfile(variablesDir + variablesFileName, outputDir + variablesFileName)

def extractFeatures(dataFrame, inputFeatures):
    X = dataFrame[inputFeatures]
    y = dataFrame['isSignal']
    w = dataFrame['train_weight']
    return X, y, w

'''
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [7,7]
plt.rcParams.update({'font.size': 16})
'''
### Evaluating the (P)DNN performance
def EvaluatePerformance(model, X_test, y_test, w_test, batchSize):
    perf = model.evaluate(X_test, y_test, sample_weight = w_test, batch_size = batchSize)
    testLoss = perf[0]
    testAccuracy = perf[1]
    return testLoss, testAccuracy

### Prediction on train and test sample (for DNN)
def PredictionTrainTest(model, X_test, X_train, batchSize):
    yhat_test = model.predict(X_test, batch_size = batchSize)
    yhat_train = model.predict(X_train, batch_size = batchSize)
    return yhat_test, yhat_train
    
### Prediction on signal and background separately (for PDNN)
def PredictionSigBkg(model, X_train_signal, X_train_bkg, X_test_signal, X_test_bkg, batchSize):
    yhat_train_signal = model.predict(X_train_signal, batch_size = batchSize)
    yhat_train_bkg = model.predict(X_train_bkg, batch_size = batchSize)
    yhat_test_signal = model.predict(X_test_signal, batch_size = batchSize)
    yhat_test_bkg = model.predict(X_test_bkg, batch_size = batchSize)
    return yhat_train_signal, yhat_train_bkg, yhat_test_signal, yhat_test_bkg

### Drawing correlation matrix
def DrawCorrelationMatrix(dataFrame, InputFeatures, outputDir, outputFileCommonName, analysis, channel, signal, bkg):
    fig, ax1 = plt.subplots(figsize = (10, 10))
    plt.set_cmap('bwr')
    im = ax1.matshow((dataFrame[InputFeatures].astype(float)).corr(), vmin = -1, vmax = 1)
    plt.colorbar(im, ax = ax1)
    plt.xticks(range(len(InputFeatures)), InputFeatures, rotation = 'vertical')
    plt.yticks(range(len(InputFeatures)), InputFeatures)
    '''
    for feature1 in range(len(InputFeatures)): ### This is really slow, perform only if needed
        for feature2 in range(len(InputFeatures)):
            ax1.text(feature2, feature1, "%.2f" % (dataFrame[InputFeatures].astype(float)).corr().at[InputFeatures[feature2], InputFeatures[feature1]], ha = 'center', va = 'center', color = 'r', fontsize = 6)
    '''
    plt.title('Correlation matrix')# (' + analysis + ' ' + channel + ' ' + signal + ' ' + bkg + ')')
    #legendText = 'jet collection: ' + jetCollection + '\nanalysis: ' + analysis + '\nchannel: ' + channel + '\nsignal: ' + signal + '\nbackground: ' + ', '.join(bkg)
    legendText = '\nanalysis: ' + analysis + '\nchannel: ' + channel + '\nsignal: ' + signal + '\nbackground: ' + ', '.join(bkg)
    #if (preselectionCuts != 'none'):
    #    legendText += '\npreselection cuts: ' + preselectionCuts
    plt.figtext(0.77, 0.45, legendText, wrap = True, horizontalalignment = 'left')
    plt.tight_layout()
    plt.subplots_adjust(left = 0.1, right = 0.75)
    CorrelationMatrixName = outputDir + '/CorrelationMatrix_' + outputFileCommonName + '.png'
    plt.savefig(CorrelationMatrixName)
    print('Saved ' + CorrelationMatrixName)
    plt.clf()
    plt.close()

### Drawing Accuracy
#def DrawAccuracy(modelMetricsHistory, testAccuracy, patienceValue, outputDir, NN, jetCollection, analysis, channel, PreselectionCuts, signal, bkg, outputFileCommonName, mass = 0):
def DrawAccuracy(modelMetricsHistory, testAccuracy, patienceValue, outputDir, NN, analysis, channel, PreselectionCuts, signal, bkg, outputFileCommonName, mass = 0):
    plt.plot(modelMetricsHistory.history['accuracy'], label = 'Training')
    lines = plt.plot(modelMetricsHistory.history['val_accuracy'], label = 'Validation')
    xvalues = lines[0].get_xdata()
    if testAccuracy != None:
        plt.scatter([xvalues[len(xvalues) - 1 - patienceValue]], [testAccuracy], label = 'Test', color = 'green')
    emptyPlot, = plt.plot([0, 0], [1, 1], color = 'white')
    titleAccuracy = NN + ' model accuracy'
    if NN == 'DNN':
        outputFileCommonName += '_' + str(int(mass))
        if mass >= 1000:
            titleAccuracy += ' (mass: ' + str(float(mass / 1000)) + ' TeV)'
        else:
            titleAccuracy += ' (mass: ' + str(int(mass)) + ' GeV)'
    plt.title(titleAccuracy)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    #plt.legend()
    #legend1 = plt.legend(['Training', 'Validation'], loc = 'lower right')
    legend1 = plt.legend(loc = 'center right')
    #legendText = 'jet collection: ' + jetCollection + '\nanalysis: ' + analysis + '\nchannel: ' + channel + '\nsignal: ' + signal + '\nbackground: ' + str(bkg)
    legendText = 'analysis: ' + analysis + '\nchannel: ' + channel + '\nsignal: ' + signal + '\nbackground: ' + str(bkg)
    if (PreselectionCuts != 'none'):
        legendText += '\npreselection cuts: ' + PreselectionCuts
    #legendText += '\nTest accuracy: ' + str(round(testAccuracy, 2))
    #plt.figtext(0.5, 0.3, legendText, wrap = True, horizontalalignment = 'left')
    #plt.legend(legendText)
    legend2 = plt.legend([emptyPlot], [legendText], frameon = False)
    plt.gca().add_artist(legend1)
    #plt.figtext(0.69, 0.28, 'Test accuracy: ' + str(round(testAccuracy, 2)), wrap = True, horizontalalignment = 'left')#, fontsize = 10)
    AccuracyPltName = outputDir + '/Accuracy_' + outputFileCommonName + '.png'
    plt.tight_layout()
    plt.savefig(AccuracyPltName)
    print('Saved ' + AccuracyPltName)
    plt.clf()
    plt.close()

### Drawing Loss
#def DrawLoss(modelMetricsHistory, testLoss, patienceValue, outputDir, NN, jetCollection, analysis, channel, PreselectionCuts, signal, bkg, outputFileCommonName, mass = 0):
def DrawLoss(modelMetricsHistory, testLoss, patienceValue, outputDir, NN, analysis, channel, PreselectionCuts, signal, bkg, outputFileCommonName, mass = 0):
    plt.plot(modelMetricsHistory.history['loss'], label = 'Training')
    lines = plt.plot(modelMetricsHistory.history['val_loss'], label = 'Validation')
    xvalues = lines[0].get_xdata()
    #print(yvalues[len(yvalues) - 1])    
    if testLoss != None:
        plt.scatter([xvalues[len(xvalues) - 1 - patienceValue]], [testLoss], label = 'Test', color = 'green')
    #emptyPlot, = plt.plot([0, 0], [1, 1], color = 'white')
    titleLoss = NN + ' model loss'
    if NN == 'DNN':
        outputFileCommonName += '_' + str(int(mass))
        if mass >= 1000:
            titleLoss += ' (mass: ' + str(float(mass / 1000)) + ' TeV)'
        else:
            titleLoss += ' (mass: ' + str(int(mass)) + ' GeV)'
    plt.title(titleLoss)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    #plt.legend(['Training', 'Validation'], loc = 'upper right')
    legend1 = plt.legend(loc = 'upper center')
    #legendText = 'jet collection: ' + jetCollection + '\nanalysis: ' + analysis + '\nchannel: ' + channel + '\npreselection cuts: ' + PreselectionCuts + '\nsignal: ' + signal + '\nbackground: ' + str(bkg)
    legendText = 'analysis: ' + analysis + '\nchannel: ' + channel + '\npreselection cuts: ' + PreselectionCuts + '\nsignal: ' + signal + '\nbackground: ' + str(bkg)
    if (PreselectionCuts != 'none'):
        legendText += '\npreselection cuts: ' + PreselectionCuts
    #legendText += '\nTest loss: ' + str(round(testLoss, 2))
    plt.figtext(0.4, 0.4, legendText, wrap = True, horizontalalignment = 'left')
    #plt.figtext(0.7, 0.7, 'Test loss: ' + str(round(testLoss, 2)), wrap = True, horizontalalignment = 'center')#, fontsize = 10)
    #legend2 = plt.legend([emptyPlot], [legendText], frameon = False, loc = 'center right')
    #plt.gca().add_artist(legend1)
    LossPltName = outputDir + '/Loss_' + outputFileCommonName + '.png'
    plt.tight_layout()
    plt.savefig(LossPltName)
    print('Saved ' + LossPltName)
    plt.clf()
    plt.close()

from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report
'''
def DrawROC(fpr, tpr, AUC, outputDir, NN, unscaledMass, jetCollection, analysis, channel, preselectionCuts, signal, background):
    plt.plot(fpr,  tpr, color = 'darkorange', lw = 2)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    titleROC = 'ROC curves (mass: ' + str(int(unscaledMass)) + ' GeV)'
    plt.title(titleROC)
    plt.figtext(0.7, 0.25, 'AUC: ' + str(round(AUC, 2)), wrap = True, horizontalalignment = 'center')
    ROCPltName = outputDir + '/oldROC.png'
    plt.savefig(ROCPltName)
    print('Saved ' + ROCPltName)
    plt.clf()
'''

### Drawing ROC and background rejection vs efficiency
#def DrawROCbkgRejectionScores(fpr, tpr, AUC, outputDir, NN, unscaledMass, jetCollection, analysis, channel, preselectionCuts, signal, background, outputFileCommonName, yhat_train_signal, yhat_test_signal, yhat_train_bkg, yhat_test_bkg, wMC_train_signal_mass, wMC_test_signal_mass, wMC_train_bkg, wMC_test_bkg, drawPlots):
def DrawROCbkgRejectionScores(fpr, tpr, AUC, outputDir, NN, unscaledMass, analysis, channel, preselectionCuts, signal, background, outputFileCommonName, yhat_train_signal, yhat_test_signal, yhat_train_bkg, yhat_test_bkg, wMC_train_signal_mass, wMC_test_signal_mass, wMC_train_bkg, wMC_test_bkg, drawPlots):

    ### ROC
    if drawPlots:
        emptyPlot, = plt.plot(fpr[0], tpr[0], color = 'white')
        plt.plot(fpr, tpr, color = 'darkorange', label = 'AUC: ' + str(round(AUC, 2)), lw = 2)
        #plt.plot(fakeFPR, fakeTPR, color = 'blue', label = 'reweighted AUC: ' + str(round(fakeAUC, 2)), lw = 2)
        legend1 = plt.legend(handlelength = 0, handletextpad = 0, loc = 'center right')
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        if unscaledMass >= 1000:
            titleScores = NN + ' scores (mass: ' + str(float(unscaledMass / 1000)) + ' TeV)'#, bkg: ' + background + ')'
            titleROC = NN + ' ROC curve (mass: ' + str(float(unscaledMass / 1000)) + ' TeV)'#, bkg: ' + background + ')'
            BkgRejTitle = NN + ' rejection curve (mass: ' + str(float(unscaledMass / 1000)) + ' TeV)'#, bkg: ' + background + ')'
        else:
            titleScores = NN + ' scores (mass: ' + str(int(unscaledMass)) + ' GeV)'#, bkg: ' + background + ')'
            titleROC = NN + ' ROC curve (mass: ' + str(int(unscaledMass)) + ' GeV)'#, bkg: ' + background + ')'
            BkgRejTitle = NN + ' rejection curve (mass: ' + str(int(unscaledMass)) + ' GeV)'#, bkg: ' + background + ')'
        plt.title(titleROC)
        #legendText = 'jet collection: ' + jetCollection + '\nanalysis: ' + analysis + '\nchannel: ' + channel + '\nsignal: ' + signal + '\nbackground: ' + str(background)
        legendText = 'analysis: ' + analysis + '\nchannel: ' + channel + '\nsignal: ' + signal# + '\nbackground: ' + str(background)
        if (preselectionCuts != 'none'):
            legendText += '\npreselection cuts: ' + preselectionCuts
        legend2 = plt.legend([emptyPlot], [legendText], loc = 'lower right', handlelength = 0, handletextpad = 0)
        for item in legend2.legendHandles:
            item.set_visible(False)

        plt.gca().add_artist(legend1)
        ROCPltName = outputDir + '/ROC_' + outputFileCommonName + '_' + str(unscaledMass) + '.png'
        plt.savefig(ROCPltName)#, bbox_inches = 'tight')
        print('Saved ' + ROCPltName)
        plt.clf()
        plt.close()
    
        ### Scores
        Nbins = 1000
        plt.hist(yhat_train_signal, weights = wMC_train_signal_mass, bins = Nbins, histtype = 'step', lw = 2, color = 'blue', label = [r'Signal train'], density = True)
        y_signal, bins_1, _ = plt.hist(yhat_test_signal, weights = wMC_test_signal_mass, bins = Nbins, histtype = 'stepfilled', lw = 2, color = 'cyan', alpha = 0.5, label = [r'Signal test'], density = True)
        plt.hist(yhat_train_bkg, weights = wMC_train_bkg, bins = Nbins, histtype = 'step', lw = 2, color = 'red', label = [r'Background train'], density = True)
        y_bkg, bins_0, _ = plt.hist(yhat_test_bkg, weights = wMC_test_bkg, bins = Nbins, histtype = 'stepfilled', lw = 2, color = 'orange', alpha = 0.5, label = [r'Background test'], density = True)
        plt.ylabel('Norm. entries')
        plt.xlabel('Score')
        plt.yscale('log')
        plt.title(titleScores)
        plt.legend(loc = 'upper center')
        ScoresPltName = outputDir + '/Scores_' + outputFileCommonName + '_' + str(unscaledMass) + '.png'
        plt.savefig(ScoresPltName)
        print('Saved ' + ScoresPltName)
        plt.clf()
        plt.close()

    ### Background rejection vs efficiency
    tprCut = tpr[tpr > 0.85]
    fprCut = fpr[tpr > 0.85]
    fprCutInverse = 1 / fprCut
    if drawPlots:
        plt.plot(tprCut, fprCutInverse)
        emptyPlot, = plt.plot(tprCut[0], fprCutInverse[0] + 30, color = 'white')
  
    WP = [0.90, 0.94, 0.97, 0.99]
    bkgRejections = np.array([])
    #print('Mass: ' + str(unscaledMass), ', background rejection at WP = 0.90: ' + str(bkgRej90))

    for i in range(0, len(WP)):
        bkgRejections = np.append(bkgRejections, np.interp(WP[i], tprCut, fprCutInverse))
        if drawPlots:
            plt.axvline(x = WP[i], color = 'Red', linestyle = 'dashed', label = 'Bkg Rejection @ ' + str(WP[i]) + ' WP: ' + str(round(bkgRejections[i], 1)))

    print(format('Working points: ' + str(WP) + '\nBackground rejection at each working point: ' + str(bkgRejections)))
    if drawPlots:
        plt.xlabel('Signal efficiency')
        plt.ylabel('Background rejection')
        plt.yscale('log')    
        plt.title(BkgRejTitle)
        legend1 = plt.legend()
        legend2 = plt.legend([emptyPlot], [legendText], loc = 'center left', handlelength = 0, handletextpad = 0)
        for item in legend2.legendHandles:
            item.set_visible(False)
        plt.gca().add_artist(legend1)
        EffPltName = outputDir + '/BkgRejection_' + outputFileCommonName + '_' + str(unscaledMass) + '.png'
        plt.savefig(EffPltName)#, bbox_inches = 'tight')
        print('Saved ' + EffPltName)
        plt.clf()
        plt.close()
    return WP, bkgRejections


def integral(y,x,bins):
    x_min=x
    s=0
    for i in np.where(bins>x)[0][:-1]:
        s=s+y[i]*(bins[i+1]-bins[i])
    return s

### Drawing scores, ROC and efficiency
#def DrawEfficiency(yhat_train_signal, yhat_test_signal, yhat_train_bkg, yhat_test_bkg, outputDir, NN, mass, jetCollection, analysis, channel, PreselectionCuts, signal, bkg, outputFileCommonName):
def DrawEfficiency(yhat_train_signal, yhat_test_signal, yhat_train_bkg, yhat_test_bkg, outputDir, NN, mass, analysis, channel, PreselectionCuts, signal, bkg, outputFileCommonName):

    ### Scores
    Nbins = 1000
    plt.hist(yhat_train_signal, bins = Nbins, histtype = 'step', lw = 2, color = 'blue', label = [r'Signal train'], density = True)
    y_signal, bins_1, _ = plt.hist(yhat_test_signal, bins = Nbins, histtype = 'stepfilled', lw = 2, color = 'cyan', alpha = 0.5, label = [r'Signal test'], density = True)
    plt.hist(yhat_train_bkg, bins = Nbins, histtype = 'step', lw = 2, color = 'red', label = [r'Background train'], density = True)
    y_bkg, bins_0, _ = plt.hist(yhat_test_bkg, bins = Nbins, histtype = 'stepfilled', lw = 2, color = 'orange', alpha = 0.5, label = [r'Background test'], density = True)
    if savePlot:
        plt.ylabel('Norm. entries')
        plt.xlabel('Score')
        plt.yscale('log')
        if mass >= 1000:
            titleScores = NN + ' scores (mass: ' + str(float(mass / 1000)) + ' TeV, bkg: ' + bkg + ')'
        else:
            titleScores = NN + ' scores (mass: ' + str(int(mass)) + ' GeV, bkg: ' + bkg + ')'
        plt.title(titleScores)
        plt.legend(loc = 'upper center')
        legendText = 'analysis: ' + analysis + '\nchannel: ' + channel + '\nsignal: ' + signal + '\nbackground: ' + str(bkg)
        if (PreselectionCuts != 'none'):
            legendText += '\npreselection cuts: ' + PreselectionCuts
        #plt.figtext(0.35, 0.45, legendText, wrap = True, horizontalalignment = 'left')
        #ScoresPltName = outputDir + '/Scores_' + bkg + '.png'
        ScoresPltName = outputDir + '/Scores_' + outputFileCommonName + '_' + str(mass) + '.png'
        plt.savefig(ScoresPltName)
        print('Saved ' + ScoresPltName)
        plt.clf()
    plt.close()
    ### ROC
    Nsignal = integral(y_signal, 0, bins_1)
    Nbkg = integral(y_bkg, 0, bins_0)
    signal_eff = np.array([])
    bkg_eff = np.array([])
    y_s = 0
    y_n = 0
    for i in range(0, Nbins + 1):
        x = i/Nbins
        y_s = integral(y_signal, x, bins_1) / Nsignal
        y_n = integral(y_bkg, x, bins_0) / Nbkg
        signal_eff = np.append(y_s, signal_eff)
        bkg_eff = np.append(y_n, bkg_eff)

    Area=round(1000*abs(integral(signal_eff,0,bkg_eff)))/1000
    if savePlot:
        lab='Area: '+str(Area)
        plt.plot(bkg_eff,signal_eff,label=lab,color = 'darkorange', lw = 2)
        #plt.plot([0,1],[0,1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        if mass >= 1000:
            titleROC = NN + ' ROC curve (mass: ' + str(float(mass / 1000)) + ' TeV, bkg: ' + bkg + ')'
        else:
            titleROC = NN + ' ROC curve (mass: ' + str(int(mass)) + ' GeV, bkg: ' + bkg + ')'
        plt.title(titleROC)
        #legendText = 'jet collection: ' + jetCollection + '\nanalysis: ' + analysis + '\nchannel: ' + channel + '\nsignal: ' + signal + '\nbackground: ' + str(bkg)
        legendText = 'analysis: ' + analysis + '\nchannel: ' + channel + '\nsignal: ' + signal# + '\nbackground: ' + str(bkg)
        if (PreselectionCuts != 'none'):
            legendText += '\npreselection cuts: ' + PreselectionCuts
        plt.figtext(0.4, 0.25, legendText, wrap = True, horizontalalignment = 'left')
        plt.figtext(0.4, 0.2, 'AUC: ' + str(round(Area, 2)), wrap = True, horizontalalignment = 'center')
        #ROCPltName = outputDir + '/ROC_' + bkg + '.png'
        ROCPltName = outputDir + '/ROC_' + outputFileCommonName + '_' + str(mass) + '.png'
        plt.savefig(ROCPltName)
        print('Saved ' + ROCPltName)
        plt.clf()
    plt.close()
    ### Background rejection vs efficiency
    WP=[0.90,0.94,0.97,0.99]
    rej=1./bkg_eff
    WP_idx=[np.where(np.abs(signal_eff-WP[i])==np.min(np.abs(signal_eff-WP[i])))[0][0] for i in range(0,len(WP))]
    WP_rej=[str(round(10*rej[WP_idx[i]])/10) for i in range(0,len(WP))]
    print('Working points (WP): ' + str(WP))
    #print('Working points (WP): ' + str(bins_0[Nbins-np.array(WP_idx)]))
    print('Background rejection at each WP: ' + str(WP_rej))

    if savePlot:
        plt.plot(signal_eff,rej)
        for i in range(0,len(WP)):
            plt.axvline(x=WP[i],color='Red',linestyle='dashed',label='Bkg Rejection @ '+str(WP[i])+' WP: '+WP_rej[i])
        plt.xlabel('Signal efficiency')
        plt.ylabel('Background rejection')
        plt.xlim([0.85,1])
        plt.yscale('log')
        if mass >= 1000:
            BkgRejTitle = NN + ' rejection curve (mass: ' + str(float(mass / 1000)) + ' TeV, bkg: ' + bkg + ')'
        else:
            BkgRejTitle = NN + ' rejection curve (mass: ' + str(int(mass)) + ' GeV, bkg: ' + bkg + ')'
        plt.title(BkgRejTitle)
        plt.legend()
        #EffPltName = outputDir + '/BkgRejection_' + bkg +'.png'
        EffPltName = outputDir + '/BkgRejection_' + outputFileCommonName + '_' + str(mass) + '.png'
        plt.savefig(EffPltName)
        print('Saved ' + EffPltName)
        plt.clf()
        plt.close()
    return Area, WP, WP_rej 

#def DrawRejectionVsMass(massVec, WP, bkgRej90, bkgRej94, bkgRej97, bkgRej99, outputDir, jetCollection, analysis, channel, preselectionCuts, signal, background, outputFileCommonName):
def DrawRejectionVsMass(massVec, WP, bkgRej90, bkgRej94, bkgRej97, bkgRej99, outputDir, analysis, channel, preselectionCuts, signal, background, outputFileCommonName):

    emptyPlot, = plt.plot(massVec[0], bkgRej90[0], color = 'white')
    plt.plot(massVec, bkgRej90, color = 'blue', label = 'WP: ' + str(WP[0]), marker = 'o', mec = 'blue')
    plt.plot(massVec, bkgRej94, color = 'orange', label = 'WP: ' + str(WP[1]), marker = 'o', mec = 'orange')
    plt.plot(massVec, bkgRej97, color = 'green', label = 'WP: ' + str(WP[2]), marker = 'o', mec = 'green')
    plt.plot(massVec, bkgRej99, color = 'red', label = 'WP: ' + str(WP[3]), marker = 'o', mec = 'red')
    plt.yscale('log')
    legend1 = plt.legend()
    plt.xlabel('Mass [GeV]')
    plt.ylabel('Background rejection')
    #legendText = 'jet collection: ' + jetCollection + '\nanalysis: ' + analysis + '\nchannel: ' + channel + '\nsignal: ' + signal + '\nbackground: ' + str(background)
    legendText = 'analysis: ' + analysis + '\nchannel: ' + channel + '\nsignal: ' + signal + '\nbackground: ' + str(background)
    if (preselectionCuts != 'none'):
        legendText += '\npreselection cuts: ' + preselectionCuts
    '''
    legend2 = plt.legend([emptyPlot], [legendText], loc = 'lower left', handlelength = 0, handletextpad = 0)
    for item in legend2.legendHandles:
        item.set_visible(False)
    plt.gca().add_artist(legend1)#, bbox_to_anchor = (1.05, 0.6))
    '''
    plt.figtext(0.77, 0.45, legendText, wrap = True, horizontalalignment = 'left')
    plt.subplots_adjust(left = 0.15, right = 0.75)
    pltName = outputDir + '/BkgRejectionVsMass_' + outputFileCommonName + '.png'
    plt.savefig(pltName)#, bbox_inches = 'tight')
    print('Saved ' + pltName)
    plt.subplots_adjust(left = 0.15, right = 0.95)
    plt.close()

#def DrawRejectionVsStat(massVec, fracTrain, WP, bkgRej90Dict, bkgRej94Dict, bkgRej97Dict, bkgRej99Dict, outputDir, jetCollection, analysis, channel, preselectionCuts, signal, background, outputFileCommonName):
def DrawRejectionVsStat(massVec, fracTrain, WP, bkgRej90Dict, bkgRej94Dict, bkgRej97Dict, bkgRej99Dict, outputDir, analysis, channel, preselectionCuts, signal, background, outputFileCommonName):
    emptyPlot, = plt.plot(massVec[0], bkgRej90Dict[fracTrain[0]], color = 'white')
    linestyles = ['loosely dotted', 'dotted', 'loosely dashed', 'solid']
    for i in range(len(fracTrain)):
        frac = fracTrain[i]
        print(bkgRej90Dict[frac])
        plt.plot(massVec, np.array(bkgRej90Dict[frac]), color = 'blue', label = 'WP: ' + str(WP[0]), marker = 'o', mec = 'blue', linestyle = linestyles[i])
        plt.plot(massVec, np.array(bkgRej94Dict[frac]), color = 'orange', label = 'WP: ' + str(WP[1]), marker = 'o', mec = 'orange', linestyle = linestyles[i])
        plt.plot(massVec, np.array(bkgRej97Dict[frac]), color = 'green', label = 'WP: ' + str(WP[2]), marker = 'o', mec = 'green', linestyle = linestyles[i])
        plt.plot(massVec, np.array(bkgRej99Dict[frac]), color = 'red', label = 'WP: ' + str(WP[3]), marker = 'o', mec = 'red', linestyle = linestyles[i])
    plt.yscale('log')
    legend1 = plt.legend()
    plt.xlabel('Mass [GeV]')
    plt.ylabel('Background rejection')
    legendText = 'analysis: ' + analysis + '\nchannel: ' + channel + '\nsignal: ' + signal + '\nbackground: ' + str(background)
    if (preselectionCuts != 'none'):
        legendText += '\npreselection cuts: ' + preselectionCuts
    '''
    legend2 = plt.legend([emptyPlot], [legendText], loc = 'lower left', handlelength = 0, handletextpad = 0)
    for item in legend2.legendHandles:
        item.set_visible(False)
    plt.gca().add_artist(legend1)#, bbox_to_anchor = (1.05, 0.6))
    '''
    plt.figtext(0.77, 0.45, legendText, wrap = True, horizontalalignment = 'left')
    plt.subplots_adjust(left = 0.15, right = 0.75)
    pltName = outputDir + '/BkgRejectionVsStat_' + outputFileCommonName + '.png'
    plt.savefig(pltName)#, bbox_inches = 'tight')
    print('Saved ' + pltName)
    plt.subplots_adjust(left = 0.15, right = 0.95)
    plt.close()


from sklearn.metrics import confusion_matrix
import itertools

#def DrawCM(yhat_test, y_test, w_test, outputDir, mass, background, outputFileCommonName, jetCollection, analysis, channel, preselectionCuts, signal, drawPlots):
def DrawCM(yhat_test, y_test, w_test, outputDir, mass, background, outputFileCommonName, analysis, channel, preselectionCuts, signal, drawPlots):
    yResult_test_cls = np.array([ int(round(x[0])) for x in yhat_test])
    cm = confusion_matrix(y_test, yResult_test_cls, sample_weight = w_test, normalize = 'true')
    TNR, FPR, FNR, TPR = cm.ravel()
    print(format('TNR: '  + str(TNR) + ', FPR: ' + str(FPR) + ', FNR: ' + str(FNR) + ', TPR: ' + str(TPR)))
    if drawPlots:
        classes = ['Background', 'Signal']
        np.set_printoptions(precision = 2)
        cmap = plt.cm.Oranges
        plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
        if mass >= 1000:
            titleCM = 'Confusion matrix (mass: ' + str(float(mass / 1000)) + ' TeV)'#, bkg: ' + background + ')'
        else:
            titleCM = 'Confusion matrix (mass: ' + str(int(mass)) + ' GeV)'#, bkg: ' + background + ')'
        plt.title(titleCM)
        #plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes, rotation = 90)
        #thresh = cm.max() / 2.
        thresh = 0.5
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], '.2f'), horizontalalignment = "center", color = "white" if cm[i, j] > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        CMPltName = outputDir + '/ConfusionMatrix_' + outputFileCommonName + '_' + str(mass) + '.png'
        legendText = 'analysis: ' + analysis + '\nchannel: ' + channel + '\nsignal: ' + signal
        if str(background) != 'all':
            legendText += '\nbackground: ' + str(background)
        if (preselectionCuts != 'none'):
            legendText += '\npreselection cuts: ' + preselectionCuts
        plt.figtext(0.77, 0.45, legendText, wrap = True, horizontalalignment = 'left')
        plt.subplots_adjust(left = 0.1, right = 0.75)
        plt.savefig(CMPltName)#, bbox_inches = 'tight')
        print('Saved ' + CMPltName)
        plt.subplots_adjust(left = 0.15, right = 0.95)
        plt.clf()
        plt.close()
    return TNR, FPR, FNR, TPR

def weightEventsOld(origin_train):
    originsList = np.array(list(set(list(origin_train))))
    originsNumber = np.array([])
    for origin in originsList:
        originsNumber = np.append(originsNumber, list(origin_train).count(origin))
    #print(originsNumber)
    minNumber = min(originsNumber)
    #weights = minNumber / originsNumber
    weights = minNumber / ( 2 * originsNumber)
    weights = np.where(weights == 0.5, 1, weights)
    #print(weights)
    w_origin_train = origin_train.copy()
    for origin in originsList:
        w_origin_train = np.where(w_origin_train == str(origin), weights[np.where(originsList == origin)], w_origin_train)
    w_origin_train = np.asarray(w_origin_train).astype(np.float32)
    return w_origin_train, originsList, originsNumber, weights


def weightEvents(origin_train, signal):
    DictNumbers = {}
    originsList = np.array(list(set(list(origin_train))))
    for origin in originsList:
        DictNumbers[origin] = list(origin_train).count(origin)
    minNumber = min(DictNumbers.values())
    minOrigin = [key for key in DictNumbers if DictNumbers[key] == minNumber]
    DictWeights = {}
    if signal in minOrigin:
        for origin in originsList:
            if origin == signal:
                DictWeights[origin] = 1
            else:
                DictWeights[origin] = minNumber / ((len(originsList) - 1) * DictNumbers[origin]) ### DictNumbers[signal]
    elif DictNumbers[signal] > ((len(originsList) - 1) * minNumber):
        for origin in originsList:
            if origin == signal:
                DictWeights[origin] = (len(originsList) - 1) * minNumber / DictNumbers[origin]
            elif origin in minOrigin:
                DictWeights[origin] = 1
            else:
                DictWeights[origin] = minNumber / DictNumbers[origin]
    else:
        print('No weights defined for this statistic')
    w_origin_train = origin_train.copy()
    for origin in originsList:
        w_origin_train = np.where(w_origin_train == origin, DictWeights[origin], w_origin_train)
    w_origin_train = np.asarray(w_origin_train).astype(np.float32)
    return w_origin_train, originsList, DictNumbers, DictWeights

'''
def cutEvents(data_train_mass):
    data_train_mass_cut = pd.DataFrame()
    originsList = np.array(list(set(list(data_train_mass['origin']))))
    originsNumber = np.array([])
    for origin in originsList:
        originsNumber = np.append(originsNumber, list(data_train_mass['origin']).count(origin))
    minNumber = int(min(originsNumber))
    for origin in originsList:
        data_train_mass_origin = data_train_mass[data_train_mass['origin'] == origin]
        if origin == 'Radion':
            data_train_mass_origin = data_train_mass_origin
        else:
            data_train_mass_origin = data_train_mass_origin[:int(minNumber)]
        data_train_mass_cut = pd.concat([data_train_mass_cut, data_train_mass_origin], ignore_index = True)
    return data_train_mass_cut
'''

def defineBins(regime):
    #print(regime)
    bins = []
    if (all(x in regime for x in ['SR', 'Res', 'GGF', 'Tag'])):
        bins = [300, 320, 350, 380, 410, 440, 480, 520, 560, 600, 650, 700, 750, 810, 870, 940, 1010, 1090, 1170, 1260, 1360, 1460, 1650, 3000]
        #print('found1')
    if (all(x in regime for x in ['SR', 'Res', 'GGF', 'Untag'])):
        bins = [300, 320, 350, 380, 410, 440, 480, 520, 560, 600, 650, 700, 750, 810, 870, 940, 1010, 1090, 1170, 1260, 1360, 1460, 1570, 1690, 1820, 1960, 2110, 3000]
        #print('found2')
    if (all(x in regime for x in ['SR', 'Res', 'GGF', 'WZ'])):
        bins = [300, 320, 350, 380, 410, 440, 470, 500, 530, 560, 600, 640, 680, 720, 770, 820, 870, 930, 990, 1060, 1130, 1210, 1290, 1380, 1470, 1570, 1680, 1790, 2140, 3000]
        #print('found11')
    if (all(x in regime for x in ['SR', 'Res', 'VBF', 'ZZ'])):
        bins = [300, 320, 350, 380, 410, 440, 470, 500, 540, 580, 620, 660, 710, 760, 810, 870, 930, 990, 1060, 1130, 1210, 1290, 1380, 1470, 1630, 1790, 3000]
        #print('found3')
    if (all(x in regime for x in ['SR', 'Res', 'VBF', 'WZ'])):
        bins = [300, 320, 350, 380, 410, 440, 470, 500, 540, 580, 620, 660, 710, 760, 810, 860, 920, 980, 1040, 1110, 1180, 1250, 1330, 1410, 1640, 1870, 3000]
        #print('found14')
    if (all(x in regime for x in ['SR', 'Merg', 'GGF', 'Tag'])):
        #print('found4')
        bins = [500, 530, 570, 610, 650, 690, 730, 770, 810, 850, 890, 930, 970, 1020, 1070, 1120, 1170, 1220, 1270, 1330, 1480, 1630, 1780, 1930, 2080, 2380, 6000]
    if (all(x in regime for x in ['SR', 'Merg', 'GGF', 'Untag'])):
        #print('found5')
        bins = [500, 530, 570, 610, 650, 690, 730, 770, 810, 850, 890, 930, 970, 1020, 1070, 1120, 1170, 1220, 1270, 1330, 1390, 1450, 1510, 1570, 1640, 1710, 1780, 1850, 1920, 2000, 2080, 2160, 2250, 2340, 2430, 2520, 2620, 2720, 2820, 3120, 3420, 4910, 6000]
    if (all(x in regime for x in ['SR', 'Merg', 'GGF', 'WZ'])):
        #print('found12')
        bins = [500, 530, 570, 610, 650, 690, 730, 770, 810, 850, 890, 930, 970, 1010, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1410, 1470, 1530, 1590, 1650, 1720, 1790, 1860, 1930, 2000, 2080, 2160, 2240, 2330, 2510, 2690, 2870, 3320, 3770, 6000]
    if (all(x in regime for x in ['SR', 'Merg', 'VBF', 'ZZ'])):
        #print('found6')
        bins = [500, 530, 570, 610, 650, 690, 730, 770, 810, 860, 910, 960, 1010, 1070, 1130, 1190, 1260, 1340, 1420, 1530, 1680, 1830, 1980, 6000]
    if (all(x in regime for x in ['SR', 'Merg', 'VBF', 'WZ'])):
        #print('found13')
        bins = [500, 530, 570, 610, 650, 690, 730, 770, 810, 850, 890, 930, 970, 1010, 1050, 1090, 1130, 1170, 1210, 1250, 1300, 1350, 1410, 1470, 1630, 2000, 2370, 6000]
    if (all(x in regime for x in ['CR', 'Res', 'GGF', 'ZCR'])):
        #print('found7')
        bins = [300, 320, 350, 380, 410, 440, 470, 500, 530, 570, 610, 660, 740, 900, 3000]
    if (all(x in regime for x in ['CR', 'Res', 'VBF', 'ZCR'])):
        #print('found8')
        bins = [300,  3000]
    if (all(x in regime for x in ['CR', 'Merg', 'GGF', 'ZCR'])):
        #print('found9')
        bins = [500, 6000]
    if (all(x in regime for x in ['CR', 'Merg', 'VBF', 'ZCR'])):
        #print('found10')
        bins = [500, 6000]

    return bins


def defineFixBins(lowerEdge, upperEdge, percLeft, percRight, mass):
    Bins2 = np.linspace(percLeft, percRight, 8)
    '''
    if mass == 500:
        Bins1 = np.linspace(lowerEdge, percRight, 8)
        Bins2 = np.array([])
        Bins3 = np.linspace(percRight, upperEdge, 11)
    if mass == 600 or mass == 700 or mass == 800:
        Bins1 = np.linspace(lowerEdge, percLeft, 2)
        Bins3 = np.linspace(percRight, upperEdge, 10)        
    if mass == 1000 or mass == 1200:
        Bins1 = np.linspace(lowerEdge, percLeft, 3)
        Bins3 = np.linspace(percRight, upperEdge, 9)
    if mass == 1400 or mass == 1500 or mass == 1600 or mass == 1800 or mass == 2000:
        Bins1 = np.linspace(lowerEdge, percLeft, 4)
        Bins3 = np.linspace(percRight, upperEdge, 8)
    if mass == 2400 or mass == 2600:
        Bins1 = np.linspace(lowerEdge, percLeft, 5)
        Bins3 = np.linspace(percRight, upperEdge, 7)
    if mass == 3000:
        Bins1 = np.linspace(lowerEdge, percLeft, 9)
        Bins3 = np.linspace(percRight, upperEdge, 3)
    if mass == 3500:
        Bins1 = np.linspace(lowerEdge, percLeft, 10)
        Bins3 = np.linspace(percRight, upperEdge, 2)
    if mass == 4000:
        Bins1 = np.linspace(lowerEdge, percRight, 11)
        Bins2 = np.array([])
        Bins3 = np.linspace(percRight, upperEdge, 8)
    Bins = np.array([])
    Bins = np.append(Bins, Bins1)
    Bins = np.append(Bins, Bins2)
    Bins = np.append(Bins, Bins3)
    Bins = np.sort(np.array(list(set(Bins))))
    print(Bins)
    binPercLeft = np.digitize(percLeft, Bins)
    binPercRight = np.digitize(percRight, Bins)
    '''
    Bins = np.linspace(percLeft, percRight, 8)
    return Bins#, binPercLeft, binPercRight

def sortColumns(values, weights, Reverse = False):
    zipped_lists = zip(list(values), list(weights))
    sorted_zipped_lists = sorted(zipped_lists, reverse = Reverse)
    sortedValues = np.array([value for value, _ in sorted_zipped_lists])
    sortedWeights = np.array([weights for _, weights in sorted_zipped_lists])
    return sortedValues, sortedWeights

import math
#def defineVariableBins(bkgEvents, weightsBkgEvents, lowerMass, upperMass, resolutionRangeLeft, resolutionRangeRight, feature, nBinsMass = None, bkgEventsInResolution= None):
def defineVariableBins(bkgEvents, weightsBkgEvents, resolutionRangeLeft, resolutionRangeRight, feature, nBinsMass = None, bkgEventsInResolution = None):
    if feature == 'Scores':
        bkgEvents, weightsBkgEvents = sortColumns(bkgEvents, weightsBkgEvents, True)
    bkgErrorSquared = 0
    resolution = (resolutionRangeRight - resolutionRangeLeft) / 6
    print('Resolution: ' + str(resolution))
    if feature == 'InvariantMass':
        #leftEdge = lowerMass
        leftEdge = resolutionRangeLeft
        #Bins = np.array([resolutionRangeLeft])
        Bins = np.array([leftEdge])
        bkgMassEventsInResolution = 0
    elif feature == 'Scores':
        #leftEdge = upperMass
        leftEdge = 1
        Bins = np.array([leftEdge])
        bkgScoresEventsInResolution = 0
    
    weightsSum = 0.
    bkgUncertainty = 0.7
    weightsSumSquared = 0.

    bkgEventsResolutionArray = []
    weightsBkgEventsResolutionArray = []
    
    for bkg, weight in zip(bkgEvents, weightsBkgEvents):
        if bkg >= resolutionRangeLeft and bkg <= resolutionRangeRight:
        #if bkg >= lowerMass and bkg <= upperMass:
            bkgEventsResolutionArray.insert(len(bkgEventsResolutionArray), bkg)
            weightsBkgEventsResolutionArray.insert(len(weightsBkgEventsResolutionArray), weight)

    if feature == 'InvariantMass':
        bkgMassEventsInResolution = sum(weightsBkgEventsResolutionArray)
        print('# weighted bkg events in resolution range ' + str(bkgMassEventsInResolution))
        print('# raw bkg events in resolution range:', len(bkgEventsResolutionArray))
    elif feature == 'Scores':
        bkgScoresEventsInResolution = sum(weightsBkgEventsResolutionArray)

    '''
    for bkg, weight in zip(bkgEvents, weightsBkgEvents):
    #for bkg, weight in zip(bkgEventsResolutionArray, weightsBkgEventsResolutionArray):
        weightsSum += weight# * weight
        if weightsSum <= 0:
            continue
        bkgError = 1 / math.sqrt(weightsSum)
        if feature == 'InvariantMass':
            
            if bkg >= resolutionRangeLeft and bkg <= resolutionRangeRight:
                bkgMassEventsInResolution += weight ### non posso metterlo qui se ho il continue prima? 
                #print(bkgMassEventsInResolution)
            
            if bkg >= resolutionRangeLeft and abs(bkg - leftEdge) >= resolution and bkgError <= bkgUncertainty and bkg <= resolutionRangeRight:# and bkg < upperMass: ## main
                Bins = np.append(Bins, bkg)
                print('Found bin edgeeeeeeeeeeeeeeeeeeeeeeeeee: ' + str(bkg))
                print('Left edge: ' + str(leftEdge))
                print('Difference: ' + str(abs(bkg - leftEdge))) ###???
                print('Error: ' + str(bkgError))
                print('weightSum: ' + str(weightsSum))
                print(Bins)
                weightsSum = 0
                leftEdge = bkg
        elif feature == 'Scores':
            if bkg >= resolutionRangeLeft and bkg <= resolutionRangeRight:
                bkgScoresEventsInResolution += weight
            
    if feature == 'InvariantMass':
        #Bins = np.append(Bins, upperMass)
        Bins = np.append(Bins, resolutionRangeRight)
        print('bins after resolutionRangeRight: ' + str(Bins))
        #Bins = np.append(Bins, np.linspace(resolutionRangeRight, upperMass, 4))
    elif feature == 'Scores':
        #Bins = np.append(Bins, lowerMass)
        #Bins = np.append(Bins, resolutionRangeLeft)
        print(nBinsMass)
        print(bkgScoresEventsInResolution)
        print(bkgEventsInResolution)
        nBins = round(nBinsMass * bkgScoresEventsInResolution / bkgEventsInResolution)
        if nBins == 0:
            nBins = 1
        print(nBins)
        Bins = np.linspace(resolutionRangeLeft, resolutionRangeRight, nBins + 1)
    Bins = np.sort(Bins)
        #Bins = np.append(Bins, np.linspace(lowerMass, resolutionRangeLeft, 11))
        #Bins = np.sort(np.array(list(set(list(Bins)))))
    '''

    if feature == 'InvariantMass':# or feature == 'Scores':
        for bkg, weight in zip(bkgEventsResolutionArray, weightsBkgEventsResolutionArray):
            #print('bkg:', bkg)
            #print('weight:', weight)
            #weightsSum += weight --> eventi bkg attesi secondo il MC che nei dati fluttuerà come sqrt(weightsSum) 
            weightsSumSquared += weight * weight # --> errore sulla predizione MC sul numero di eventi 
            weightsSum += weight
            if weightsSum <= 0:
                continue
            #bkgError = 1 / math.sqrt(weightsSum)        
            relativeBkgError = (math.sqrt(weightsSumSquared)) / weightsSum

            ''' ## Rob
            sumWeightsSquared = 0.
            sumWeightsSquared += weight * weight
            if weightsSum <= 0:
                continue
            relativeBkgError = 1 / math.sqrt(weightsSum)
            '''

            #if bkg >= resolutionRangeLeft and abs(bkg - leftEdge) >= resolution and bkgError <= bkgUncertainty and bkg <= resolutionRangeRight:# and bkg < upperMass: ## main
            if (bkg - leftEdge) >= resolution and relativeBkgError <= bkgUncertainty:
                Bins = np.append(Bins, bkg)
                print('Left edge: ' + str(leftEdge))
                print('Difference: ' + str(abs(bkg - leftEdge))) ###???
                print('Error: ' + str(relativeBkgError))
                print('weightSum: ' + str(weightsSum))
                print(Bins)
                weightsSum = 0
                weightsSumSquared = 0
                leftEdge = bkg
        Bins = np.append(Bins, resolutionRangeRight)
        print('Bins: ' + str(Bins))
        return Bins, bkgMassEventsInResolution


    elif feature == 'Scores':
        print('nBinsMass: ', nBinsMass)
        print('bkgScoresEventsInResolution:', bkgScoresEventsInResolution)
        print('bkgEventsInResolution:', bkgEventsInResolution)
        if bkgEventsInResolution <= 0 or bkgScoresEventsInResolution <= 0:
            nBins = 1
        else:
            nBins = round(nBinsMass * bkgScoresEventsInResolution / bkgEventsInResolution)
            if nBins == 0:
                nBins = 1
        print('nBins:', nBins)
        Bins = np.linspace(resolutionRangeLeft, resolutionRangeRight, nBins + 1)
        print('Bins: ' + str(Bins))
        return(Bins)

    '''
    #Bins = np.sort(Bins)
    #print(Bins)
    if feature == 'InvariantMass':
        Bins = np.append(Bins, upperMass)
        return Bins, bkgMassEventsInResolution
    else:
        Bins = np.append(Bins, 0)
        Bins = np.sort(Bins)
         return Bins
    '''

def defineVariableBinsNew(bkgEvents, weightsBkgEvents, resolution, leftEdge, rightEdge, feature, nBinsMass = None, bkgEventsInResolution = None):
    '''
    if feature == 'Scores':
        bkgEvents, weightsBkgEvents = sortColumns(bkgEvents, weightsBkgEvents, True)
    '''
    bkgErrorSquared = 0
    resolutionFraction = resolution / 6

    weightsSum = 0.
    bkgUncertainty = 0.7
    weightsSumSquared = 0.
    Bins = np.array([leftEdge])
    #print('Bins:', Bins)
    bkgEventsResolutionArray = []
    weightsBkgEventsResolutionArray = []
    
    if feature == 'InvariantMass':
        bkgMassEventsInResolution = sum(weightsBkgEvents)
        print('# weighted bkg events in invariant mass distribution: ' + str(bkgMassEventsInResolution))
        #print('# raw bkg events:', len(bkgEventsResolutionArray))
    elif feature == 'Scores':
        bkgScoresEventsInResolution = sum(weightsBkgEvents)
        print('# weighted bkg events in scores distribution: ' + str(bkgScoresEventsInResolution))
        #print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa i due numeri sopra devono essere uguali!!!')

    if feature == 'InvariantMass':# or feature == 'Scores':
        reachedLastElement = False
        for bkg, weight in zip(bkgEvents, weightsBkgEvents):
            #weightsSum += weight --> eventi bkg attesi secondo il MC che nei dati fluttuerà come sqrt(weightsSum) 
            weightsSumSquared += weight * weight # --> errore sulla predizione MC sul numero di eventi 
            weightsSum += weight
            if weightsSum <= 0 and bkg != bkgEvents[len(bkgEvents) - 1]:
                continue

            if bkg == bkgEvents[len(bkgEvents) - 1]: ### when we reach the last bkg event we check that the bin is large enough to meet the requirements we set, otherwise we remove the last binEdge from the bin array
                reachedLastElement = True
                if weightsSum > 0:
                    relativeBkgError = (math.sqrt(weightsSumSquared)) / weightsSum
                else: 
                    relativeBkgError = 0
                #print('Last bkg eventttttttttttttttttttttttttttttt')
                if weightsSum <= 0 or (bkg - leftEdge) < resolutionFraction or relativeBkgError > bkgUncertainty:
                    #print('Bins before deletinggggggg:' + str(Bins))
                    #print('Removing last binnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn')
                    Bins = np.delete(Bins, -1)
                    #print(Bins)

            if reachedLastElement == True:
                break

            #bkgError = 1 / math.sqrt(weightsSum)        
            relativeBkgError = (math.sqrt(weightsSumSquared)) / weightsSum

            ''' ## Rob
            sumWeightsSquared = 0.
            sumWeightsSquared += weight * weight
            if weightsSum <= 0:
                continue
            relativeBkgError = 1 / math.sqrt(weightsSum)
            '''

            if (bkg - leftEdge) >= resolutionFraction and relativeBkgError <= bkgUncertainty and bkg != bkgEvents[len(bkgEvents) - 1]:
                Bins = np.append(Bins, bkg)
                '''
                print('Left edge: ' + str(leftEdge))
                print('Difference: ' + str(abs(bkg - leftEdge))) ###???
                print('Error: ' + str(relativeBkgError))
                print('weightSum: ' + str(weightsSum))
                print(Bins)
                '''
                weightsSum = 0
                weightsSumSquared = 0
                leftEdge = bkg

        Bins = np.append(Bins, rightEdge)
        #print('Bins: ' + str(Bins))
        return Bins, bkgMassEventsInResolution


    elif feature == 'Scores':
        print('nBinsMass: ', nBinsMass)
        print('bkgScoresEventsInResolution:', bkgScoresEventsInResolution)
        print('bkgEventsInResolution:', bkgEventsInResolution)
        if bkgEventsInResolution <= 0 or bkgScoresEventsInResolution <= 0:
            nBins = 1
        else:
            nBins = round(nBinsMass * bkgScoresEventsInResolution / bkgEventsInResolution)
            if nBins == 0:
                nBins = 1
        #print('nBins:', nBins)
        Bins = np.linspace(leftEdge, rightEdge, nBins + 1)
        BinsArray = np.array([])
        for iBin in Bins:
            BinsArray = np.append(BinsArray, iBin)
        BinsArray = np.sort(BinsArray)

        print('Bins array before deleting: ' + str(BinsArray))
        bkgContents, bkgEdges, _ = plt.hist(bkgEvents, weights = weightsBkgEvents, bins = BinsArray)
        print('Bkg contents: ' + str(bkgContents))
        plt.clf()
        #numberOfZeros = len(bkgContents[np.where(bkgContents == 0)])
        numberOfZeros = len(bkgContents[np.where(bkgContents <= 0)])
        while numberOfZeros != 0:
            print('numberOfZeros:', numberOfZeros)
            #zeroIndex = np.where(bkgContents == 0)[0][0]
            zeroIndex = np.where(bkgContents <= 0)[0][0]
            print(zeroIndex)
            print('Deleting edge ' + str(BinsArray[zeroIndex + 1]))
            #bkgEdges = np.delete(bkgEdges, zeroIndex + 1)
            '''
            if zeroIndex != (len(BinsArray) - 2):
                BinsArray = np.delete(BinsArray, zeroIndex + 1)
            '''
            if zeroIndex == 0:
                BinsArray = np.delete(BinsArray, zeroIndex + 1)
            else:
                BinsArray = np.delete(BinsArray, zeroIndex)
            #BinsArray = np.delete(BinsArray, zeroIndex)
            bkgContents, bkgEdges, _ = plt.hist(bkgEvents, weights = weightsBkgEvents, bins = BinsArray)
            plt.clf()
            print(bkgContents)
            #numberOfZeros = len(bkgContents[np.where(bkgContents == 0)]) ### This is the value that will be checked at the next loop
            numberOfZeros = len(bkgContents[np.where(bkgContents <= 0)]) ### This is the value that will be checked at the next loop

        '''
            for bkgContent, bkgEdge in zip(bkgContents, bkgEdges):
                if bkgContent == 0:
                    zeroIndex = np.where(bkgContent == 0)[0]
                    trova posizione di questo all'interno del vettore di content e rimuovi l'elemento +1 dal vettore di edges
        ''' 
        '''
        for iBin in range(len(BinsArray) - 1):
            firstEdge = BinsArray[iBin]
            secondEdge = BinsArray[iBin + 1]
        '''
        
        print('Bins: ' + str(BinsArray))
        return(BinsArray)


def computeBins(bkgEvents, weightsBkgEvents, resolutionRangeLeft, resolutionRangeRight, feature):
    '''
    if feature == 'Scores':
        bkgEvents, weightsBkgEvents = sortColumns(bkgEvents, weightsBkgEvents, True)
    '''
    if feature == 'Scores':
        bkgEvents, weightsBkgEvents = sortColumns(bkgEvents, weightsBkgEvents)
    Bins = np.array([resolutionRangeLeft, resolutionRangeRight])
    #print(bkgEvents)

    bkgEventsResolutionArray = []
    weightsBkgEventsResolutionArray = []

    for bkg, weight in zip(bkgEvents, weightsBkgEvents):
        if bkg >= resolutionRangeLeft and bkg <= resolutionRangeRight:
        #if bkg >= lowerMass and bkg <= upperMass:
            bkgEventsResolutionArray.insert(len(bkgEventsResolutionArray), bkg)
            weightsBkgEventsResolutionArray.insert(len(weightsBkgEventsResolutionArray), weight)
    #print(bkgEventsResolutionArray)
    '''
    if feature == 'InvariantMass':
        bkgMassEventsInResolution = sum(weightsBkgEventsResolutionArray)
        print('# weighted bkg events in resolution range ' + str(bkgMassEventsInResolution))
        print('# raw bkg events in resolution range:', len(bkgEventsResolutionArray))
    elif feature == 'Scores':
        bkgScoresEventsInResolution = sum(weightsBkgEventsResolutionArray)
    '''
    totalBkgEvents = sum(weightsBkgEventsResolutionArray)
    bkgLimit = 0.007 * totalBkgEvents
    if bkgLimit <= 0:
        plotMass = False
        print('Skipping mass beacuse bkg content in region with 95% of the signal is <= 0: ' + str(totalBkgEvents))
    else:
        plotMass = True
        bkgUpperLimit = 0.05 * totalBkgEvents
        weightsSum = 0.
        weightsSum2 = 0.
        print('totalBkgEvents:', totalBkgEvents)
        print('bkgLimit:', bkgLimit)
    
        for bkg, weight in zip(bkgEventsResolutionArray, weightsBkgEventsResolutionArray):
            #weightsSum += weight
            #if weightsSum >= bkgLimit:
            if weightsSum >= bkgUpperLimit:
                #print('found weightsSum: ', weightsSum)
                #print('bin edge:', bkg)
                Bins = np.append(Bins, bkg)
                #print(Bins)
                weightsSum = weight
            else:
                weightsSum += weight
                '''
                if bkg == bkgEventsResolutionArray[len(bkgEventsResolutionArray) - 1] and weightsSum < bkgLimit: ### when we reach the last bkg event we check that the bin is large enough to meet the requirements we set, otherwise we remove the last binEdge from the bin array
                print('Bins before deletinggggggg:' + str(Bins))
                print('Removing last binnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn')
                Bins = np.delete(Bins, -1)
                #print(Bins)
                '''
            if bkg == bkgEventsResolutionArray[len(bkgEventsResolutionArray) - 1] and weightsSum < bkgUpperLimit:
                #cutBkg = bkgEventsResolutionArray[bkgEventsResolutionArray > Bins[len(Bins) - 1]]
                for bkg2, weight2 in zip(bkgEventsResolutionArray, weightsBkgEventsResolutionArray):
                    if bkg2 < Bins[len(Bins) - 1]:
                        continue
                    #weightsSum2 += weight2
                    if weightsSum2 >= bkgLimit:
                        #print('found weightsSum: ' + str(weightsSum2) + ' (bkg limit: ' + str(bkgLimit) + ')')
                        #print('bin edge:', bkg2)
                        Bins = np.append(Bins, bkg2)
                        weightsSum2 = weight2
                    else:
                        weightsSum2 += weight2
                    
            
        Bins = np.sort(Bins)
        print('Bins: ' + str(Bins))
        bkgEvents, _, _ = plt.hist(bkgEventsResolutionArray, weights = np.array(weightsBkgEventsResolutionArray), bins = Bins)
        print('bkg contents from functions: ' + str(bkgEvents))
        plt.clf()

        numberOfZeros = len(bkgEvents[np.where(bkgEvents <= 0)])
        while numberOfZeros != 0:
            print('numberOfZeros:', numberOfZeros)
            #zeroIndex = np.where(bkgContents == 0)[0][0]
            zeroIndex = np.where(bkgEvents <= 0)[0][0]
            print(zeroIndex)
            print('Deleting edge ' + str(Bins[zeroIndex + 1]))
            #bkgEdges = np.delete(bkgEdges, zeroIndex + 1)
            '''
            if zeroIndex != (len(BinsArray) - 2):
                BinsArray = np.delete(BinsArray, zeroIndex + 1)
            '''
            if zeroIndex == 0:
                Bins = np.delete(Bins, zeroIndex + 1)
            else:
                Bins = np.delete(Bins, zeroIndex)
            #BinsArray = np.delete(BinsArray, zeroIndex)
            bkgEvents, _, _ = plt.hist(bkgEventsResolutionArray, weights = np.array(weightsBkgEventsResolutionArray), bins = Bins)
            plt.clf()
            print(bkgEvents)
            numberOfZeros = len(bkgEvents[np.where(bkgEvents <= 0)]) ### This is the value that will be checked at the next loop
        '''
        while bkgEvents[len(bkgEvents) - 1] < bkgLimit:
            print('Bins before deletinggggggg:' + str(Bins))
            print('Removing last binnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn')
            Bins = np.delete(Bins, -2)
            bkgEvents, _, _ = plt.hist(bkgEventsResolutionArray, weights = np.array(weightsBkgEventsResolutionArray), bins = Bins)
            print('bkg contents from functions: ' + str(bkgEvents))
            plt.clf()
        '''
        print('Bins: ' + str(Bins))            

    return Bins, plotMass

def defineVariableBinsOld(bkgEvents, weightsBkgEvents, lowerMass, upperMass, resolutionRangeLeft, resolutionRangeRight):
    resolution = resolutionRangeRight - resolutionRangeLeft
    zipped_lists = zip(list(bkgEvents), list(weightsBkgEvents))
    print('sorting')
    sorted_zipped_lists = sorted(zipped_lists)
    print('done sorting')
    bkgErrorSquared = 0
    #Bins1 = np.linspace(lowerMass, resolutionRangeLeft, 6)
    #Bins = np.array([lowerMass, resolutionRangeLeft])
    leftEdge = lowerMass
    Bins = np.array([lowerMass])
    '''
    for bkg, weight in zip(valuesSorted, weightsSorted):
        bkgErrorSquared += weight ** 2
        print(str(bkg - leftEdge) + ' --- ' + str(bkgErrorSquared))
    '''
    for bkg, weight in sorted_zipped_lists:
        bkgErrorSquared += weight ** 2
        #if (bkg - leftEdge) >= resolution and bkgErrorSquared <= 0.5 ** 2 and bkg <= upperMass: ### TODO check the last end with high mass
        #if (bkg - leftEdge) >= resolution and bkgErrorSquared <= 0.7 ** 2 and bkg <= resolutionRangeRight: ### TODO check the last end with high mass
        if (bkg - leftEdge) >= resolution and bkgErrorSquared <= 0.7 ** 2 and bkg <= upperMass:
            #print('Found bin')
            Bins = np.append(Bins, bkg)
            bkgErrorSquared = 0
            leftEdge = bkg
    Bins = np.append(Bins, upperMass)
    #Bins = np.append(Bins, np.linspace(resolutionRangeRight, upperMass, 4))
    #print(Bins)
    
    return Bins


### Copmuting weighted F1 score ---> WRONG!!!
from tensorflow.keras import backend as K
def computeWeightedMetrics(y_true, y_pred):
    dataTrainInput = pd.read_pickle(dfPath + '/data_train_' + tag + '_' + jetCollection + '_' + analysis + '_' + channel + '_' + preselectionCuts + '_' + str(signal) + '_' + background + '_' + str(trainingFraction) + 't.pkl')
    w_train = dataTrainInput['train_weight'].values ### how can I pass the vector here?
    y_true_len = int(len(y_true))
    print('\ny_true_len: ' + str(y_true_len))
    if(y_true_len > int(len(w_train) / 2)): ### Safe assumption (I think) NO!
        print('-------------------- Evaluation on the train sample ------------------------')
        w_train = w_train[:y_true_len]
    else:
        print('\n-------------------- Evaluation on the validation sample -------------------------')
        w_train = w_train[(y_true_len + 1):]
    print('Y_true from backend\n', K.get_value(y_true))
    print('Y_pred from backend\n', K.get_value(y_pred))
    print('w_train: ' + str(w_train))
    true_positives = 0
    possible_positives = 0
    predicted_positives = 0
    for event in range(y_true_len):
        true_positives += y_true[event] * K.round(y_pred[event]) * w_train[event] ##serve clip?
        possible_positives += y_true[event] * w_train[event] ##serve clip? sono i positive da LABEL
        predicted_positives += K.round(y_pred[event]) * w_train[event]
    recall = true_positives / (possible_positives + K.epsilon()) ### can I safely ignore K.epsilon()?
    precision = true_positives / (predicted_positives + K.epsilon())
    print('True positives: '  + str(true_positives))
    print('Possible (MC) positives: '  + str(possible_positives))
    print('Predicted positives: '  + str(predicted_positives))
    print('Recall (TP / possible positives): ' + str(recall))
    print('Precision (TP / predicted positives): ' + str(precision))
    return precision, recall

### Metric definition
def f1_score(y_true, y_pred):
    precision, recall = computeWeightedMetrics(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def weighted_percentileOld(values, weights, percentileLeft, percentileRight = 100): ### TODO BETTER! USING ZIP TO SORT
    valuesList = list(values)
    valuesCopy = values.copy()
    valuesSorted = np.sort(valuesCopy)
    weightsSorted = np.array([])
    for value in valuesSorted:
        valueIndex = valuesList.index(value)
        weightsSorted = np.append(weightsSorted, weights.iloc[valueIndex])        
    sumWeights = weights.sum()
    areaLeft = sumWeights * percentileLeft / 100
    areaRight = sumWeights * percentileRight / 100
    cumulativeSum = 0
    resolutionSum = 0
    for index in range(len(weightsSorted)):
        #cumulativeSum += weightsSorted[index]
        if cumulativeSum <= areaLeft:
            cumulativeSum += weightsSorted[index]
            percentileLeftIndex = index
        if resolutionSum <= 17.5 * sumWeights / 100:
            resolutionSum += weightsSorted[index]
            resolutionRangeLeft = index
        if percentileRight != 100 and resolutionSum >= (16 * sumWeights / 100) and resolutionSum <= (84 * sumWeights / 100):
            resolutionRangeRight = index
            resolutionSum += weightsSorted[index]
        if percentileRight != 100 and cumulativeSum >= areaLeft and cumulativeSum <= areaRight:
            cumulativeSum += weightsSorted[index]
            percentileRightIndex = index
        #else:
        #    break

    percentileXleft = valuesSorted[percentileLeftIndex]
    resolutionXleft = valuesSorted[resolutionRangeLeft]
    if percentileRight != 100:
        percentileXright = valuesSorted[percentileRightIndex]
        resolutionXright = valuesSorted[resolutionRangeRight]
    else:
        percentileXright = None
        resolutionXright = 1
    #print('Area left:', areaLeft)
    resolution = resolutionXright - resolutionXleft
    print('resolutionLeft:', resolutionXleft)
    print('resolutionRight:', resolutionXright)
    #print(resolution)
    return percentileXleft, percentileXright, resolutionXleft, resolutionXright

def weighted_percentile(values, weights, feature):
    sortedValues, sortedWeights = sortColumns(values, weights)
    sumWeights = np.array(sortedWeights).sum()
    #print('sumWeights:', sumWeights)
    resolutionSum = 0
    resolutionRangeLeft = 0
    #foundLeftEdge = False
    for (value, weight) in zip(sortedValues, sortedWeights):
        #print('value:', value)
        resolutionSum += weight
        #print('resolutionSum:', resolutionSum)
        if resolutionSum <= (0.5 * sumWeights): ## 0.16
            median = value
        if feature == 'InvariantMass' and resolutionSum <= (0.025 * sumWeights):
            resolutionRangeLeft = value
            #foundLeftEdge = True
            #print('updating resolutionRangeLeft to ', resolutionRangeLeft)
        if feature == 'InvariantMass' and resolutionSum > (0.5 * sumWeights) and resolutionSum <= (0.975 * sumWeights):
            resolutionRangeRight = value
            #print('updating resolutionRangeRight to', resolutionRangeRight)
        if feature == 'Scores' and resolutionSum <= (0.05 * sumWeights): ## 0.32
            resolutionRangeLeft = value
    if feature == 'Scores':
        resolutionRangeRight = 1.
    '''
    if (median - sortedValues[0]) > (sortedValues[len(sortedValues) - 1] - median):
        diff = median - sortedValues[0]
    elif (median - sortedValues[0]) < (sortedValues[len(sortedValues) - 1] - median):
        diff = sortedValues[len(sortedValues) - 1] - median
    diff = 0
    '''
    '''
    if foundLeftEdge == False:
        resolutionRangeLeft = sortedValues[0]
    '''
    if resolutionRangeLeft == 0:
        resolutionRangeLeft = min(sortedValues)
    print('resolutionLeft:', resolutionRangeLeft)
    print('resolutionRight:', resolutionRangeRight)
    print('median:', median)
    return resolutionRangeLeft, resolutionRangeRight#, diff

def defineBinsNew(hist_signal_norm, signalMCweightsMass):
    sortedValues, sortedWeights = sortColumns(hist_signal_norm, signalMCweightsMass, True)
    sumWeights = np.array(sortedWeights).sum()
    cumulativeSum = 0.
    #percArray = np.array([0.3, 0.2, 0.2, 0.1, 0.1]) * sumWeights
    percArray = np.full(0.05, 20)# * sumWeights
    print(percArray)
    exit()
    print(sumWeights)
    print(percArray)
    iPerc = 0
    Bins = np.array([1.])
    for (value, weight) in zip(sortedValues, sortedWeights):
        cumulativeSum += weight
        if cumulativeSum <= percArray[iPerc]:
            binEdge = value
        else:
            Bins = np.append(Bins, binEdge)
            iPerc += 1
            cumulativeSum = 0.
        if iPerc == len(percArray):
            break
    Bins = np.append(Bins, 0.)
    Bins = np.sort(Bins)
    return Bins

import json
def scaleVariables(modelDir, dataFrameSignal, dataFrameBkg, inputFeatures, outputDir):
    variablesFileName = 'variables.json'
    variablesFile = modelDir + variablesFileName 
    print('Loading ' + variablesFile)
    jsonFile = open(variablesFile, 'r')
    values = json.load(jsonFile)
    for field in values['inputs']:
        feature = field['name']
        if feature not in inputFeatures:
            continue
        print('Scaling ' + feature)
        offset = field['offset']
        scale = field['scale']
        dataFrameSignal[feature] = (dataFrameSignal[feature] - offset) / scale
        if feature != 'mass':
            dataFrameBkg[feature] = (dataFrameBkg[feature] - offset) / scale
    jsonFile.close()
    shutil.copyfile(variablesFile, outputDir + variablesFileName)
    print('Copied variables file to ' + outputDir + variablesFileName)
    return dataFrameSignal, dataFrameBkg

### Scaling train dataset according to the scale factor saved in variables.json file
def scaleTrainTestDataset(dataTrain, variablesDir, inputFeatures, sampleType):
    variablesFileName = 'variables.json'
    variablesFile = variablesDir + variablesFileName 
    print('Loading scaling factors from ' + variablesFile)
    jsonFile = open(variablesFile, 'r')
    values = json.load(jsonFile)
    for field in values['inputs']:
        feature = field['name']
        if feature not in inputFeatures:
            continue
        print('Scaling ' + feature + ' for ' + sampleType + ' dataFrame')
        offset = field['offset']
        #print('Offset: ' + str(offset))
        scale = field['scale']
        #print('Scale: ' + str(scale))
        dataTrain[feature] = (dataTrain[feature] - offset) / scale
    jsonFile.close()
    '''
    outputFileName = outputDir + '/' + variablesFileName
    shutil.copyfile(variablesFile, outputFileName)
    print('Copied variables file to ' + outputFileName)
    '''
    return dataTrain


from keras.utils.vis_utils import plot_model
def loadModelAndWeights(modelDir, outputDir):
    from keras.models import model_from_json
    architectureFileName = 'architecture.json'
    architectureFile = modelDir + architectureFileName
    with open(architectureFile, 'r') as json_file:
        print('Loading ' + architectureFile)
        model = model_from_json(''.join(json_file.readlines()))
    shutil.copyfile(architectureFile, outputDir + architectureFileName)
    print('Copied architecture file to ' + outputDir + architectureFileName)
    ### Plotting and saving the model
    plot_model(model, to_file = outputDir + 'loadedModel.png', show_shapes = True, show_layer_names = True)
    print('Saved ' + outputDir + 'loadedModel.png')
    ### Loading weights into the model
    weightsFileName = 'weights.h5'
    weightsFile = modelDir + weightsFileName
    #weightsFile = modelDir + preselectionCuts + '/weights.h5'
    print('Loading ' + weightsFile)
    model.load_weights(weightsFile)
    shutil.copyfile(weightsFile, outputDir + weightsFileName)
    print('Copied weights file to ' + outputDir + weightsFileName)
    batchSize = 2048
    #print(model.summary())
    return model, batchSize

### Loading NN if not performing training in buildPDNN
from keras.models import model_from_json
def LoadNN(outputDir):
    ### Loading architecture and weights from file
    print('Loading architecture and weights')
    architectureFileName = outputDir + '/architecture.json'
    with open(architectureFileName, 'r') as architectureFile:
        loadedModel = architectureFile.read()
    print('Loaded ' + architectureFileName)
    model = model_from_json(loadedModel)
    weightsFileName = outputDir + '/weights.h5'
    model.load_weights(weightsFileName)
    print('Loaded ' + weightsFileName)
    model.compile(loss = 'binary_crossentropy', weighted_metrics = ['accuracy']) #-> We don't care about the optimizer since we will only perform test, if loss and/or metric change save and then load them
    return model

def SignalBackgroundScores(wMC_test, wMC_train, yhat_test, yhat_train):
    wMC = np.concatenate((wMC_test, wMC_train))
    yhat = np.concatenate((yhat_test, yhat_train))
    Nbins = np.linspace(0, 1, 51)
    y, bins, _ = plt.hist(yhat, weights = wMC, bins = Nbins, histtype = 'step', lw = 2, color = 'blue', density = False) #, label = [r'Signal train + test']          
    y = y / np.sum(wMC) ### perché density = False                                                                                                                    
    print(y)

    ### Computing error on bin contents neglecting correlation between single bin content and whole histogram content                                                 
    yhat_sorted, wMC_sorted = sortColumns(yhat, wMC)
    weightsSum = 0
    weightsSquaredSum = 0
    iBin = 0
    y_error2 = np.array([])
    for score, weight in zip(yhat, wMC):
        if score >= bins[iBin] and score < bins[iBin + 1]:
            weightsSquaredSum += weight ** 2
        if iBin == (len(bins) - 2) and score == bins[iBin + 1]:
            weightsSquaredSum += weight ** 2
        else:
            y_error2 = np.append(y_error2, weightsSquaredSum)
            weightsSquaredSum = weight ** 2
            iBin += 1
            if iBin == len(bins) - 1:
                break

    ### fin qui ho y_signal_error che è un array con la somma dei quadrati dei pesi in ogni bin                                                                       
    histo_error = np.sqrt(np.sum(y_error2)) ### l'errore su tutto l'histo è la somma degli errori in ogni bin                                                         
    t1 = 1 / sum(wMC)
    t2 = y_error2
    t3 = (y / sum(wMC)) **2
    t4 = histo_error ** 2
    t5 = np.sqrt(t2 + t3 * t4)
    epsilon_bcr = t1 * t5

    return y, epsilon_bcr, bins

### Calibration curve (pag 17 in https://cds.cern.ch/record/2808844/files/ATL-COM-PHYS-2022-366.pdf)                                                                  
def CalibrationCurves(wMC_test_signal, wMC_train_signal, yhat_test_signal, yhat_train_signal, wMC_test_bkg, wMC_train_bkg, yhat_test_bkg, yhat_train_bkg, unscaledMass, outputDir, outputFileCommonName):
    y_signal, epsilon_bcr_signal, bins = SignalBackgroundScores(wMC_test_signal, wMC_train_signal, yhat_test_signal, yhat_train_signal)
    y_bkg, epsilon_bcr_bkg, _ = SignalBackgroundScores(wMC_test_bkg, wMC_train_bkg, yhat_test_bkg, yhat_train_bkg)
    sum_histo = y_signal + y_bkg
    print('sum:', sum_histo)
    ratio = y_signal / sum_histo
    print('ratio:', ratio)
    ratioErrorArray = np.array([])

    for sum_value, score_signal, score_bkg, score_signal_error, score_bkg_error in zip(sum_histo, y_signal, y_bkg, epsilon_bcr_signal, epsilon_bcr_bkg):
        t1 = 1 / (sum_value) ** 2
        t2 = (score_bkg * score_signal_error) ** 2
        t3 = (score_signal * score_bkg_error) ** 2
        t4 = np.sqrt(t2 + t3)
        t5 = t1 * t4
        ratioErrorArray = np.append(ratioErrorArray, t5)

    binsBar = np.array([])
    for iBinEdge in range(len(bins) - 1):
        binsBarEdge = (bins[iBinEdge] + bins[iBinEdge + 1]) / 2
        binsBar = np.append(binsBar, binsBarEdge)

    #plt.bar(binsBar, ratio, yerr = ratioErrorArray, width = 0.02, edgecolor = 'blue', color = 'white')                                                               
    plt.clf()
    plt.bar(binsBar, ratio, width = 0.02, edgecolor = 'blue', color = 'white')
    x = np.linspace(0, 1, 26)
    plt.plot(x, x, color = 'orange', linestyle='--')
    plt.ylabel('Probability ratio')
    plt.xlabel('Score')
    plt.figtext(0.15, 0.8, 'Mass hypothesis: ' + str(unscaledMass) + ' GeV', wrap = True, horizontalalignment = 'left')
    ScoresPltName = outputDir + '/CalibrationCurve_' + outputFileCommonName + '_' + str(unscaledMass) + '.png'
    plt.savefig(ScoresPltName)                                                                                                                                       
    #plt.show()
    print('Saved ' + ScoresPltName)
    plt.clf()
'''
def PredictionAndAUC(X, Y):
    y_pred = model.predict(X)
    fpr, tpr, thresholds = roc_curve(Y, y_pred)
    roc_auc = auc(fpr, tpr)
    return roc_auc
'''

### Features ranking
import eli5
from eli5.permutation_importance import get_score_importances
def FeaturesRanking(model, X_signal, X_bkg, deltasDict, inputFeatures, signal, analysis, channel, outputDir, outputFileCommonName, drawPlots):
    X = np.concatenate((X_signal, X_bkg))
    y = np.concatenate((np.ones(len(X_signal)), np.zeros(len(X_bkg))))
    nIter = 100
    base_score, score_decreases = get_score_importances(PredictionAndAUC, X, y, model, n_iter = nIter)
    print('###### base_score ########')
    print(base_score)
    print('###### score_decreases ########')
    print(score_decreases)
    feature_importances = np.mean(score_decreases, axis = 0)
    print('###### mean score_decreases #######')
    print(feature_importances)
    relative_feature_importances = feature_importances / base_score
    print('##### relative_feature_importances ######')
    deltasDict[unscaledMass] = relative_feature_importances
    print(deltasDict)
    '''
    if drawPlots:
        plt.clf()
        for iFeature in range(len(nputFeatures)):                                                                                                                    
            feature = inputFeatures[iFeature]
            print(feature)
            histoValues = np.array([item[iFeature] for item in score_decreases])
            print('##### histoValues #####')
            print(histoValues)
            histoValues = histoValues / base_score
            print('##### histoValues / base_score #####')
            print(histoValues)
            legendText = 'Mean: ' + str(round(histoValues.mean(), 2)) + '\nstd: ' + str(round(histoValues.std(), 2)) + '\nEntries: ' + str(len(histoValues))
            plt.hist(histoValues, label = legendText)
            plt.xlabel('Relative AUC difference')
            plt.title(feature + ' - ' + signal + ' (1 TeV) ' + analysis + ' ' + channel)
            plt.savefig(outputDir + '/Histo_AUCdifference_' + feature + '_' + outputFileCommonName + '.png')
            plt.clf()
    '''
    return deltasDict

def PlotFeaturesRanking(inputFeatures, deltasDict, outputDir, outputFileCommonName):
    fig, ax1 = plt.subplots(figsize = (13, 13))
    xLabels = inputFeatures
    yLabels = list(deltasDict.keys())
    deltasList = []
    for yLabel in yLabels:
        aa = list(deltasDict[yLabel])
        deltasList.append(aa)
    print(deltasList)
    im = ax1.matshow(deltasList)#, vmin = -1, vmax = 1)                                                                                                               
    plt.colorbar(im, ax = ax1)
    plt.xticks(range(len(inputFeatures)), inputFeatures, rotation = 'vertical')
    plt.yticks(range(len(yLabels)), yLabels)
    for i in range(len(xLabels)):
        for j in range(len(yLabels)):
            massValue = yLabels[j]
            c = deltasDict[massValue][i]
            cDisplay = round(c, 4)
            ax1.text(i, j, str(cDisplay), va = 'center', ha = 'center', fontsize = 8, color = 'r')
    plt.tight_layout()
    plt.savefig(outputDir + '/featuresRanking_' + outputFileCommonName + '.png')


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
import keras_tuner
from keras_tuner.tuners import RandomSearch

def buildOptimizedModel(hp):
    model = tf.keras.Sequential()
    #model.add(layers.Dense(units = hp.Int('units', min_value = 8, max_value = 200, step = 8), input_dim = 17, activation = 'relu'))
    model.add(layers.Dense(units = hp.Int('units', min_value = 40, max_value = 1000, step = 4), input_dim = 17, activation = 'relu'))
    #model.add(layers.Dense(units = hp.Int('units', min_value = 40, max_value = 1000, step = 4), input_dim = len(inputFeatures), activation = 'relu'))
    #model.add(tf.keras.layers.Dropout(hp.Float('dropout', 0, 0.3, step = 0.1)))
    model.add(Dropout(0.2))
    for iLayer in range(hp.Int('layers', 2, 6)):
        model.add(tf.keras.layers.Dense(units = hp.Int('units_' + str(iLayer), 40, 1000, step = 4), activation = 'relu'))
        #model.add(tf.keras.layers.Dropout(hp.Float('dropout_' + str(iLayer), 0, 0.3, step = 0.1)))
        model.add(Dropout(0.2))
    model.add(Dense(1, activation = 'sigmoid'))
    '''
    hp_lr = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4]),
        decay_steps = 10000,
        decay_rate = 0.95)
    optimizer = hp.Choice('optimizer', values = ['RMSprop', 'Adam'])
    if optimizer == 'RMSprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate = hp_lr)
    elif optimizer == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate = hp_lr)
    else:
        raise
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', weighted_metrics = ['accuracy'])#, run_eagerly = True)
    '''
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01), loss = 'binary_crossentropy', weighted_metrics = ['accuracy'])#, run_eagerly = True)
    plot_model(model, to_file='untrainedModel.png', show_shapes=True, show_layer_names=True)                                                                         
    return model

def HpOptimization(patienceValue, X, y, w, numberOfEpochs, validationFraction, batchSize, outputDir):
    tuner = RandomSearch(
        buildOptimizedModel,
        objective = keras_tuner.Objective('val_loss', direction = 'min'),
        max_trials = 4000,
        executions_per_trial = 2,
        directory = outputDir + '/tunerTrials/',
        overwrite = True
    )

    print(tuner.search_space_summary())
    stop_early = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = patienceValue)
    tuner.search(X, y, sample_weight = w, epochs = numberOfEpochs, validation_split = validationFraction, callbacks = [stop_early], batch_size = batchSize)
    tuner.results_summary()
    best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
    model = tuner.hypermodel.build(best_hps)
    #print(model.summary())

    #logFile.write('\n************************** HYPERPARAMETERS OPTIMIZATION RESULTS **************************')
    logString = '\n************************** HYPERPARAMETERS OPTIMIZATION RESULTS **************************'
    print('Number of nodes in layer number 0: ', tuner.get_best_hyperparameters()[0].get('units'))
    #logFile.write('\nNumber of nodes in layer number 0: ' + str(tuner.get_best_hyperparameters()[0].get('units')))
    logString += '\nNumber of nodes in layer number 0: ' + str(tuner.get_best_hyperparameters()[0].get('units'))
    #print('Dropout in layer number 0: ', tuner.get_best_hyperparameters()[0].get('dropout'))
    #logFile.write('\nDropout in layer number 0: ' + str(tuner.get_best_hyperparameters()[0].get('dropout')))
    #logString += '\nDropout in layer number 0: ' + str(tuner.get_best_hyperparameters()[0].get('dropout'))
    layersNumber = tuner.get_best_hyperparameters()[0].get('layers')
    print('Number of hidden layers: ', layersNumber)
    #logFile.write('\nNumber of hidden layers: ' + str(layersNumber))
    logString += '\nNumber of hidden layers: ' + str(layersNumber)
    for iLayer in range(1, layersNumber + 1):
        hp_nodes = 'units_' + str(iLayer - 1)
        #hp_dropout = 'dropout_' + str(iLayer - 1)
        print('Number of nodes in hidden layer number ' + str(iLayer) + ': ' + str(tuner.get_best_hyperparameters()[0].get(hp_nodes)))
        #logFile.write('\nNumber of nodes in hidden layer number ' + str(iLayer) + ': ' + str(tuner.get_best_hyperparameters()[0].get(hp_nodes)))
        logString += '\nNumber of nodes in hidden layer number ' + str(iLayer) + ': ' + str(tuner.get_best_hyperparameters()[0].get(hp_nodes))
        #print('Dropout in hidden layer number ' + str(iLayer) + ': ' + str(tuner.get_best_hyperparameters()[0].get(hp_dropout)))
        #logFile.write('\nDropout in hidden layer number ' + str(iLayer) + ': ' + str(tuner.get_best_hyperparameters()[0].get(hp_dropout)))
        #logString += '\nDropout in hidden layer number ' + str(iLayer) + ': ' + str(tuner.get_best_hyperparameters()[0].get(hp_dropout))
    #print('Optimizer:', model.optimizer.get_config())
    #logFile.write('\nOptimizer: ' + str(model.optimizer.get_config()))
    #logString += '\nOptimizer: ' + str(model.optimizer.get_config())
    #logFile.write('\n*******************************************************************************************')
    logString += '\n*******************************************************************************************'
    return model, logString

def SameStatAsVBF(dataTrain):
    dataTrain_signal = dataTrain[dataTrain['isSignal'] == 1]
    dataTrain_bkg = dataTrain[dataTrain['isSignal'] != 1]
    dataTrain_signal = dataTrain_signal[:87156]
    dataTrain_bkg = dataTrain_bkg[:35500]
    dataTrain = pd.concat((dataTrain_signal, dataTrain_bkg), ignore_index = True)
    dataTrain = ShufflingData(dataTrain)
    XTrain = np.array(dataTrain[InputFeatures].values).astype(np.float32)
    yTrain = np.array(dataTrain['isSignal'].values).astype(np.float32)
    return dataTrain, XTrain, yTrain

def plotHistory(patiences, series, feature, outputDir, outputFileCommonName):
    plt.clf()
    for i in range(len(patiences)):
        plt.subplot(220 + (i+1))
        plt.plot(series[i])
        #plt.title('patience = ' + str(patiences[i]), pad = -80)
    #plt.show()
    pltName = outputDir + '/' + feature + '_DifferentPatience_' + outputFileCommonName + '.png'
    plt.savefig(pltName)
    print('Saved ' + pltName)
    plt.clf()

from keras import backend
from keras.callbacks import ReduceLROnPlateau, Callback, ModelCheckpoint
def TrainNN(X_train, y_train, w_train, numberOfEpochs, batchSize, validationFraction, model, studyLearningRate):#, iLoop, loop):
    #print('Training the ' + NN + ' -- loop ' + str(iLoop) + ' out of ' + str(loop - 1))

    ### If we want to stop the training when we don't have an improvement in $monitor after $patience epochs
    #earlyStoppingCB = EarlyStopping(verbose = True, patience = patienceValue, monitor = 'val_loss', restore_best_weights = True)
    patienceEarlyStopping = 15#5#15
    monitorEarlyStopping = 'val_loss'
    earlyStoppingCB = EarlyStopping(verbose = True, patience = patienceEarlyStopping, monitor = monitorEarlyStopping, restore_best_weights = True)

    ### If we want to reduce the LR when we don't have an improvement in $monitor after $patience epochs
    decreaseLR = tf.keras.callbacks.LearningRateScheduler(scheduler) ### needed?
    #rlrp = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = patienceValue, min_lr = 0.0001)
    patienceLR = 5
    deltaLR = 0.1
    minLR = 0.00001
    rlrp = ReduceLROnPlateau(monitor = monitorEarlyStopping, factor = deltaLR, patience = patienceLR, min_lr = minLR)

    checkpoint_filepath = 'tmp/checkpoint/model.hdf5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    verbose = True,
    save_best_only=True)

    # If we want to monitor the learning rate in each epoch
    class LearningRateMonitor(Callback):
        # start of training
        def on_train_begin(self, logs={}):
            self.lrates = list()

        # end of each training epoch
        def on_epoch_end(self, epoch, logs={}):
            # get and store the learning rate
            optimizer = self.model.optimizer
            lrate = float(backend.get_value(self.model.optimizer.lr))
            print('Learning rate:', lrate)
            self.lrates.append(lrate)

    if studyLearningRate:
        #lrm = LearningRateMonitor(Callback)
        lrm = LearningRateMonitor()
        CallbacksList = [rlrp, lrm, earlyStoppingCB] #earlyStoppingCB

    else:
        #lrm = LearningRateMonitor(Callback)
        lrm = LearningRateMonitor()
        CallbacksList = [earlyStoppingCB, lrm, rlrp]#[rlrp, model_checkpoint_callback] #earlyStoppingCB
        #CallbacksList = [earlyStoppingCB]
    #model, Loss, Metrics, learningRate, Optimizer = BuildDNN(len(inputfeatures), numberOfNodes, numberOfLayers, dropout)
    #model.compile(loss = Loss, optimizer = Optimizer, weighted_metrics = Metrics)
    modelMetricsHistory = model.fit(X_train, y_train, sample_weight = w_train, epochs = numberOfEpochs, batch_size = batchSize, validation_split = validationFraction, verbose = 1, callbacks = CallbacksList)

    if studyLearningRate:
        return modelMetricsHistory, lrm.lrates
    else:
        return modelMetricsHistory, CallbacksList, patienceEarlyStopping, monitorEarlyStopping, patienceLR, deltaLR, minLR


def WeightedDistributionComparison(dataTrainSignal, dataTrainBkg, dataTestSignal, dataTestBkg, scoresTrainSignal, scoresTrainBkg, scoresTestSignal, scoresTestBkg, featureToPlot):
    dataSignal = pd.concat([dataTrainSignal, dataTestSignal], ignore_index = True)
    dataBkg = pd.concat([dataTrainBkg, dataTestBkg], ignore_index = True)
    scoresSignal = np.concatenate((scoresTrainSignal, scoresTestSignal))
    scoresBkg = np.concatenate((scoresTrainBkg, scoresTestBkg))
    weightsBkg = dataBkg['weight']
    minHisto = min(min(dataBkg[featureToPlot]), min(dataSignal[featureToPlot]))
    maxHisto = max(max(dataBkg[featureToPlot]), max(dataSignal[featureToPlot]))
    Bins = np.linspace(minHisto, maxHisto, 101)
    MCweightsSignal = np.array(dataSignal['weight'])
    bkgContent, bkgEdges, _ = plt.hist(dataBkg[featureToPlot], bins = Bins, weights = weightsBkg, histtype = 'step', lw = 2, color = 'blue', density = True, label = ['Background'])
    signalContent, signalEdges, _ = plt.hist(dataSignal[featureToPlot], bins = Bins, weights = MCweightsSignal, histtype = 'step', lw = 2, color = 'red', density = True, label = ['Signal'])
    plt.legend()
    #plt.show()
    plt.clf()
    leftBin = 0
    rightBin = 0
    iBin = 0
    for bkg, signal in zip(bkgContent, signalContent):
        if bkg != 0 and signal != 0 and leftBin == 0:
            leftBin = bkgEdges[iBin]
        if bkg != 0 and signal != 0:
            rightBin = bkgEdges[iBin + 1]
        iBin += 1
    print(leftBin)
    print(rightBin)
    reducedBins = np.linspace(leftBin, rightBin, 51)
    secondWeightsSignal = np.array([])
    secondWeightsSignalList = np.asarray((1 - scoresSignal) / scoresSignal).astype(np.float32)
    for secondWeight in secondWeightsSignalList:
        secondWeightsSignal = np.append(secondWeightsSignal, secondWeight[0])
    weightsSignal = np.array(MCweightsSignal * secondWeightsSignal)
    print(MCweightsSignal)
    print('Len(MCweightsSignal):', len(MCweightsSignal))
    print(round(MCweightsSignal[0], 10))
    print(secondWeightsSignal)
    print('len(secondWeightsSignal):', len(secondWeightsSignal))
    print(round(secondWeightsSignal[0], 10))
    print(weightsSignal)
    print('len(weightsSignal):', len(weightsSignal))
    print(round(weightsSignal[0], 10))
    plt.hist(dataBkg[featureToPlot], bins = reducedBins, weights = weightsBkg, histtype = 'step', lw = 2, color = 'blue', density = True, label = ['Background'])
    plt.hist(dataSignal[featureToPlot], bins = reducedBins, weights = MCweightsSignal, histtype = 'step', lw = 2, color = 'red', density = True, label = ['Signal'])
    plt.hist(dataSignal[featureToPlot], bins = reducedBins, weights = weightsSignal, histtype = 'step', lw = 2, color = 'green', density = True, label = ['Signal with Pb/Ps'])
    plt.xlabel(featureToPlot)
    plt.ylabel('Weighted counts')
    plt.legend()
    plt.show()

def computePx(ptArray, phi):
    px = ptArray * np.cos(phi)
    return px

def computePy(ptArray, phi):
    px = ptArray * np.sin(phi)
    return px

def computePz(ptArray, eta):
    pz = ptArray * np.sinh(eta)
    return pz

def computeE(ptArray, etaArray, massArray):
    e = np.sqrt((ptArray ** 2) * (1 + np.sinh(etaArray) ** 2) + (massArray ** 2))
    return e

def computeDeltaPhi(phi1, phi2):
    diff1 = np.abs(phi1 - phi2)
    diff2 = 2 * math.pi - np.abs(phi1 - phi2)
    #delta_phi = np.min((np.abs(phi1 - phi2), 2 * math.pi - np.abs(phi1 - phi2)))
    delta_phi = np.minimum(diff1, diff2)
    return delta_phi
'''
def computeDeltaPhi(phi1, phi2):
    delta_phi = np.min((np.abs(phi1 - phi2), 2 * math.pi - np.abs(phi1 - phi2)))
    return delta_phi
'''
def computeDerivedVariables(variablesToDerive, dataFrame, signal, analysis):
    for variableToDerive in variablesToDerive:
        print(variableToDerive)
        objectName = variableToDerive.split('_')[0]
        variable = variableToDerive.split('_')[1]
        if variable == 'e':
            newColumn = computeE(dataFrame[objectName + '_pt'], dataFrame[objectName + '_eta'], dataFrame[objectName + '_m'])
        if variable == 'px':
            newColumn = computePx(dataFrame[objectName + '_pt'], dataFrame[objectName + '_phi'])
        if variable == 'py':
            newColumn = computePy(dataFrame[objectName + '_pt'], dataFrame[objectName + '_phi'])
        if variable == 'pz':
            newColumn = computePy(dataFrame[objectName + '_pt'], dataFrame[objectName + '_eta'])
        if variableToDerive == 'delta_phi_lep12':
            newColumn = computeDeltaPhi(dataFrame['lep1_phi'], dataFrame['lep2_phi'])
        if variableToDerive == 'delta_phi_jetlep':
            if analysis == 'merged':
                newColumn = computeDeltaPhi(dataFrame['fatjet_phi'], dataFrame['lep2_phi'])
            elif analysis == 'resolved' and  'HVT' in signal:
                newColumn = computeDeltaPhi(dataFrame['sigVJ1_phi'], dataFrame['lep2_phi'])
            else:
                newColumn = computeDeltaPhi(dataFrame['sigVJ1_phi'], dataFrame['lep2_phi'])
        if variableToDerive == 'delta_phi_jet12':
            if 'HVT' in signal:
                newColumn = computeDeltaPhi(dataFrame['sigVJ1_phi'], dataFrame['sigVJ2_phi'])
            else:
                newColumn = computeDeltaPhi(dataFrame['sigVJ1_phi'], dataFrame['sigVJ2_phi'])
        if variableToDerive == 'delta_phi_lepjet':
            if analysis == 'merged':
                newColumn = computeDeltaPhi(dataFrame['lep1_phi'], dataFrame['fatjet_phi'])            
            elif 'HVT' in signal:
                newColumn = computeDeltaPhi(dataFrame['lep1_phi'], dataFrame['sigVJ2_phi'])            
            else:
                newColumn = computeDeltaPhi(dataFrame['lep1_phi'], dataFrame['sigVJ2_phi'])            
        #dataFrame = dataFrame.assign({variableToDerive: newColumn})
        dataFrame[variableToDerive] = newColumn
    return dataFrame


### Training the newtork with different learning rate and saving plots of learning rate, accuracy and loss vs epochs
def studyLRpatience(X_train, y_train, w_train, numberOfEpochs, batchSize, validationFraction, model, outputDir, outputFileCommonName):
    patiences = [2, 5, 10, 15]                                                                                                                                    
    print('Training ' + str(len(patiences)) + ' pDNN with decreasing learning rate and different patience values')
    lr_list, loss_list, acc_list, = list(), list(), list()
    for iPatience in range(len(patiences)):
        iPatienceValue = patiences[iPatience]
        print('NN number ' + str(iPatience) + ' out of ' + str(len(patiences) - 1) + ' -> patience = ' + str(iPatienceValue))
        modelMetricsHistory, lrm_rates = TrainNN(X_train, y_train, w_train, iPatienceValue, numberOfEpochs, batchSize, validationFraction, model, True)#, iLoop, loop)
        #modelMetricsHistory, lrm_rates = TrainNN(X_train, y_train, w_train, iPatienceValue, numberOfEpochs, batchSize, validationFraction, NN, model, Loss, Optimizer, Metrics, iLoop, loop)
        loss_list.append(modelMetricsHistory.history['loss'])
        acc_list.append(modelMetricsHistory.history['accuracy'])
        lr_list.append(lrm_rates)

    # plotting learning rates vs epochs
    plotHistory(patiences, lr_list, 'LearningRateLRstudies', outputDir, outputFileCommonName)
    # plotting loss vs epochs
    plotHistory(patiences, loss_list, 'LossLRstudies', outputDir, outputFileCommonName)
    # plotting accuracy vs epochs
    plotHistory(patiences, acc_list, 'AccuracyLRstudies', outputDir, outputFileCommonName)


granularity=1e3
#b_inf = 0.005*np.where(y==0)[0].shape[0]*w[np.where(y==0)[0]].mean()

def derive_distr(bkg_yeld,bkg_weights,sig_yeld,sig_weights,a,b,x_opt):
    hist_bins_range=np.array([a,x_opt,b])

    yeld_bkg,_ = np.histogram(bkg_yeld, bins=hist_bins_range, weights=bkg_weights)
    yeld_sig,_ = np.histogram(sig_yeld, bins=hist_bins_range, weights=sig_weights)
    
    return yeld_bkg,yeld_sig

def significance(n,b):
    print('n:', n)
    print('b:', b)
    print(np.sqrt( (2*(n*np.log(n/b)+b-n)).sum() ))
    return np.sqrt( (2*(n*np.log(n/b)+b-n)).sum() )

def rescaling_distr(yeld_bkg,yeld_sig,ϵ):
    #return int_v(yeld_bkg+int_v(ϵ*yeld_sig)),int_v(yeld_bkg)
    return yeld_bkg+ϵ*yeld_sig, yeld_bkg

def compute_Z0(bkg_yeld,bkg_weights,sig_yeld,sig_weights,x_min,x_max,x,ϵ):
    #b_inf=10
    if x==-1:
        yeld_bkg,_=np.histogram(bkg_yeld,bins=1,weights=bkg_weights)
        yeld_sig,_=np.histogram(sig_yeld,bins=1,weights=sig_weights)
    else:
        yeld_bkg,yeld_sig=derive_distr(bkg_yeld,bkg_weights,sig_yeld,sig_weights,x_min,x_max,x)
        
#    print(x,yeld_bkg,yeld_sig)
    n,b=rescaling_distr(yeld_bkg,yeld_sig,ϵ)
#    print('printing n,b from compute_Z0:', n,b)

    #if b.min()>b_inf:
    #    return significance(n,b), b.min()
    #else:
    #    return 0,b.min()
    return significance(n,b), b.min()

def compute_sigz0(bkg_yeld,bkg_weights,sig_yeld,sig_weights,ϵ,flag):
    b_inf = 0.005 * sum(bkg_weights)
    a=min(bkg_yeld.min(),sig_yeld.min())
    b=max(sig_yeld.max(),bkg_yeld.max())
    
    ampl=int(b*granularity)-int(a*granularity)+1
    x_interval=np.arange(int(a*granularity+1),int(b*granularity),ampl*(1./granularity))*(1./granularity)
    
    Z0=np.array([])
    bmin=np.array([])
    if flag=='boosted_m':
        x0=a #x_interval[0]
        x1=b #x_interval[-1]
    else:
        x0=0. #a
        x1=1. #b
    #print(x0,x1)
    for x in x_interval:
        z,b=compute_Z0(bkg_yeld,bkg_weights,sig_yeld,sig_weights,x0,x1,x,ϵ)
        Z0=np.append(Z0,z)
        bmin=np.append(bmin,b)

    try:
        Z0_max=Z0[np.where(bmin>b_inf)[0]].max()
        x0_max=x_interval[Z0==Z0_max].max()
        print('max z0:',Z0_max, 'bmin.max:', bmin.max())
    except ValueError:  #raised if `y` is empty.
        Z0_max=0. #Z0.max()
        x0_max=-1
        print('Exception', 'bmin.max:', bmin.max())
        pass        

    return Z0_max,x0_max,Z0,bmin,x_interval

def derive_Z_test_boostedmass(X_boosted_bkg, bkg_weights, X_boosted_sig, sig_weights):
#def derive_Z_test_boostedmass(X_test,y_test,test_weights,show_plots):
    '''
    Z0_boosted_max_list=list()
    masses_list=np.unique(signal_masses)
    
    Z0_pnn_max_list=list()
    n_sig_m=list()
    n_bkg_m=list()
    for test_mass in masses_list:
        
        #rescaled mass:
        res_mass_test=compute_rescaled_mass(test_mass)
        print('mass:', test_mass)#, res_mass_test)
        X_testmass=X_test

        #write bkg mass to test mass (signal mass left invariant)
        X_testmass[y_test==0,-1]=res_mass_test
        
        #select only masses equal to res_mass_test:
        sig_selection=np.intersect1d(np.where(y_test==1)[0], np.where(X_testmass[:,-1]==res_mass_test)[0])
        bkg_selection=np.where(y_test==0)[0]
        sig_weights=test_weights[sig_selection]
        bkg_weights=test_weights[bkg_selection]

        #print('weights (sig, bkg) mean:',sig_weights.mean(),bkg_weights.mean())
        
        print(f'total of weighted events - signal: {sig_selection.shape[0]*sig_weights.mean():.4}, bkg: {bkg_selection.shape[0]*bkg_weights.mean():.4}')
        n_sig_m.append(sig_selection.shape[0]*sig_weights.mean())
        n_bkg_m.append(bkg_selection.shape[0]*bkg_weights.mean())

        X_boosted_sig=boosted_m[sig_selection]
        X_boosted_bkg=boosted_m[bkg_selection]
    '''
    
    Z0_boosted_max,x0_boosted_m_max,Z0_boosted_m,bmin_m,x_int_m=compute_sigz0(X_boosted_bkg,
                                                                              bkg_weights,
                                                                              X_boosted_sig,
                                                                              sig_weights,
                                                                              1.,
                                                                              'boosted_m'
    )
    '''        
    Z0_boosted_max_list.append(Z0_boosted_max)
        
    if show_plots==True:
        plt.hist(X_boosted_sig,weights=sig_weights,label='sig',bins=100,histtype='step')
        plt.hist(X_boosted_bkg,weights=bkg_weights,label='bkg',bins=100,histtype='step')
        plt.yscale('log')
        plt.legend()
        plt.show()
            
        plt.plot(x_int_m,Z0_boosted_m,label='boosted_m',color='blue')
        plt.axvline(x=x0_boosted_m_max)
        plt.legend()
        plt.show()
        
        print(bmin_m[x_int_m==x0_boosted_m_max])
        plt.plot(x_int_m,bmin_m,label='bmin',color='blue')
        plt.axvline(x=x0_boosted_m_max)
        plt.axhline(y=b_inf)
        plt.legend()
        plt.show()
        
    if show_plots==True:
        plt.plot(masses_list,n_sig_m,label='signal',color='orange')
        plt.plot(masses_list,n_bkg_m,label='bkg',color='blue')
        plt.legend()
        plt.show()
        
        
    return masses_list, Z0_boosted_max_list
    '''
    return Z0_boosted_max

def derive_Z_test_mass(PNN_bkg_pred, bkg_weights, PNN_sig_pred, sig_weights):
#def derive_Z_test_mass(NN,X_test,y_test,test_weights,show_plots):
    '''
    #test on given mass
    masses_list=np.unique(signal_masses)
    #print(masses_list)
    
    Z0_boosted_max_list=list()
    Z0_pnn_max_list=list()
    n_sig_m=list()
    n_bkg_m=list()
    for test_mass in masses_list:
        
        #rescaled mass:
        res_mass_test=compute_rescaled_mass(test_mass)
        print('mass:', test_mass)#, res_mass_test)
        X_testmass=X_test

        #write bkg mass to test mass (signal mass left invariant)
        X_testmass[y_test==0,-1]=res_mass_test
        
        #select only masses equal to res_mass_test:
        sig_selection=np.intersect1d(np.where(y_test==1)[0], np.where(X_testmass[:,-1]==res_mass_test)[0])
        bkg_selection=np.where(y_test==0)[0]
        sig_weights=test_weights[sig_selection]
        bkg_weights=test_weights[bkg_selection]

        #print('weights (sig, bkg) mean:',sig_weights.mean(),bkg_weights.mean())
        
        print(f'total of weighted events - signal: {sig_selection.shape[0]*sig_weights.mean():.4}, bkg: {bkg_selection.shape[0]*bkg_weights.mean():.4}')
        n_sig_m.append(sig_selection.shape[0]*sig_weights.mean())
        n_bkg_m.append(bkg_selection.shape[0]*bkg_weights.mean())            
        
        PNN_sig_pred=NN.predict(X_testmass[sig_selection,:])#[:,1]
        PNN_bkg_pred=NN.predict(X_testmass[bkg_selection,:])#[:,1]#.flatten()
        
        n_outs = PNN_sig_pred.shape[1]
        #print(n_outs)
        if n_outs==2:
            PNN_sig_pred=PNN_sig_pred[:,1]
            PNN_bkg_pred=PNN_bkg_pred[:,1]
        if n_outs==1:
            PNN_sig_pred=PNN_sig_pred.flatten()
            PNN_bkg_pred=PNN_bkg_pred.flatten()

    '''
    Z0_pnn_max,x0_pnn_m_max,Z0_pnn_m,pnn_bmin_m,pnn_x_int_m=compute_sigz0(
        PNN_bkg_pred,
        bkg_weights,
        PNN_sig_pred,
        sig_weights,
        1.,
        'pnn'
    )
    '''
        Z0_pnn_max_list.append(Z0_pnn_max)

        if show_plots==True:
            plt.hist(PNN_sig_pred,weights=sig_weights,label='sig',bins=100,histtype='step')
            plt.hist(PNN_bkg_pred,weights=bkg_weights,label='bkg',bins=100,histtype='step')
            plt.yscale('log')
            plt.legend()
            plt.show()

            plt.plot(pnn_x_int_m,Z0_pnn_m,label='pnn',color='orange')
            plt.axvline(x=x0_pnn_m_max)
            plt.legend()
            plt.show()
            
            print(pnn_bmin_m[pnn_x_int_m==x0_pnn_m_max])
            plt.plot(pnn_x_int_m,pnn_bmin_m,label='bmin',color='blue')
            plt.axvline(x=x0_pnn_m_max)
            plt.axhline(y=b_inf)
            plt.legend()
            plt.show()
        
        
    return Z0_pnn_max_list
    '''
    return Z0_pnn_max
