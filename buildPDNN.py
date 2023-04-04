from Functions import *
from keras.utils.vis_utils import plot_model
import eli5
from eli5.permutation_importance import get_score_importances
from keras import backend

### Setting a seed for reproducibility
#tf.random.set_seed(1234)

#blockPrint()

NN = 'PDNN'
batchSize = 2048
patienceValue = 5

### Reading the command line
tag, analysis, channel, preselectionCuts, background, trainingFraction, signal, numberOfNodes, numberOfLayers, numberOfEpochs, validationFraction, dropout, testMass, doTrain, doTest, loop, doHpOptimization, drawPlots, trainSet, doStudyLRpatience, configFile = ReadArgParser()
originsBkgTest = list(background.split('_'))

drawPlots = True
doFeaturesRanking = False
doSameStatAsVBF = False
doStatisticTest = False

if channel == 'ggF':
    signalLabel = signal
elif channel == 'VBF':
    signalLabel = signal.replace('VBF', '')
if 'WZ' in signalLabel:
    signalLabel = signal.replace('WZ', '')

### Reading the configuration file
ntuplePath, dfPath, InputFeatures = ReadConfig(tag, analysis, signal, configFile)
#inputDir = dfPath + analysis + '/' + channel + '/' + preselectionCuts + '/ggFVBF' + '/' + signal + '/' + background + '/'# + 'tmp/' # + '_fullStat/'
#inputDir = dfPath + analysis + '/' + channel + '/' + preselectionCuts + '/' + signal + '/' + background + '/'# + 'tmp/' # + '_fullStat/'
inputDir = dfPath + analysis + '/' + channel + '/' + preselectionCuts + '/' + signal + '/' + background + '/'# + 'tmp/' # + '_fullStat/' <<<<<----- questo
print(Fore.GREEN + 'Input files directory: ' + inputDir)
#inputDir = dfPath + analysis + '/' + channel + '/' + preselectionCuts + '/' + signal + '/' + background + '/' + 'ggFsameStatAsVBF/'# + 'tmp/' # + '_fullStat/'
outputFileCommonName = NN + '_' + analysis + '_' + channel + '_' + preselectionCuts + '_' + signal + '_' + background

### Creating the output directory and the logFile
#outputDir = inputDir + NN + '_trainSet' + str(trainSet)#0'# + '_3'# + '/withDNNscore'# + '/test1'# + '/3layers'#'/ggFsameStatAsVBF'# + '/withDNNscore' #'/DNNScore_Z'# + '/' + preselectionCuts # + '_halfStat'
#outputDir = inputDir + NN + '_2layers48nodesSwish_mptetaphi/withoutLowWeights'
if 'mptetaphi' in configFile:
    outtt = 'mptetaphi'
else:
    outtt = 'deltaphi'
outputDir = inputDir + NN + '_2layers48nodesSwish_' + outtt + '/10feb2023'
#outputDir = inputDir + NN + '_3layers48nodesRelu_deltaphi'
#outputDir = inputDir + NN + 'hpOptimization'
print(format('Output directory: ' + Fore.GREEN + outputDir), checkCreateDir(outputDir))
print(Fore.GREEN + 'Input files directory: ' + inputDir)

logFileName = outputDir + '/logFile_' + outputFileCommonName + '.txt'
logFile = open(logFileName, 'w')
logInfo = ''
logString = WriteLogFile(tag, ntuplePath, InputFeatures, inputDir, doHpOptimization, doTrain, doTest, validationFraction, batchSize)#, patienceValue)
logFile.write(logString)
logInfo += logString

### If doStatisticTest, open files where to score AUC, loss and scores values in each iteration
if doStatisticTest:
    AUCfile = open(outputDir + '/AUCvalues.txt', 'w')
    AUCfile.write('AUC values in each iteration\n')
    lossFile = open(outputDir + '/lossValues.txt', 'w')
    lossFile.write('Train loss minimum - validation loss minimum - train loss at the validation loss minimum (in each iteration)\n')
    scoresFileName = outputDir + '/scoresValues.txt'
    scoresFile = open(scoresFileName, 'w')
    scoresFile.write('Scores of the 20 fixed events in each iteration\n')
    
### Loading input data
data_train, data_test = LoadData(inputDir, tag, signal, analysis, channel, background, trainingFraction, preselectionCuts, InputFeatures, trainSet)

### Computing dataframe statistics
rawTrain = data_train.shape[0]
rawTrainSignal = data_train[data_train['isSignal'] == 1].shape[0]
rawTrainBkg = data_train[data_train['isSignal'] == 0].shape[0]
MCtrain = round(sum(data_train['weight']), 1)
MCtrainSignal = round(sum(data_train[data_train['isSignal'] == 1]['weight']), 1)
MCtrainBkg = round(sum(data_train[data_train['isSignal'] == 0]['weight']), 1)
rawTest = data_test.shape[0]
rawTestSignal = data_test[data_test['isSignal'] == 1].shape[0]
rawTestBkg = data_test[data_test['isSignal'] == 0].shape[0]
MCtest = round(sum(data_test['weight']), 1)
MCtestSignal = round(sum(data_test[data_test['isSignal'] == 1]['weight']), 1)
MCtestBkg = round(sum(data_test[data_test['isSignal'] == 0]['weight']), 1)
print(Fore.BLUE + 'Number of train events: ' + str(rawTrain) + ' (' + str(rawTrainSignal) + ' signal and ' + str(rawTrainBkg) + ' background), with MC weights: ' + str(MCtrain) + ' (' + str(MCtrainSignal) + ' signal and ' + str(MCtrainBkg) + ' background)\nNumber of test events: ' + str(rawTest) + ' (' + str(rawTestSignal) + ' signal and ' + str(rawTestBkg) + ' background), with MC weights: ' + str(MCtest) + ' (' + str(MCtestSignal) + ' signal and ' + str(MCtestBkg) + ' background)')
logString = '\nNumber of train events: ' + str(rawTrain) + ' (' + str(rawTrainSignal) + ' signal and ' + str(rawTrainBkg) + ' background), with MC weights: ' + str(MCtrain) + ' (' + str(MCtrainSignal) + ' signal and ' + str(MCtrainBkg) + ' background)\nNumber of test events: ' + str(rawTest) + ' (' + str(rawTestSignal) + ' signal and ' + str(rawTestBkg) + ' background), with MC weights: ' + str(MCtest) + ' (' + str(MCtestSignal) + ' signal and ' + str(MCtestBkg) + ' background)'
logFile.write(logString)
logInfo += logString

### Scaling input features
data_train = scaleTrainTestDataset(data_train, inputDir, InputFeatures, 'train')    
data_test = scaleTrainTestDataset(data_test, inputDir, InputFeatures, 'test')
outputFileName = outputDir + '/variables.json'
shutil.copyfile(inputDir + 'variables.json', outputFileName)
print(Fore.GREEN + 'Copied variables file to ' + outputFileName)

### Saving input features, truth and train weights vectors
X_train, y_train, w_train = extractFeatures(data_train, InputFeatures)
X_test, y_test, w_test = extractFeatures(data_test, InputFeatures)

'''
### to make same stat as VBF
if doSameStatAsVBF:
    data_train, X_train, y_train = SameStatAsVBF(data_train)
'''
if doHpOptimization:
    model, logString = HpOptimization(patienceValue, X_train, y_train, w_train, numberOfEpochs, validationFraction, batchSize, outputDir)
    logFile.write(logString)
    logInfo += logString

else: 
    ### Building and compiling the PDNN
    model, Loss, Metrics, learningRate, Optimizer, activationFunction = BuildNN(len(InputFeatures), numberOfNodes, numberOfLayers, dropout, doStudyLRpatience)
    model.compile(loss = Loss, optimizer = Optimizer, weighted_metrics = Metrics)
    NNdiagramName = outputDir + '/trainedModel.png'
    plot_model(model, to_file = NNdiagramName, show_shapes = True, show_layer_names = True)
    print(Fore.GREEN + 'Saved ' + NNdiagramName)
    logString = '\nNumber of nodes: ' + str(numberOfNodes) + '\nNumber of hidden layers: ' + str(numberOfLayers) + '\nDropout: ' + str(dropout) + '\nLoss: ' + Loss + '\nOptimizer: ' + str(Optimizer) + '\nInitial learning rate: ' + str(learningRate) + '\nMetrics: ' + str(Metrics) + '\nActivation function in hidden layers: ' + activationFunction
    logFile.write(logString)
    logInfo += logString
                
if doStudyLRpatience:
    studyLRpatience(X_train, y_train, w_train, numberOfEpochs, batchSize, validationFraction, model, outputDir, outputFileCommonName)

for iLoop in range(loop):

    if iLoop != 0:
        outputDirLoop = outputDir + '/loop' + str(iLoop)
        outputFileCommonName += '_loop' + str(iLoop)
    else: 
        outputDirLoop = outputDir
    print(format('Output directory: ' + Fore.GREEN + outputDirLoop), checkCreateDir(outputDirLoop))

    if not doTrain:
        model = LoadNN('/nfs/kloe/einstein4/HDBS_new/NNoutput/r33-24/merged/ggF/none/Radion/all/PDNN_2layers48nodesSwish_deltaphi/')#outputDirLoop)

    if doTrain:

        ### Trainig the pDNN if not doStudyLRpatience, otherwise first choose the patience and then run this script withouth doStudyLRpatience
        if not doStudyLRpatience:
            print(Fore.BLUE + 'Training the ' + NN + ' -- loop ' + str(iLoop) + ' out of ' + str(loop - 1))
            modelMetricsHistory, callbacksList, patienceEarlyStopping, monitorEarlyStopping, patienceLR, deltaLR, minLR = TrainNN(X_train, y_train, w_train, numberOfEpochs, batchSize, validationFraction, model, doStudyLRpatience)#, iLoop, loop)
            logString = '\nCallbacks list: ' + str(callbacksList) + '\nPatience early stopping: ' + str(patienceEarlyStopping) + '\nMonitor early stopping: ' + monitorEarlyStopping + '\nLearning rate decrease at each step: ' + str(deltaLR) + '\nPatience learning rate: ' + str(patienceLR) + '\nMinimum learning rate: ' + str(minLR)
            logFile.write(logString)
            logInfo += logString

        if doStatisticTest:
            ### Writing best losses to file
            loss_hist = modelMetricsHistory.history['loss']
            val_loss_hist = modelMetricsHistory.history['val_loss']
            best_epoch = np.argmin(loss_hist)# + 1
            lossFile.write(str(np.min(loss_hist)) + ' ' + str(np.min(val_loss_hist)) + ' ' + str(loss_hist[best_epoch]) + '\n')

        ### Saving to files
        #model.load_weights('tmp/checkpoint/model.hdf5')
        SaveModel(model, outputDirLoop, NN)
        plot_model(model, to_file = 'trainedModel.png', show_shapes = True, show_layer_names = True)

    if doTest:
        ### Evaluating the performance of the PDNN on the test sample and writing results to the log file
        print(Fore.BLUE + 'Evaluating the performance of the ' + NN)
        testLoss, testAccuracy = EvaluatePerformance(model, X_test, y_test, w_test, batchSize) ### using train_weight

        if iLoop == 0:
            logString = '\nTest loss: ' + str(testLoss) + '\nTest accuracy: ' + str(testAccuracy)
            logFile.write(logString)
            logInfo += logString
            
    else:
        testLoss = testAccuracy = None

    ### Drawing accuracy and loss
    if drawPlots and doTrain: ### THINK
        DrawLoss(modelMetricsHistory, testLoss, patienceEarlyStopping, outputDirLoop, NN, analysis, channel, preselectionCuts, signal, background, outputFileCommonName)
        DrawAccuracy(modelMetricsHistory, testAccuracy, patienceEarlyStopping, outputDirLoop, NN, analysis, channel, preselectionCuts, signal, background, outputFileCommonName)
    '''
    if iLoop == 0:
        logFile.close()
        print(Fore.GREEN + 'Saved ' + logFileName)
    '''
    
    logFile.close()
    if loop != 0:
        logFileLoop = outputDirLoop + '/logFile_' + outputFileCommonName + '.txt'
        shutil.move(logFileName, logFileLoop)
        print(Fore.GREEN + 'Saved ' + logFileLoop)
    else:
        print(Fore.GREEN + 'Saved ' + logFileName)
    
    if doTest == False:
        exit()

    if doFeaturesRanking:
        deltasDict = {}

    def PredictionAndAUC(X, Y):
        y_pred = model.predict(X)
        fpr, tpr, thresholds = roc_curve(Y, y_pred)
        roc_auc = auc(fpr, tpr)
        return roc_auc
        
    ### Dividing signal from background
    data_test_signal = data_test[data_test['isSignal'] == 1]
    data_test_bkg = data_test[data_test['isSignal'] != 1]
    data_train_signal = data_train[data_train['isSignal'] == 1]
    data_train_bkg = data_train[data_train['isSignal'] != 1]

    ### Saving unscaled test signal mass values
    m_test_unscaled_signal = data_test_signal['unscaledMass']
    unscaledTestMassPointsList = list(dict.fromkeys(list(m_test_unscaled_signal)))

    ### If testMass = 'all', defining testMass as the list of test signal masses 
    if testMass == ['all']:
        testMass = list(int(item) for item in set(list(m_test_unscaled_signal)))
    else:
        testMass = list(int(item) for item in testMass)
    testMass.sort()

    for unscaledMass in testMass:

        if unscaledMass != 1000:# and unscaledMass != 600:
            continue

        ### Checking whether there are test events with the selected mass
        if unscaledMass not in unscaledTestMassPointsList:
            print(Fore.RED + 'No test signal with mass ' + str(unscaledMass))
            continue

        ### Creating new output directory and log file
        newOutputDir = outputDirLoop + '/' + str(int(unscaledMass))
        print(format('Output directory: ' + Fore.GREEN + newOutputDir), checkCreateDir(newOutputDir))
        newLogFileName = newOutputDir + '/logFile_' + outputFileCommonName + '_' + str(unscaledMass) + '.txt'
        newLogFile = open(newLogFileName, 'w')

        ### Selecting only test signal events with the same mass value and saving them as an array
        data_test_signal_mass = data_test_signal[data_test_signal['unscaledMass'] == unscaledMass]
        scaledMass = list(set(list(data_test_signal_mass['mass'])))[0]
        X_test_signal_mass = np.asarray(data_test_signal_mass[InputFeatures].values).astype(np.float32)
        newLogFile.write(logInfo + '\nNumber of test signal events with mass ' + str(int(unscaledMass)) + ' GeV: ' + str(len(X_test_signal_mass)))
        wMC_test_signal_mass = np.array(data_test_signal_mass['weight'])

        ### Assigning the same mass value to test background events and saving them as an array
        data_test_bkg = data_test_bkg.assign(mass = np.full(len(data_test_bkg), scaledMass))
        X_test_bkg = np.asarray(data_test_bkg[InputFeatures].values).astype(np.float32)
        wMC_test_bkg = np.array(data_test_bkg['weight'])

        if doFeaturesRanking:
            X = np.concatenate((X_test_signal_mass, X_test_bkg))
            y = np.concatenate((np.ones(len(X_test_signal_mass)), np.zeros(len(X_test_bkg))))
            nIter = 100
            base_score, score_decreases = get_score_importances(PredictionAndAUC, X, y, n_iter = nIter)
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
            #FeaturesRanking(model, X_test_signal_mass, X_test_bkg, deltasDict, InputFeatures, signal, analysis, channel, outputDir, outputFileCommonName, drawPlots)

        if doStatisticTest:
            if iLoop == 0:
                ### Creating dataset with just 20 events
                data_test_signal_mass_10 = data_test_signal_mass[:10]
                data_test_bkg_10 = data_test_bkg[:10]
                data_test_20 = pd.concat((data_test_signal_mass_10, data_test_bkg_10))
                #print(data_test_20)
                #print(data_test_20.shape)
                data_test_20.to_pickle(outputDir + '/20events_test.pkl')
                print(Fore.GREEN + 'Saved ' + outputDir + '/20events_test.pkl')
                #print(yhat_20)
                #print(len(yhat_20))
            else:
                data_test_20 = pd.read_pickle(outputDir + '/20events_test.pkl')

            yhat_20 = np.array(model.predict(data_test_20[InputFeatures], batch_size = batchSize))

            for iScore in range(len(yhat_20)):
                scoresFile.write(str(yhat_20[iScore][0]) + ' ')
                if iScore == (len(yhat_20) - 1):
                    scoresFile.write('\n')

        ### Selecting train signal events with the same mass
        data_train_signal_mass = data_train_signal[data_train_signal['unscaledMass'] == unscaledMass]
        X_train_signal_mass = np.asarray(data_train_signal_mass[InputFeatures].values).astype(np.float32)
        wMC_train_signal_mass = np.array(data_train_signal_mass['weight'])

        ### Assigning the same mass value to train background events
        data_train_bkg = data_train_bkg.assign(mass = np.full(len(data_train_bkg), scaledMass))
        X_train_bkg = np.asarray(data_train_bkg[InputFeatures].values).astype(np.float32)
        wMC_train_bkg = np.array(data_train_bkg['weight'])

        ### Prediction on signal and background
        yhat_train_signal_mass, yhat_train_bkg_mass, yhat_test_signal_mass, yhat_test_bkg_mass = PredictionSigBkg(model, X_train_signal_mass, X_train_bkg, X_test_signal_mass, X_test_bkg, batchSize)

        ### Drawing confusion matrix
        yhat_test_mass = np.concatenate((yhat_test_signal_mass, yhat_test_bkg_mass))
        y_test_mass = np.concatenate((np.ones(len(yhat_test_signal_mass)), np.zeros(len(yhat_test_bkg_mass))))
        wMC_test_mass = np.concatenate((wMC_test_signal_mass, wMC_test_bkg))

        TNR, FPR, FNR, TPR = DrawCM(yhat_test_mass, y_test_mass, wMC_test_mass, newOutputDir, unscaledMass, background, outputFileCommonName, analysis, channel, preselectionCuts, signal, drawPlots)
        newLogFile.write('\nTNR (TN/N): ' + str(TNR) + '\nFPR (FP/N): ' + str(FPR) + '\nFNR (FN/P): ' + str(FNR) + '\nTPR (TP/P): ' + str(TPR))

        ### Computing ROC AUC
        fpr, tpr, thresholds = roc_curve(y_test_mass, yhat_test_mass)#, sample_weight = wMC_test_mass)
        roc_auc = auc(fpr, tpr)
        print(format(Fore.BLUE + 'ROC_AUC: ' + str(roc_auc)))
        newLogFile.write('\nROC_AUC: ' + str(roc_auc))
        if doStatisticTest:
            AUCfile.write(str(roc_auc) + '\n')

        '''
        from scipy import integrate
        sorted_index = np.argsort(fpr)
        fpr_sorted =  np.array(fpr)[sorted_index]
        tpr_sorted = np.array(tpr)[sorted_index]
        #auc = integrate.trapz(y = tpr_sorted, x = fpr_sorted)
        #print(format(Fore.BLUE + 'ROC_AUC: ' + str(auc)))
        #newLogFile.write('\nROC_AUC: ' + str(auc))
        #WP, bkgRejWP = DrawROCbkgRejectionScores(fpr_sorted, tpr_sorted, auc, newOutputDir, NN, unscaledMass, jetCollection, analysis, channel, preselectionCuts, signal, background, outputFileCommonName, yhat_train_signal_mass, yhat_test_signal_mass, yhat_train_bkg_mass, yhat_test_bkg_mass, wMC_train_signal_mass, wMC_test_signal_mass, wMC_train_bkg, wMC_test_bkg)
        '''

        ### Plotting ROC, background rejection and scores
        WP, bkgRejWP = DrawROCbkgRejectionScores(fpr, tpr, roc_auc, newOutputDir, NN, unscaledMass, analysis, channel, preselectionCuts, signal, background, outputFileCommonName, yhat_train_signal_mass, yhat_test_signal_mass, yhat_train_bkg_mass, yhat_test_bkg_mass, wMC_train_signal_mass, wMC_test_signal_mass, wMC_train_bkg, wMC_test_bkg, drawPlots)
        newLogFile.write('\nWorking points: ' + str(WP) + '\nBackground rejection at each working point: ' + str(bkgRejWP))
        
        ### Closing the newLogFile
        newLogFile.close()
        print(Fore.GREEN + 'Saved ' + newLogFileName)

        if drawPlots and doFeaturesRanking:
            PlotFeaturesRanking(InputFeatures, deltasDict, newOutputDir, outputFileCommonName)

        ### Plotting calibration curve on signal (train + test) scores
        #CalibrationCurves(wMC_test_signal_mass, wMC_train_signal_mass, yhat_test_signal_mass, yhat_train_signal_mass, wMC_test_bkg, wMC_train_bkg, yhat_test_bkg_mass, yhat_train_bkg_mass, unscaledMass, newOutputDir, outputFileCommonName)

        #WeightedDistributionComparison(data_train_signal_mass, data_train_bkg, data_test_signal_mass, data_test_bkg, yhat_train_signal_mass, yhat_train_bkg_mass, yhat_test_signal_mass, yhat_test_bkg_mass, 'lep1_eta')
        

    #if (len(testMass) > 1):
    #    DrawRejectionVsMass(testMass, WP, bkgRej90, bkgRej94, bkgRej97, bkgRej99, outputDir, analysis, channel, preselectionCuts, signal, background, outputFileCommonName) 

if doStatisticTest:
    AUCfile.close()
    print(Fore.GREEN + 'Saved ' + outputDir + '/AUCvalues.txt')
    lossFile.close()
    print(Fore.GREEN + 'Saved ' + outputDir + '/lossValues.txt')
    scoresFile.close()
    print(Fore.GREEN + 'Saved ' + outputDir + '/scoresValues.txt')
