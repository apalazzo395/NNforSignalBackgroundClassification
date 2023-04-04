# This script takes the input root files and converts the relevant branches into pkl
# It is not recommended to convert the whole input tree because the script may crash due to memory issues with large trees 

#import uproot
import uproot3
from Functions import ReadArgParser, ReadConfigSaveToPkl
from colorama import init, Fore
init(autoreset = True)

### Reading the command line
tag = ReadArgParser()

### Reading from config file
ntuplePath, inputFiles, dfPath, rootBranchSubSample = ReadConfigSaveToPkl(tag)

### Creating log file
logFileName = dfPath + 'EventsNumberNtuples_' + tag + '.txt'
logFile = open(logFileName, 'w')
logFile.write('CxAOD tag: ' + tag)
logFile.write('\nPath to the input nutples: ' + ntuplePath)
logFile.write('\nNumber of events in the input ntuples:\n')

### Loading, converting and saving each input file
totalEvents = 0 
weightedTotalEvents = 0 

for i in inputFiles:
    inFile = ntuplePath + i + '.root'
    print('Loading ' + inFile)
    theFile = uproot3.open(inFile)
    tree = theFile['Nominal']
    Nevents = tree.numentries
    totalEvents += Nevents
    if Nevents == 0:
        print(Fore.RED + 'Ignoring empty file')
        continue
    DF = tree.pandas.df(rootBranchSubSample)
    weightedEvents = DF['weight'].sum()
    weightedTotalEvents += weightedEvents
    print(Fore.BLUE + 'Number of events in ' + inFile, '->\t' + str(Nevents) + ' (weighted: ' + str(weightedEvents) + ')')
    logFile.write(i + ' -> ' + str(Nevents) + ' events (weighted: ' + str(weightedEvents) + ')')
    outFile = dfPath + i + '_DF.pkl'
    DF.to_pickle(outFile)
    '''
    with uproot.open(inFile) as theFile:
        tree = theFile['Nominal']
        Nevents = tree.num_entries
        totalEvents += Nevents
        if Nevents == 0:
            print(Fore.RED + 'Ignoring empty file')
            continue

        ### Converting to pandas dataframe only the variables listed in rootBranchSubSample
        DF = tree.arrays(rootBranchSubSample, library = 'pd')
    
        weightedEvents = DF['weight'].sum()
        weightedTotalEvents += weightedEvents
        print(Fore.BLUE + 'Number of events in ' + inFile, '->\t' + str(Nevents) + ' (weighted: ' + str(weightedEvents) + ')')
        logFile.write(i + ' -> ' + str(Nevents) + ' events (weighted: ' + str(weightedEvents) + ')\n')
        outFile = dfPath + i + '_DF.pkl'
        
        ### Saving output dataframe
        DF.to_pickle(outFile)
        print(Fore.GREEN + 'Saved ' + outFile)
    '''
print(Fore.BLUE + 'Total events: ' + str(totalEvents) + ' (weighted: ' + str(weightedTotalEvents) + ')')
logFile.write('Number of total events: ' + str(totalEvents) + ' (weighted: ' + str(weightedTotalEvents) + ')\n')
logFile.write('\npkl files saved in ' + dfPath)
logFile.close()
print(Fore.GREEN + 'Saved ' + logFileName)
