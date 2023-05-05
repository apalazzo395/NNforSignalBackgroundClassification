import pandas as pd
from termcolor import colored, cprint
'''
parser = ArgumentParser(add_help = False)
parser.add_argument('-a', '--Analysis', help = 'Type of analysis: \'merged\' or \'resolved\'', type = str)
parser.add_argument('-c', '--Channel', help = 'Channel: \'ggF\' or \'VBF\'', type = str)
parser.add_argument('-s', '--Signal', help = 'Signal: \'VBFHVTWZ\', \'Radion\', \'RSG\', \'VBFRSG\', \'HVTWZ\' or \'VBFRadion\'', type = str)

args = parser.parse_args()

analysis = args.Analysis
channel = args.Channel
signal = args.Signal    
'''
analysis = 'merged'
channel = 'VBF'
signal = 'Radion'

passDict = {'merged':{'ggF': {'Radion': ['Pass_VV2Lep_MergHP_GGF_ZZ_2btag_SR', 'Pass_VV2Lep_MergLP_GGF_ZZ_2btag_SR', 'Pass_VV2Lep_MergHP_GGF_ZZ_01btag_SR', 'Pass_VV2Lep_MergLP_GGF_ZZ_01btag_SR', 'Pass_VV2Lep_MergHP_GGF_ZZ_2btag_ZCR', 'Pass_VV2Lep_MergLP_GGF_ZZ_2btag_ZCR', 'Pass_VV2Lep_MergHP_GGF_ZZ_01btag_ZCR', 'Pass_VV2Lep_MergLP_GGF_ZZ_01btag_ZCR'],
                            'RSG': ['Pass_VV2Lep_MergHP_GGF_ZZ_2btag_SR', 'Pass_VV2Lep_MergLP_GGF_ZZ_2btag_SR', 'Pass_VV2Lep_MergHP_GGF_ZZ_01btag_SR', 'Pass_VV2Lep_MergLP_GGF_ZZ_01btag_SR', 'Pass_VV2Lep_MergHP_GGF_ZZ_2btag_ZCR', 'Pass_VV2Lep_MergLP_GGF_ZZ_2btag_ZCR', 'Pass_VV2Lep_MergHP_GGF_ZZ_01btag_ZCR', 'Pass_VV2Lep_MergLP_GGF_ZZ_01btag_ZCR'],
                            'HVTWZ': ['Pass_VV2Lep_MergHP_GGF_WZ_SR', 'Pass_VV2Lep_MergLP_GGF_WZ_SR', 'Pass_VV2Lep_MergHP_GGF_WZ_ZCR', 'Pass_VV2Lep_MergLP_GGF_WZ_ZCR']},
                    'VBF': {'Radion': ['Pass_VV2Lep_MergLP_VBF_ZZ_SR', 'Pass_VV2Lep_MergHP_VBF_ZZ_SR', 'Pass_VV2Lep_MergHP_VBF_ZZ_ZCR', 'Pass_VV2Lep_MergLP_VBF_ZZ_ZCR'],
                            'RSG': ['Pass_VV2Lep_MergLP_VBF_ZZ_SR', 'Pass_VV2Lep_MergHP_VBF_ZZ_SR', 'Pass_VV2Lep_MergHP_VBF_ZZ_ZCR', 'Pass_VV2Lep_MergLP_VBF_ZZ_ZCR'],
                            'HVTWZ': ['Pass_VV2Lep_MergHP_VBF_WZ_SR', 'Pass_VV2Lep_MergLP_VBF_WZ_SR', 'Pass_VV2Lep_MergHP_VBF_WZ_ZCR', 'Pass_VV2Lep_MergLP_VBF_WZ_ZCR']}
                },
            'resolved': {'ggF': {'Radion': ['Pass_VV2Lep_Res_GGF_ZZ_2btag_SR', 'Pass_VV2Lep_Res_GGF_ZZ_01btag_SR', 'Pass_VV2Lep_Res_GGF_ZZ_2btag_ZCR', 'Pass_VV2Lep_Res_GGF_ZZ_01btag_ZCR'],
                                 'RSG': ['Pass_VV2Lep_Res_GGF_ZZ_2btag_SR', 'Pass_VV2Lep_Res_GGF_ZZ_01btag_SR', 'Pass_VV2Lep_Res_GGF_ZZ_2btag_ZCR', 'Pass_VV2Lep_Res_GGF_ZZ_01btag_ZCR'],
                                 'HVTWZ': ['Pass_VV2Lep_Res_GGF_WZ_SR', 'Pass_VV2Lep_Res_GGF_WZ_ZCR']},
                         'VBF': {'Radion': ['Pass_VV2Lep_Res_VBF_ZZ_SR', 'Pass_VV2Lep_Res_VBF_ZZ_ZCR'],
                                 'RSG': ['Pass_VV2Lep_Res_VBF_ZZ_SR', 'Pass_VV2Lep_Res_VBF_ZZ_ZCR'],
                                 'HVTWZ': ['Pass_VV2Lep_Res_VBF_WZ_SR', 'Pass_VV2Lep_Res_VBF_WZ_ZCR']}
                     }
        }



inputDfMerged = pd.read_pickle('/nfs/kloe/einstein4/HDBS/NNoutput/r33-24/reader30032023/r33-24/merged/ggF/none/Radion/all/MixData_r33-24_merged_ggF_none_Radion_all.pkl')
inputDfResolved = pd.read_pickle('/nfs/kloe/einstein4/HDBS/NNoutput/r33-24/reader30032023/r33-24/resolved/ggF/none/Radion/all/MixData_r33-24_resolved_ggF_none_Radion_all.pkl')

dataFrame = pd.concat([inputDfMerged, inputDfResolved], ignore_index = True)

combinedMergResPass = passDict['merged'][channel][signal] + passDict['resolved'][channel][signal]
print('combinedMergResPass: ' + str(combinedMergResPass))

dfDict = {}
for passVar in combinedMergResPass:
    dataSetSingleRegion = dataFrame[dataFrame[passVar] == True]
    dfDict[passVar] = dataSetSingleRegion

for iPass in range(len(combinedMergResPass)):
    passVar = combinedMergResPass[iPass]
    cprint('First region: ' + passVar, 'blue')
    for iPass2 in range(iPass + 1, len(combinedMergResPass)):
        passVar2 = combinedMergResPass[iPass2]
        print('Searching for overlap between ' + passVar + ' and ' + passVar2 + ' in whole (signal + background) dataset')
        intDf = pd.merge(dfDict[passVar], dfDict[passVar2], how ='inner')
        if (intDf.shape[0] != 0):
            cprint('Found intersection between region ' + passVar + ' and ' + passVar2 + ' -> found ' + str(intDf.shape[0]) + ' common events!', 'red')
    cprint('----------------', 'blue')
