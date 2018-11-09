# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 22:50:57 2018

@author: cyfho
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 23:08:36 2018

@author: cyfho
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 19:59:35 2018

@author: cyfho
"""
import sys
sys.path.append(r'D:\Python Projects\Fiber photometry analysis project\Functions')
sys.path.append(r'D:\Python Projects\Behavior analysis project\Functions')
import os  #  for access folder
# #import glob #for access folder and file
import re  #for word processing
import numpy as np   #for calculaiton
#import matplotlib.pyplot as plt   #for ploting
import pandas as pd    #for data loading and manipulation
import warnings
from openpyxl import load_workbook
from datetime import datetime
from tqdm import tqdm #for progress bar in terminal
from rodentfun import rodentfun #import customized class
from scanFiles import scanFiles #import customized class

def beh_analysis (IDfile, input_folder, date_range, output_folder, trialAnalysis_suffix, summary_filename, Input_dict,
                  session_analysis, InputOnOff_dict, trial_analysis, trialWindow_dict, InputAnalysis_dict):
    #scan file folders and get all files
    Behfiles = scanFiles (input_folder,*['.csv','.xlsx','.xlx']) 
    files = Behfiles.get(date_range)
    
    #start analysis
    ##initialize some dictionary for later use
    summary_dict = {key[1]: pd.DataFrame() for key in Input_dict.values()} #for session analysis
    
    sessionAnalysisDate_dict = {} #for session analysis
    trialAnalysisDate_dict = {}
    
    for d in tqdm([d.strftime('%Y%m%d') for d in pd.date_range(str(date_range[0]), str(date_range[1]))]):  #tdqm add progress bar
    
        #every new date initialize a new dataframe for session analysis
        IEI_dict = {key[1]: pd.DataFrame() for key in Input_dict.values()}
        IEI_cumP_dict = {key[1]: pd.DataFrame() for key in Input_dict.values()}
        duration_dict = {key[1]: pd.DataFrame() for key in Input_dict.values()}
        distrib_dict = {key[1]: pd.DataFrame() for key in Input_dict.values()}
        latency_dict = {key[1]: pd.DataFrame() for key in Input_dict.values()}
        
        sessionAnalysisRat_dict = {key[1]: pd.DataFrame() for key in Input_dict.values()} #for session analysis
        
        trialAnalysisRat_dict= {} # initiate a new dictionary for each experiment date for trial analysis
        
        #iterate through all animals' data
        for f in files:
            rat = rodentfun (f, IDfile)
            
            
            #check the experiment date
            if datetime.strptime(rat.date, "%m/%d/%Y").strftime("%Y%m%d") == str(d):
                
                try:
                    
                #check if this file contains a right animal
                #if rat.subject == '':
                #    pass
                
                    #session analysis
                    if session_analysis:
                        sessionResult = rat.session_counting(Input_dict, IEI_timeRange = (0, 900, 10), cal_IEI = False, cal_duration = True, cal_latency = True, cal_inputDistribution = True, num_bins = 6, Input1OnOff = InputOnOff_dict['Input1OnOff'], Input2OnOff = InputOnOff_dict['Input2OnOff'], Input3OnOff = InputOnOff_dict['Input3OnOff'], Input4OnOff =InputOnOff_dict['Input4OnOff'])
                        
                        for value in Input_dict.values():
                            summary_dict [value[1]].at[rat.subject, rat.date] = sessionResult[value[1]]['sum']
                            latency_dict [value[1]].at[rat.subject, 0] = sessionResult[value[1]]['latency']
                            IEI_cumP_dict [value[1]] =  pd.concat ([IEI_cumP_dict[value[1]], sessionResult[value[1]]['IEI_cumP']], axis = 1)
                            IEI_dict [value[1]] = pd.concat ([IEI_dict[value[1]], sessionResult[value[1]]['IEI']], axis = 1)
                            duration_dict [value[1]] = pd.concat ([duration_dict[value[1]], sessionResult[value[1]]['duration']], axis = 1)
                            distrib_dict [value[1]] = pd.concat ([distrib_dict[value[1]], sessionResult[value[1]]['Distrib']], axis = 1)
                    
                    #trial analysis
                    if trial_analysis:
                        trial_result = rat.trial_counting(Input_dict, trialWindow_dict, model = 's-e', Input1Analysis = InputAnalysis_dict['Input1Analysis'], Input2Analysis = InputAnalysis_dict['Input2Analysis'], Input3Analysis = InputAnalysis_dict['Input3Analysis'], Input4Analysis = InputAnalysis_dict['Input4Analysis'])
                        
                        trialAnalysisRat_dict [rat.subject] = trial_result
                
                except:
                    filename = re.split('\\\\',f)[-1] 
                    print ('%s may have problem'%filename)
        
    
        ###Concatenate results
        ###################concatenate session analysis results
        for key in sessionAnalysisRat_dict:
            sessionAnalysisRat_dict [key] = pd.concat([IEI_cumP_dict[key].T, IEI_dict[key].T, duration_dict[key].T, distrib_dict[key].T, latency_dict[key]], keys = ['IEI_cumP', 'IEI', 'duration', 'distribution', 'latency'])
            sessionAnalysisRat_dict [key].reset_index(inplace=True)
        
        if np.sum([len(sessionAnalysisRat_dict[key]) for key in sessionAnalysisRat_dict]) == 0:
            pass
        else:
            sessionAnalysisDate_dict[d] = sessionAnalysisRat_dict
        
        ##############summary trial analysis results
        #create a function to average or sum the trial analysis results
        def get_trialSummary (k, rat, measure, method = 'sum'):
            temp = trialAnalysisRat_dict[rat]
            temp = temp[(temp['level_0'] == k)&(temp['Measurement'] == measure)].copy()
            temp.drop (columns = ['level_0', 'Measurement', '# trials'], inplace = True)
            if method == 'sum':
                return np.nansum (temp.values)
            elif method == 'mean':
                #handle runtime warning because sometimes the date could be all NaN
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        ave = np.nanmean (temp.values)
                    except RuntimeWarning:
                        ave = np.NaN
                return ave
        
        #index results in a dictionary   
        trialSummary_dict = {key[1]: pd.DataFrame() for key in Input_dict.values()}
        for rat in trialAnalysisRat_dict:
            for k in trialSummary_dict:
                trialSummary_dict[k].at[rat, 'count_sum'] = get_trialSummary (k, rat, 'sum', method = 'sum')
                trialSummary_dict[k].at[rat, 'duration_sum'] = get_trialSummary (k, rat, 'duration', method = 'sum')
                trialSummary_dict[k].at[rat, 'latency_sum'] = get_trialSummary (k, rat, 'latency', method = 'sum')
                
                trialSummary_dict[k].at[rat, 'count_ave'] = get_trialSummary (k, rat, 'sum', method = 'mean')
                trialSummary_dict[k].at[rat, 'duration_ave'] = get_trialSummary (k, rat, 'duration', method = 'mean')
                trialSummary_dict[k].at[rat, 'latency_ave'] = get_trialSummary (k, rat, 'latency', method = 'mean')
    
        #update old trialAnalysis file       
        trialAnalysisRat_dict.update (trialSummary_dict)
        
        #concatenate trial analysis results
        if np.sum([len(trialAnalysisRat_dict[key]) for key in trialAnalysisRat_dict]) == 0:
            pass
        else:
            trialAnalysisDate_dict [d] = trialAnalysisRat_dict
    
    
                
    #write summary data into excel file
    if session_analysis:
        print ('Start writing summary analysis results into excel:\n')
        output_dir = os.path.join(output_folder, summary_filename)
        #check if writing path exists, if not, create a folder
        if not os.path.exists(os.path.dirname(output_dir)):
            print ('folder and file dosen\'t exist, creating a new folder and file:\n')
            os.makedirs(os.path.dirname(output_dir), exist_ok=True)  #make a root folder for group analysis
            #with pd.ExcelWriter(output_dir, engine='xlsxwriter') as writer:
            writer = pd.ExcelWriter(output_dir, engine='openpyxl') 
            for key in summary_dict:
                data = summary_dict[key].copy()
                data.to_excel (writer, sheet_name = key)
                print ('Writing summary %s data to the excel sheet...'%key)
            writer.save()
            print ('Done!\n')
        
        else:
            #Check if summary file exists, if not, write all data into a new excel 
            if not os.path.isfile(output_dir):
                print ('file dosen\'t exist, creating a new file:\n')
                writer = pd.ExcelWriter(output_dir, engine='openpyxl') 
                for key in summary_dict:
                    data = summary_dict[key].copy()
                    #with pd.ExcelWriter(output_dir, engine='xlsxwriter') as writer:
                    data.to_excel (writer, sheet_name = key)
                    print ('Writing summary %s data to the Excel...'%key)
                writer.save()
                print ('Done!\n')
            #if yes, write current data into exisitng files without duplication
            else:
                print ('Found an exisiting file, writing new data into this file:\n')
                #load current excel
                book = load_workbook (output_dir)
                writer = pd.ExcelWriter (output_dir, engine='openpyxl')
                writer.book = book
                #preserve all current sheets
                writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
                
                for key in summary_dict:
                    data = summary_dict[key].copy()
                    
                    oldSheet = pd.read_excel (output_dir, sheet_name = key, na_filter = False)
            
                    for excel_col in oldSheet.columns:
                        for df_col in summary_dict[key]:
                            if excel_col == df_col:
                                data.drop(columns = df_col,inplace = True)
                    
                    ##add more data into the datasheets
                    newsheet = pd.concat([data, oldSheet], axis = 1)
                    ##sort the column
                    newsheet.sort_index (axis = 1, inplace = True)
                    print ('Writing new summary %s data to the Excel...'%key)
                    newsheet.to_excel(writer, sheet_name = key)
            
                writer.save()
                print ('Done!\n')
                
        ##Write session analysis data into individual date file
        print ('\nStatrt writing session analysis results into files:\n')
        sessionAnalysis_output_folder = os.path.join(output_folder, 'SessionAnalysis')
        for d in sessionAnalysisDate_dict:
            print ('Writing %s session analysis result into excel file...'%d)
            sessionAnalysis_output_dir = os.path.join(sessionAnalysis_output_folder, str(d)+'_sessionReduce.xlsx')
            if not os.path.exists(os.path.dirname(sessionAnalysis_output_dir)):
                os.makedirs(os.path.dirname(sessionAnalysis_output_dir), exist_ok=True)
                with pd.ExcelWriter(sessionAnalysis_output_dir, engine='openpyxl') as writer:
                    for key in sessionAnalysisDate_dict[d]:
                        sessionAnalysisDate_dict[d][key].to_excel(writer, sheet_name = key)
                print ('Done!')
            else:
                if not os.path.isfile(sessionAnalysis_output_dir):
                    with pd.ExcelWriter(sessionAnalysis_output_dir, engine='openpyxl') as writer:
                        for key in sessionAnalysisDate_dict[d]:
                            sessionAnalysisDate_dict[d][key].to_excel(writer, sheet_name = key)
                    print ('Done!')
                else:
                    print ('%s File exists'%d)
                    pass
    else:
        print ('No new data is being written')
    
    if trial_analysis:
        ##Write trail analysis data into individual date file
        print ('\nStatrt writing trial analysis results into files:\n')
        trailAnalysis_output_folder = os.path.join(output_folder, 'TrialAnalysis')
        for d in trialAnalysisDate_dict:
            print ('Writing %s trial analysis result into excel file...'%d)
            trailAnalysis_output_dir = os.path.join(trailAnalysis_output_folder, str(d)+'_%s.xlsx'%trialAnalysis_suffix)
            if not os.path.exists(os.path.dirname(trailAnalysis_output_dir)):
                os.makedirs(os.path.dirname(trailAnalysis_output_dir), exist_ok=True)
                with pd.ExcelWriter(trailAnalysis_output_dir, engine='openpyxl') as writer:
                    for key in trialAnalysisDate_dict[d]:
                        trialAnalysisDate_dict[d][key].to_excel(writer, sheet_name = key)
                print ('Done!')
            else:
                if not os.path.isfile(trailAnalysis_output_dir):
                    with pd.ExcelWriter(trailAnalysis_output_dir, engine='openpyxl') as writer:
                        for key in trialAnalysisDate_dict[d]:
                            trialAnalysisDate_dict[d][key].to_excel(writer, sheet_name = key)
                    print ('Done!')
                else:
                    print ('%s File exists'%d)
                    pass

                      
                      
                      
                      