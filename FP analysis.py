# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 22:40:40 2018

Author: Yifeng Cheng, Texas A&M University Health Science Center
"""

import sys
sys.path.append(r'D:\Python Projects\Fiber photometry analysis project\Functions')
import os #for access folder
import glob #for access folder and file
import re  #for word processing
import time
from fprecording import fprecording #import customized class for analysis fiber photometry signal
import pandas as pd #for data handling
from tqdm import tqdm #for progress bar in terminal
tqdm.pandas()  #provide progress bar
#%%
path =r'C:\Users\cyfho\Desktop'  #set folder path
#path = input('Provide data folder path: \n')
allFiles = glob.glob(os.path.join(path, "*.txt"))  #get all files's path
num_files = len(allFiles)   
print ('\nFind following files in the foler ...\\%s:\n'%path)
for num, f in enumerate(allFiles,1):
    print (num,":",re.split('_Sub|\\\\',f)[-2])  #provide file name (fixed). Now, works for both full path and shortpath 

#%%
file_num = 1           
f = allFiles[file_num-1] #open the ith file
filename = re.split('_Sub|\\\\',f)[-2]    #get the filename 
ref_file = r'D:\Python Projects\Fiber photometry analysis project\10-Dual-Gs&Td.csv'
#%%
rat_rec = fprecording (datafile = f, ref_file = ref_file, auto_unmixing = True, plot_unmixing = False)
#%%
with pd.ExcelWriter ('C:/Users/cyfho/Desktop/temp.xlsx') as writer:
    if False:
        rat_rec.raw_G.to_excel (writer, sheet_name = 'aa')
    if True:
        rat_rec.raw_R.to_excel (writer, sheet_name = 'bb')
#%%    
rat_rec.raw_G

#%%
event_window = (0,1200)
events = [('EtOH',0, 'r')]
AUC_result = rat_rec.event_analysis(event_window = event_window, events = events, expand = 2)
#%%
with pd.ExcelWriter(os.path.join(path, 'L77_EtOH.xlsx'), engine='xlsxwriter') as writer:
    AUC_result.to_excel(writer, sheet_name = 'L77 saline')
    print ('Export data to the Excel')
    
#%%
#unmixing_file = r'C:\Users\cyfho\Desktop\L94_PavlovianAlcohol_R_unmixing.csv'    
#Exmaples:
#-----------------
#>>D1M59 = fprecording(datafile = f, ref_file = ref_file, auto_unmixing = True)
#>>D1M59 = fprecording(datafile = f, ref_file = ref_file, unmixing_file = unmixing_file)
#D1M59.plot_unmixingResult()
#event_window = (180,210)
#events = [('PreCS', 180, 'g'),
#          ('Tone', 190, 'b'),
#          ('Tone+Reward',194, 'r'),
#          ('PostCS', 200, 'g')]

#L94rec.event_analysis(event_window = event_window, events = events, expand = 2, shift = 0)

