# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 09:44:20 2018

@author: cyfho
"""

import sys
sys.path.append(r'D:\Python Projects\Fiber photometry analysis project\Functions')
sys.path.append(r'D:\Python Projects\Behavior analysis project\Functions')
import os #for access folder
import glob #for access folder and file
import re  #for word processing
import numpy as np   #for calculaiton
import matplotlib.pyplot as plt   #for ploting
import pandas as pd    #for data loading and manipulation
from tqdm import tqdm #for progress bar in terminal
from rodentfun import rodentfun #import customized class
from scanFiles import scanFiles #import customized class
from fprecording import fprecording
#%%
#IDfile = r'C:\Users\cyfho\Desktop\oneID.txt'
#behfile = r'C:\Users\cyfho\Desktop\raw data\20180628\L94_DotsAlmond_2018_06_28__13_36_02.csv'

recfile = r'E:\03_Projects\38_Project 19. In vivo Photometery\05_Raw Data\20180530 D1D2CrRat homecage drinking contraRec\D1Rat_L90_left_homecage_EtOH_countra_Subt2_14-50-40-706.txt'

reffile = r'D:\Python Projects\Fiber photometry analysis project\10-Dual-Gs&Td.csv'

rec_output = r'E:\03_Projects\38_Project 19. In vivo Photometery\05_Raw Data\20180530 D1D2CrRat homecage drinking contraRec'

#unmixing_file = r'E:\03_Projects\38_Project 19. In vivo Photometery\05_Raw Data\20180628\L94_PavlovianAlcohol_R_unmixing.csv'
#%%
#rat = rodentfun (behfile, IDfile)
#beh_duration = rat.df['Time'].iloc[-1]
#trueFrame_N = beh_duration/0.04
#%%
#rat_rec = fprecording(recfile, reffile, unmixing_file = unmixing_file, rec_duration = beh_duration)
rat_rec = fprecording(recfile, reffile)
#error_time = (trueFrame_N - len(rat_rec.rawdata_reduce))*0.04
#print ('the recording error is: %s s'%error_time)
#%%
fig1 = rat_rec.plot_unmixingResult()
#%%
fig1.savefig(r'C:\Users\cyfho\Desktop\xxx.svg')
#%%
#rat_rec.save_unmixingResult(rec_output)
#%%



#%%
MagEntin_ts= pd.DataFrame()
MagEntin_ts['Time'] = rat.df['Time']
MagEntin_ts['MagEntin'] = rat.df.apply(lambda x: 1 if (x['Events'] == 'MagEntin') and (x['States'] == 'Input') else None, axis = 1)
MagEntin_ts.set_index('Time', inplace=True)

PreCS_time_s = rat.df[(rat.df['Events'] == 'PreCS') & (rat.df['States'] == 'Entry')]['Time']
Tone_time_s = rat.df[(rat.df['Events'] == 'Tone') & (rat.df['States'] == 'Entry')]['Time']
ToneR_time_s = rat.df[(rat.df['Events'] == 'Tone_Reward') & (rat.df['States'] == 'Entry')]['Time']
PostCS_time_s = rat.df[(rat.df['Events'] == 'PostCS') & (rat.df['States'] == 'Entry')]['Time']
PreCS_time_e = rat.df[(rat.df['Events'] == 'PreCS') & (rat.df['States'] == 'Exit')]['Time']
Tone_time_e = rat.df[(rat.df['Events'] == 'Tone') & (rat.df['States'] == 'Exit')]['Time']
ToneR_time_e = rat.df[(rat.df['Events'] == 'Tone_Reward') & (rat.df['States'] == 'Exit')]['Time']
PostCS_time_e = rat.df[(rat.df['Events'] == 'PostCS') & (rat.df['States'] == 'Exit')]['Time']
#event_window = (180,210)
#events = [('PreCS', 180, 'g'),
#          ('Tone', 190, 'b'),
#          ('Tone+Reward',194, 'r'),
#          ('PostCS', 200, 'g')]
#L94rec.event_analysis(event_window = event_window, events = events, expand = 2, shift = 0)
event_window = []
events = []
for n in range(len(Tone_time_s)):
    event_window.append((PreCS_time_s.iloc[n],PostCS_time_e.iloc[n]))
    events.append ([('PreCS',PreCS_time_s.iloc[n],'g'),
                    ('Tone',Tone_time_s.iloc[n],'b'),
                    ('Tone+Reward',ToneR_time_s.iloc[n],'r'),
                    ('PostCS', PostCS_time_s.iloc[n], 'g')])
#%%
##Examples 
##---------
#event_window[0]
#events[0]
temp = rat_rec.event_analysis(event_window = event_window[0], events= events[0], expand = 2, shift = 0, 
                              beh_data = MagEntin_ts, beh_target = 'MagEntin', beh_event_idx = 1)
#%%
rat_rec.blhistfig.savefig(r'C:\Users\cyfho\Desktop\xxx.svg')
#%%

temp = rat_rec.norm_EtOH

#%%
plt.plot(temp,'k-')
ax = plt.gca()
fig = plt.gcf()
ax.fill_between(temp.index, temp.values,'r')

rat_rec._format_ax(ax, xlabel = 'Time(s)', ylabel = '%Change of Bl')
#%%
fig.savefig(r'C:\Users\cyfho\Desktop\xxx-2.svg')

#%%
AUC_results = pd.DataFrame()
AUC_t_results = pd.DataFrame()
sampleTrace_df = pd.DataFrame()   
AUC_results['event'] = temp['event']
AUC_t_results['event'] = temp['event']

#%%
for n in range(len(Tone_time_s)):
    temp = rat_rec.event_analysis(event_window = event_window[n], events= events[n], expand = 2, shift = 0, 
                                  beh_data = MagEntin_ts, beh_target = 'MagEntin', beh_event_idx = n+1, 
                                  blhist_display = False, show_result = False)
    AUC_results[n+1] = temp['AUC']
    AUC_t_results[n+1] = temp['AUC/t']
    sampleTrace_df = pd.concat([sampleTrace_df,rat_rec.get_RecEventSample(n, 0.02)], axis = 1)

AUC_t_results.drop(index = [0,5], inplace=True)
#%%
with pd.ExcelWriter(os.path.join(rec_output, '%s timelock_analysis.xlsx'%rat.subject), engine='xlsxwriter') as writer:
    AUC_t_results.to_excel (writer, sheet_name = 'AUC_t')
    sampleTrace_df.to_excel (writer, sheet_name = 'sample trace')
    print ('Export data to the Excel')

    
    
    
    
    
    
    
    
