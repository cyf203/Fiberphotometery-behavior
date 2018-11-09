# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 23:43:58 2018

@author: cyfho
"""
from __future__ import division, print_function

__author__ = "Yifen Cheng"
__version__ = "1.0.0"
__license__ = "NA"


try:
    import re
except ImportError:
    print('pandas are not available. Run \'pip install re\' in the terminal')
    
try:
    import pandas as pd
except ImportError:
    print('pandas are not available. Run \'pip install pandas\' in the terminal')
    
try:
    import numpy as np
except ImportError:
    print('pandas are not available. Run \'pip install numpy\' in the terminal')

class rodentfun:
    """A vehicle for sale by Jeffco Car Dealership.

    Attributes:
        subject: 
        date: 
        df: 
        event: 
        session_counting: 
        trail_counting: 
    """
    
    def __init__ (self, csvfile, IDfile):
        self.subject = 'NA'
        self.date = '1/1/1111'
        self.df = pd.DataFrame()
        self.event = pd.DataFrame()
        self.subject, self.date, self.df = self.__get_data(csvfile, IDfile)
        
    #private method    
    def __get_data (self, csvfile, IDfile):
        
        #get_kwards = lambda x: ("","") if x == None else x
        
        #input1on, input1off = get_kwards(ratInput.get('Input1'))
        #input2on, input2off = get_kwards(ratInput.get('Input2'))
        #input3on, input3off = get_kwards(ratInput.get('Input3'))
        #input4on, input4off  = get_kwards(ratInput.get('Input4'))
        
        df = pd.DataFrame()
        with open(csvfile) as file:
            for num, line in enumerate(file,1):
                l = [t for t in re.split(',|\n',line) if t != '']
                if 'Subject' in l:
                    subject = l[1]
                if 'Date' in l:
                    date = l[1]
                if 'Start' in l:
                    startline = num
                    break
        IDlist = self.__get_id (IDfile)
        if subject in IDlist:
            df = pd.read_csv (csvfile, header = None, sep=',', skiprows = startline - 1, usecols = [1,2,3,4], na_filter = False)
            df.columns = ['Time','States','Code','Events']
            df['Time'] = df['Time'].apply(self.__getSec)
            #add timestamp mark for each input. On = 1 off =-1
            #if Input exists, calculate 
            #if ratInput.get('Input1') != None:
            #    df[input1on] = df['Events'].apply(lambda x: self.__input_mark(x, input1on, input1off))
            #if ratInput.get('Input2') != None:
            #    df[input2on] = df['Events'].apply(lambda x: self.__input_mark(x, input2on, input2off))
            #if ratInput.get('Input3') != None:
            #    df[input3on] = df['Events'].apply(lambda x: self.__input_mark(x, input3on, input3off))
            #if ratInput.get('Input4') != None:
            #    df[input4on] = df['Events'].apply(lambda x: self.__input_mark(x, input4on, input4off))
            return subject, date, df
        else:
            return 'NA','1/1/1111',df
    
    #private method
    def __get_id (self, IDfile):
        with open(IDfile) as f:
            ID = [line.strip('\n') for line in f]
        return ID
    
    #private method
    def __getSec (self, s) -> float:
            """
            Convert hh:mm:ss.ms string to second
            
            Parameter
            ---------
                First: string
                        first param named 's'
            
            Returns
            ------
                Float
                    a float number
            
            Exmaples
            --------
            >> getSec('20') --> 20
            >> getSec('1:20') -->  80
            >> getSec('1:30:01') --> 5401
            >> getSec('1:07:40.220') --> 4060.2200000000003
            """
            if type(s) == float:
                return round(s,3)
            elif type(s) == str:
                if '.' in s:  
                    l = list(map(float, re.split(':|\\.',s))) 
                    return round(sum(n * sec for n, sec in zip(l[::-1], (0.001, 1, 60, 3600))),3)
                else:
                    l = list(map(float, re.split(':',s))) 
                    return round(sum(n * sec for n, sec in zip(l[::-1], (1, 60, 3600))),3)
    
    def session_counting (self, target_dict, IEI_timeRange = (0, 900, 1), cal_IEI = True, cal_duration = True, cal_latency = True, cal_inputDistribution = True, num_bins = 10, **kwards):
        """
        This function aims to analysis all animal input and its triggered events in the sessions
        """
        if len(self.df) == 0:
            #if empty then pass without analysis
            pass
        else:
            get_kwards = lambda x: ("","") if x == None else x
            input1OnOff = get_kwards(kwards.get('Input1OnOff'))
            input2OnOff = get_kwards(kwards.get('Input2OnOff'))
            input3OnOff = get_kwards(kwards.get('Input3OnOff'))
            input4OnOff = get_kwards(kwards.get('Input4OnOff'))
            
            #get a start and end time point
            if 'Start' in self.df['States'].values:
                start = self.df[(self.df['Events'] == '') & (self.df['States'] == 'Start')]['Time']
            else:
                start = pd.Series({'Start': self.df['Time'].values[0]-1})
                
            if 'Finish' in self.df['States'].values:
                end = self.df[(self.df['Events'] == '') & (self.df['States'] == 'Finish')]['Time']
            else:
                end = pd.Series({'Finish': self.df['Time'].values[-1]+1})  #plus 1 second
            
            #get data from each event window
            event = self.df[(self.df['Time']>= start.iloc[0])&(self.df['Time']<= end.iloc[0])]
            
            result_dict = {}
            for k, v in target_dict.items():
                
                if k == 'Input1':
                    result_dict [v[1]] = self.__counting (event, v, self.subject, IEI_timeRange, cal_IEI, cal_duration, cal_latency, cal_inputDistribution, num_bins, InputOnOff = input1OnOff)
                elif k == 'Input2':
                    result_dict [v[1]] = self.__counting (event, v, self.subject, IEI_timeRange, cal_IEI, cal_duration, cal_latency, cal_inputDistribution, num_bins, InputOnOff = input2OnOff)
                elif k == 'Input3':
                    result_dict [v[1]] = self.__counting (event, v, self.subject, IEI_timeRange, cal_IEI, cal_duration, cal_latency, cal_inputDistribution, num_bins, InputOnOff = input3OnOff)
                elif k == 'Input4':
                    result_dict [v[1]] = self.__counting (event, v, self.subject, IEI_timeRange, cal_IEI, cal_duration, cal_latency, cal_inputDistribution, num_bins, InputOnOff = input4OnOff)
                else:
                    result_dict [v[1]] = self.__counting (event, v, self.subject)
            
            return result_dict
    
    
    #public method    
    def trial_counting(self, target_dict, trialWindow_dict, IEI_timeRange = (0, 900, 1), model = 's-e', **kwards):
        """
        This function aims to analysis animal inputs and triggered events based on individual trail
        Parameters
        ----------
        First: datafarme
                the 1st param name 'df', pass individual data
            
        Fourth: **kwards, optional 
                {'start'： tuple ('state', 'event'),
                 'end': tuple ('state', 'event'),
                 'time2start': tuple (BF, AF)}
                 'Input1Analysis': dict
                 'Input2Analysis': dict
                 'Input3Analysis': dict
                 'Input4Analysis': dict
        
        Returns
        --------
        Dictonary
            
        Examples
        --------
        rat.event_counting(target = ('Input','MagEntin'))
        
        """
        #check if dataframe is empty
        if len(self.df) == 0:
            #if empty then pass without analysis
            pass
        else:
            #pass all the key words in
            get_kwards = lambda x: {'cal_IEI':False, 'cal_duration':False, 'cal_latency': False, 'cal_inputDistribution': False,'InputOnOff': ("","")} if x == None else x
            Input1Analysis = get_kwards(kwards.get('Input1Analysis'))
            Input2Analysis = get_kwards(kwards.get('Input2Analysis'))
            Input3Analysis = get_kwards(kwards.get('Input3Analysis'))
            Input4Analysis = get_kwards(kwards.get('Input4Analysis'))
            
            cal_IEI1, cal_duration1, cal_latency1, cal_inputDistribution1, input1OnOff = Input1Analysis['cal_IEI'],  Input1Analysis['cal_duration'], Input1Analysis['cal_latency'], Input1Analysis['cal_inputDistribution'], Input1Analysis['InputOnOff']
            
            cal_IEI2, cal_duration2, cal_latency2, cal_inputDistribution2, input2OnOff = Input2Analysis['cal_IEI'], Input2Analysis['cal_duration'], Input2Analysis['cal_latency'], Input2Analysis['cal_inputDistribution'], Input2Analysis['InputOnOff']
            
            cal_IEI3, cal_duration3, cal_latency3, cal_inputDistribution3, input3OnOff = Input3Analysis['cal_IEI'], Input3Analysis['cal_duration'], Input3Analysis['cal_latency'], Input3Analysis['cal_inputDistribution'], Input3Analysis['InputOnOff']
            
            cal_IEI4, cal_duration4, cal_latency4, cal_inputDistribution4, input4OnOff = Input4Analysis['cal_IEI'], Input4Analysis['cal_duration'], Input4Analysis['cal_latency'], Input4Analysis['cal_inputDistribution'], Input4Analysis['InputOnOff']
            
            
            BF, AF, s_state, s_event, e_state, e_event = "", "", "", "", "", ""
            #get each trial window start and end
            for key in trialWindow_dict:
                if key == 'start':
                    s_state, s_event = trialWindow_dict[key]
                elif key == 'end':
                    e_state, e_event = trialWindow_dict[key]
                elif key == 'before':
                    BF = trialWindow_dict[key]
                elif key == 'after':
                    AF = trialWindow_dict[key]
            
            #three way to define a event window:
            #get a start and end time point
            ## have a specific start event and end event
            if BF == "" and (AF == ""):
                start = self.df[(self.df['Events'] == s_event) & (self.df['States'] == s_state)]['Time']
                end = self.df[(self.df['Events'] == e_event) & (self.df['States'] == e_state)]['Time']
            ## have a specific start event and a specific time window
            elif BF =="":
                start = self.df[(self.df['Events'] == s_event) & (self.df['States'] == s_state)]['Time']
                end = start + AF
            ## have a specific end event and a specific time window
            else:
                end = self.df[(self.df['Events'] == s_event) & (self.df['States'] == s_state)]['Time']
                start = end + BF
            
            
            result_dict = {}
            #check if original datafile miss one frame or not
            if len(start) != len(end):
                print ('%s %s data file error with %s: start time and end time dosen\'t match'%(self.date, self.subject,s_event))
                print ('\nstart: %s and end: %s'(start, end))
                pass
            #if not event exists return 0
            elif len(start) == 0:
                pass
            #count events in the specific time window
            else:
                
                def analysis ():
                    for k, v in target_dict.items():
                        if k == 'Input1':
                            result_dict [v[1]] = self.__counting (event, v, n, IEI_timeRange, cal_IEI1, cal_duration1, cal_latency1, cal_inputDistribution1, InputOnOff = input1OnOff)
        
                        elif k == 'Input2':
                            result_dict [v[1]] = self.__counting (event, v, n, IEI_timeRange, cal_IEI2, cal_duration2, cal_latency2, cal_inputDistribution2, InputOnOff = input2OnOff)
                            
                        elif k == 'Input3':
                            result_dict [v[1]] = self.__counting (event, v, n, IEI_timeRange, cal_IEI3, cal_duration3, cal_latency3, cal_inputDistribution3, InputOnOff = input3OnOff)
                            
                        elif k == 'Input4':
                            result_dict [v[1]] = self.__counting (event, v, n, IEI_timeRange, cal_IEI4, cal_duration4, cal_latency4, cal_inputDistribution4, InputOnOff = input4OnOff)
                            
                        else:
                            result_dict [v[1]] = self.__counting (event, v, n)
                        
                        summary_dict [v[1]].at[n, 0] = result_dict[v[1]]['sum']
                        latency_dict [v[1]].at[n, 0] = result_dict[v[1]]['latency']
                        IEI_cumP_dict [v[1]] =  pd.concat ([IEI_cumP_dict[v[1]], result_dict[v[1]]['IEI_cumP']], axis = 1)
                        IEI_dict [v[1]] = pd.concat ([IEI_dict[v[1]], result_dict[v[1]]['IEI']], axis = 1)
                        duration_dict [v[1]] = pd.concat ([duration_dict[v[1]], result_dict[v[1]]['duration']], axis = 1)
                        distrib_dict [v[1]] = pd.concat ([distrib_dict[v[1]], result_dict[v[1]]['Distrib']], axis = 1)
                
                
                summary_dict = {v[1]: pd.DataFrame() for v in target_dict.values()}
                IEI_dict = {v[1]: pd.DataFrame() for v in target_dict.values()}
                IEI_cumP_dict = {v[1]: pd.DataFrame() for v in target_dict.values()}
                duration_dict = {v[1]: pd.DataFrame() for v in target_dict.values()}
                distrib_dict = {v[1]: pd.DataFrame() for v in target_dict.values()}
                latency_dict = {v[1]: pd.DataFrame() for v in target_dict.values()}
                
                analysis_concat = {v[1]: pd.DataFrame() for v in target_dict.values()}
                
                if model == 's-e':
                    for n in range(len(start)):
                    #get data from each event window
                        event = self.df[(self.df['Time']>= start.iloc[n])&(self.df['Time']<= end.iloc[n])] #get event from start to end
                        analysis () #all result will be updated 
                        
                        for key in analysis_concat:
                            analysis_concat [key] = pd.concat([summary_dict[key], IEI_cumP_dict[key].T, IEI_dict[key].T, duration_dict[key].T, distrib_dict[key].T, latency_dict[key]], keys = ['sum', 'IEI_cumP', 'IEI', 'duration', 'distribution', 'latency'])
                            analysis_concat [key].reset_index(inplace=True)
                            analysis_concat [key].rename(columns = {'level_0': 'Measurement', 'level_1': '# trials'}, inplace = True)
                        
                        allAnalysis_concat = pd.concat([analysis_concat [key] for key in analysis_concat], keys = [key for key in analysis_concat])
                        allAnalysis_concat.reset_index (inplace = True)
                        allAnalysis_concat.drop (columns = 'level_1', inplace = True)
                       
                        
                else:
                    for n in range (1, len(end)):
                        event = self.df[(self.df['Time'] >= end.iloc[n]) & (self.df['Time'] <= start.iloc[n-1])] # Get event from previous end and start
                        analysis ()
                        
                        for key in analysis_concat:
                            analysis_concat [key] = pd.concat([summary_dict[key], IEI_cumP_dict[key].T, IEI_dict[key].T, duration_dict[key].T, distrib_dict[key].T, latency_dict[key]], keys = ['sum', 'IEI_cumP', 'IEI', 'duration', 'distribution', 'latency'])
                            analysis_concat [key].reset_index(inplace=True)
                            analysis_concat [key].rename(columns = {'level_0': 'Measurement', 'level_1': '# trials'}, inplace = True)
                                            
                        allAnalysis_concat = pd.concat([analysis_concat [key] for key in analysis_concat], keys = [key for key in analysis_concat])
                        allAnalysis_concat.reset_index (inplace = True)
                        allAnalysis_concat.drop (columns = 'level_1', inplace = True)
                        
                        
                return allAnalysis_concat 
    
    
    def __counting (self, event_df, target, label, IEI_timeRange = (0, 900, 1), cal_IEI = False, cal_latency = False, cal_duration = False, cal_inputDistribution = False, num_bins = 10, **kwards):
        """
        This is a general counting method for an entire session or a given trial
        """
        if len(event_df) == 0:
            #if empty then pass without analysis
            pass
        else:
            target_state, target_event = target
            
            T_start, T_end, T_bin = IEI_timeRange 
            
            get_kwards = lambda x: ("","") if x == None else x
            inputon, inputoff = get_kwards(kwards.get('InputOnOff'))
            
            count = 0
            latency = None
            input_IEI = pd.DataFrame()
            IEI_cumP = pd.DataFrame()
            input_duration = pd.DataFrame()
            input_distribution = pd.DataFrame()
            
            #get the count from each event window
            event_count = event_df.groupby(['States','Events']).count()
            event_count.reset_index(inplace=True)
            if target_event in event_count['Events'].values:
                count = event_count.loc[(event_count['States'] == target_state) & (event_count['Events'] == target_event), 'Time'].values[0]
            else:
                count = 0
            
            if cal_IEI:
                #calculate inter-event interval
                input_IEI = pd.DataFrame({label: self.__get_interval (event_df, mode = 's-e', start = inputon, end = inputoff)})
                #calculate inter-event interval cumulative probability
                if np.ma.count(input_IEI) == 0:
                    pass
                else:
                    IEI_cumP = pd.DataFrame (np.arange(T_start,T_end,T_bin), columns = [label+'_Time'])
                    print ('Here')
                    IEI_cumP[label] = IEI_cumP[label+'_Time'].apply(lambda x: np.ma.count(input_IEI[input_IEI.values <= x])/np.ma.count(input_IEI))                
                    IEI_cumP.drop(columns = label+'_Time', inplace = True)
            
            if cal_duration:    
                #caculate input duration
                input_duration = pd.DataFrame({label: self.__get_interval (event_df, mode = 'e-s', start = inputon, end = inputoff)})
                
            if cal_latency:    
                #caculate latency for any given input
                if inputon in event_df['Events'].values:
                    input_start = event_df[(event_df['Events']== inputon)]['Time'].values[0]
                    trial_start = event_df ['Time'].values[0]
                    latency = input_start - trial_start 
                else:
                    latency = None
            
            if cal_inputDistribution:
                input_distribution = self.__get_InputDistribution (event_df, inputon, label, num_bins)
                         
            return {'sum': count,'IEI_cumP': IEI_cumP, 'IEI': input_IEI, 'duration': input_duration, 'latency':latency, 'Distrib':input_distribution}
    
                
    def __get_interval (self, event_df, mode = 's-s',**target):
        """
        This function take event dataframe and calculate an given event intervals
        This function is dependent on event_count
        """
        if len(event_df) == 0:
            pass

        else:
            s_event = target.get('start')
            e_event = target.get('end')
            
            timeMakeup = lambda s, e, event_df: (s, e) if len(s) == len(e) else (s, np.append(e, event_df['Time'].values[-1])) if len(s) > len(e) else (np.append(event_df['Time'].values[0], s),e)
            
            start = event_df[(event_df['Events']== s_event) ]['Time']
            end = event_df[(event_df['Events']== e_event) ]['Time']
            if mode == 's-s': #interval start to start
                interval_list = start[1:].values - start[:-1].values
            if mode == 'e-e': #interval end to end
                interval_list = end[1:].values - end[:-1].values
            if mode == 'e-s': #duration
                start_array, end_array = timeMakeup (start.values, end.values, event_df)
                interval_list = end_array - start_array
            if mode == 's-e': #Interal event interal (IEI)
                start_array, end_array = timeMakeup (start.values, end.values, event_df)
                interval_list = start_array[1:] - end_array[:-1]
                
            return interval_list
    
      
    def __get_InputDistribution (self, event_df, inputon, label, num_bins = 10):
        """
        Calculate dirstibution profile for the a given event
        
        return a pandas series
        
        """
        df = event_df.copy()
        
        if len(df) == 0:
            pass
        else:
        
            duration = event_df['Time'].values[-1]
            bins = [n*duration/num_bins for n in range(0,num_bins+1)]
            
            distrb_data =[len(df[(df['Time'] >= bins[n-1])&(df['Time'] <= bins[n])&(df['Events'] == inputon)]) for n in range(1,len(bins))]
        
            distrbDf  = pd.Series (data = distrb_data, index = [n for n in range(1,len(bins))], name = label)
            
            return distrbDf
    
    
    def __get_cumulative (self, event_df, inputon, inputoff, num_bins = 10):
        """
        Calculate cumulative response for the a given event
        
        return a pandas series
        """
        
        input_mark = lambda x, inputon, inputoff: 1 if x == inputon else -1 if x == inputoff else 0
        
        if len(event_df) == 0:
            pass
        else:
            df = event_df.copy()
            df[inputon] = df['Events'].apply(lambda x: input_mark(x, inputon, inputoff))
            start_cum = df[df['Events']== inputon][['Time',inputon]] #defaults is input state
            start_cum[self.subject+'_cum'] = start_cum[inputon].cumsum()
            #start_cum[self.subject+'_cumP'] = 100*start_cum[self.subject+'_cum']/start_cum[inputon].sum()
            start_cum.reset_index(inplace=True)
            start_cum.drop(columns = inputon,inplace = True)
            start_cum.drop(columns = 'index',inplace = True)
            return start_cum
        

        
        
        
        
        
        
        
    
    