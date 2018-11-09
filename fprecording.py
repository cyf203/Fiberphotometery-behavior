# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 18:02:34 2018

@author: cyfho
"""
try:
    import re
except ImportError:
    print ('re are not available. Run \'pip install re\' in the terminal')

try:
    import os
except ImportError:
    print ('os are not available. Run \'pip install os\' in the terminal')

try:
    from scipy.optimize import curve_fit
except ImportError:
    print('scipy are not available. Run \'pip install scipy\' in the terminal')

try:
    from tqdm import tqdm #for progress bar in terminal
    tqdm.pandas()  #provide progress bar
except ImportError:
    print('tqdm are not available. Run \'pip install tqdm\' in the terminal')    
    
try:
    import pandas as pd    #for data loading and manipulation
except ImportError:
    print('pandas are not available. Run \'pip install pandas\' in the terminal')   

try:
    import numpy as np    #for data loading and manipulation
except ImportError:
    print('numpy are not available. Run \'pip install numpy\' in the terminal')  

try:
    import matplotlib as mpl
except ImportError:
    print ('matplotlib is not available. Run \'pip install matplotlib\' in the terminal')

try:
    import matplotlib.pyplot as plt   #for ploting
except ImportError:
    print ('matplotlib is not available. Run \'pip install matplotlib\' in the terminal')


class fprecording:
    def __init__ (self, datafile, ref_file, unmixing_file = None, auto_unmixing = True, plot_unmixing = True, **kwards):    
        self.f = datafile
        self.ref_file = ref_file
        self.unmixing_file = unmixing_file
        
        #get all kwards
        get_kwards = lambda x, y: y if x == None else x
        self.c1 = get_kwards(kwards.get('c1'), 'Dual-Gs')
        self.c2 = get_kwards(kwards.get('c2'), 'Dual-Td')
        self.trim_range = get_kwards(kwards.get('data_range'), [450, 750])
        self.G_range = get_kwards(kwards.get('G_range'), [500, 540])
        self.R_range = get_kwards (kwards.get('R_range'), [575, 650])
        self.resolving_range = get_kwards (kwards.get('resolving_range'), [500, 749.3])
        self.rec_duration = get_kwards (kwards.get('rec_duration'), None)
    
        self.lw = get_kwards (kwards.get('lw'), 1)
        self.titlesize = get_kwards (kwards.get('titlesize'), 12)
        self.titlepad = get_kwards (kwards.get('titlepad'), 3)
        self.labelsize = get_kwards (kwards.get('labelsize'), 10)
        self.labelpad = get_kwards (kwards.get('labelpad'), 10)
        self.dpi = get_kwards (kwards.get('dpi'), 150)
        self.autolayout = get_kwards (kwards.get('autolayout'), True)
        
        #Initialize all attribute
        self.rec_len = None
        self.rec_window = None
        self.raw_G, self.raw_R = None, None
        self.unmixing_result = pd.DataFrame()  #initialize an empty dataframe 
        self.ratio = None
        
        #Set up figure format
        plt.style.use('ggplot')  #plot style
        self._setupFigure()
        
        self._get_ref()
        self.rawdata_reduce = self._get_data()
        
        if self.unmixing_file == None:
            if auto_unmixing:
                self._unmixing_data()
                if plot_unmixing:
                    self.plot_unmixingResult()
            else:
                print ('Unmixing method not applied')
        else:
            self.load_unmixingResult(self.unmixing_file)
            print ('Loaded exisiting unmixing file: %s'%self.unmixing_file)
            
    def _get_data (self):
        """Extract data from raw .ascii or .txt file with plot or export into Excel option 
     
        Returns
        -------
        1. trimmed dataframe (df)
        """
        self.filename = re.split('_Sub|\\\\',self.f)[-2]
        #Find the data start line number 
        count = 0
        rec_resolution = 0
        data_line = 0
        with open(self.f) as myFile:
            for num, line in enumerate(myFile, 1):
                temp = re.split("\t| |\n",line)
                count+=1
                if 'Integration Time' in line:
                    acqt = float(temp[3])*1000
                if 'Number of Pixels' in line:
                    rec_resolution = int(temp[5])
                    print ("The recording resolution is %s nm wavelength"%rec_resolution)
                if 'Begin Spectral Data' in line:
                    data_line = num
                    break
        
        #open text file to csv
        try:
            df = pd.read_csv (self.f, delim_whitespace=True,header=data_line-1) 
        except:
            df = pd.read_csv (self.f, delim_whitespace=True,header=data_line) 
        df.reset_index(inplace=True)
        df.rename(columns={'level_0':'Date','level_1':'time','level_2':'time_2'},inplace=True)

        #check if rec_len param passed in
        if self.rec_duration == None:
            #Ocean view version 1.5.x and 1.6.x has different datafile header and column index
            #handle that
            try:
                df_time = df['time_2']  
            except:
                df_time = df['time']
            #Check average recording frequency and recording length
            diff = 0    
            for n in range(1,len(df_time)):
                diff += df_time[n] - df_time[n-1]
            try:
                acqt = diff/(len(df_time)-1)
            except:
                pass
            rec_len = (len(df_time)-1)*acqt/1000
        else:
            rec_len = self._getSec(str(self.rec_duration))
            acqt = rec_len/len(df)*1000.00
        
        print ('''This recording length is ~%s (s);\nThis recording frequency is ~%s Hz;\nOne recording frame is ~%s (ms)
        '''%(rec_len,(1000/acqt),acqt))
        
        #format the dateframe
        df.insert(3,'time(s)',df.index)
        #apply a new time scale
        df['time(s)'] = df['time(s)'].apply(lambda x: round(x*acqt/1000,3))
        df.set_index('time(s)',inplace=True)
        try:
            df.drop(columns={'Date','time','time_2'},inplace=True)
        except:
            df.drop(columns={'Date','time'},inplace=True)
            
        df_wave = df.T.copy()
        df_trim, signal = self._get_GT_signal (df_wave, self.trim_range, self.G_range, self.R_range)   # df_trim index is wavelength, column is time
        
        self.rec_len = rec_len
        self.rec_window = acqt
        self.raw_G, self.raw_R = signal
        self.rawdata = df_trim
        
        return df_trim.T # transpose df_tirm--> index is time, column is wavelength
    
    def _get_GT_signal (self, df, trim_range, GFP_range, tdt_range):
        '''
        Depends on the reocording type and filter applied status
        '''
        df.index = df.index.astype(float)
        
        trim_min, trim_max = trim_range
        df_trim  = df[(df.index>=trim_min)& (df.index<=trim_max)].copy()  # index is wavelength, column is time
        
        GFP_min, GFP_max = GFP_range
        tdt_min, tdt_max = tdt_range
        
        GFP = df[(df.index >=GFP_min) & (df.index <=GFP_max)].copy()  
        tdt = df[(df.index >=tdt_min) & (df.index <=tdt_max)].copy()  
     
        GFP.loc['sum']=GFP.sum()
        tdt.loc['sum']=tdt.sum()
    
        GFP_signal = GFP.loc['sum'].copy()
        tdt_signal = tdt.loc['sum'].copy()
        
        GFP_signal.rename('GFP_raw', inplace=True)            # index is time, column is intensity
        tdt_signal.rename('tdt_raw', inplace=True)
        
        signal = (GFP_signal,tdt_signal)
        
        return df_trim, signal
    
    def _getSec(self, s) -> float:
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
        #if contain ms component
        if type(s) == float:
            return round(s,3)
        elif type(s) == str:
            if '.' in s:  
                l = list(map(float, re.split(':|\\.',s))) 
                return round(sum(n * sec for n, sec in zip(l[::-1], (0.001, 1, 60, 3600))),3)
            else:
                l = list(map(float, re.split(':',s))) 
                return round(sum(n * sec for n, sec in zip(l[::-1], (1, 60, 3600))),3)
    
    def _unmixing_data (self):
        self.unmixing_result['Coeff of Gs'], self.unmixing_result['Coeff of td'], \
        self.unmixing_result['Intercept'], self.unmixing_result['R-squared'], \
        self.unmixing_result['label'] = zip(*self.rawdata_reduce.progress_apply(lambda x: self.unmixing (x),axis=1)) #x stands for each row's data
        self.unmixing_result.index = self.rawdata_reduce.index
        
        self.unmixing_signal = (self.unmixing_result['Coeff of Gs'], self.unmixing_result['Coeff of td'])
        self.highest_R2 = self.unmixing_result['R-squared'].idxmax()
        self.lowest_R2 = self.unmixing_result['R-squared'].idxmin()
        print ('Unmixing method applied')
        print ('Highest R2 is %s, and lowest R2 is %s'%(self.unmixing_result['R-squared'][self.highest_R2], self.unmixing_result['R-squared'][self.lowest_R2]))
    
    def save_unmixingResult (self, path):
        #Save unmixing result
        self.unmixing_result.to_csv(os.path.join(path, '%s_unmixing.csv'%self.filename), sep=',')
        print ('Export %s unmixing data to the csv'%self.filename)
        
    def load_unmixingResult (self, path):
        self.filename = re.split('.csv|\\\\',path)[-2]
        self.unmixing_result = pd.read_csv (path, sep=',', index_col = 'time(s)', na_filter = False)

    def _get_ref (self):
        #prepare reference data
        ref_df = pd.read_csv(self.ref_file, sep=',')
        self.up_lim, self.low_lim = self.resolving_range
        ref_df.reset_index(inplace=True)
        ref_df.set_index('Wavelength (nm)',inplace=True)
        del ref_df['index']
        ref_trim = ref_df[(ref_df.index > self.up_lim) & (ref_df.index < self.low_lim)].copy()
        self.G_ref = ref_trim[self.c1].values
        self.R_ref = ref_trim[self.c2].values
        
    def unmixing (self, mix_df):
        """
        Spectrum unmixing funciton
        Parameters
        ----------
        mix_df: pandas series, from dataframe at a given time point
        
        Returns
        -------
    
        1. C1
        2. C2
        3. intercept
        4. R2
        5. label  
        """
        #prepare raw mixture data from recording
        mix_trim = mix_df[(mix_df.index > self.up_lim) & (mix_df.index < self.low_lim)].copy()
        mix_data = mix_trim.values 
        
        TSS = sum(np.array((mix_data - np.mean(mix_data))**2)) 
        
        C1, C2, intercept, R2 = 0,0,0,0
        #check if the len of data is matched
        if len(self.G_ref) == len(mix_data):
            p = self._fit_lm(self.G_ref, self.R_ref, mix_data)
            interception, C1, C2 = p
            pred = self._yn((self.G_ref,self.R_ref), *p)
            R2 = self._R2(mix_data, pred, TSS)        
            label = 'fitting: %5.3f+%5.3f*G+%5.3f*R'%tuple(p) + r'$, R^{2} = %5.3f$'%R2
            #if len(ref_trim.index) == len(mix_trim.index):
            #    fitting_df = pd.Series(pred, index = mix_trim.index)
            #else:
            #    fitting_df = None
                #print ('error: ref data recording resolution is not the same as mix date')
        else:
            print ('the recording resolution is different between reference data and mixture recording')
            
        return C1, C2, intercept, R2, label
    
    def _yn (self, X, *p):
        G,R = X
        return p[0]+p[1]*G+p[2]*R
    
    def _fit_lm (self, G, R, raw_data, p0=[1,1,1]):
        popt, _ = curve_fit(self._yn, (G,R), raw_data, p0)
        return popt
    
    def _R2 (self, y_raw, y_pred, TSS):
        return 1- (sum(np.array((y_raw - y_pred)**2))/TSS)
    
    def _get_mixdata (self, data_series):
        mix_data = data_series
        mix_trim = mix_data[(mix_data.index > self.up_lim) & (mix_data.index < self.low_lim)].copy()
        return mix_trim
        
    def plot_unmixingResult (self, n = 0):
        if len(self.unmixing_result) == 0:
            print ('Unmixing method not apply')
        else:
            p = self.unmixing_result [['Intercept','Coeff of Gs','Coeff of td']]
            
            mix_trim_n = self._get_mixdata(self.rawdata_reduce.iloc[n])
            x_n = mix_trim_n.index
            y_n = self._yn((self.G_ref,self.R_ref), *list(p.iloc[n].values))
            
            self.highest_R2 = self.unmixing_result['R-squared'].idxmax()
            self.lowest_R2 = self.unmixing_result['R-squared'].idxmin()
            
            mix_trim_highR = self._get_mixdata(self.rawdata_reduce.loc[round(self.highest_R2,3)])
            x_highR = mix_trim_highR.index
            y_highR =  self._yn((self.G_ref,self.R_ref), *list(p.loc[self.highest_R2].values))
            
            mix_trim_lowR = self._get_mixdata(self.rawdata_reduce.loc[round(self.lowest_R2,3)])
            x_lowR = mix_trim_lowR.index
            y_lowR = self._yn((self.G_ref,self.R_ref), *list(p.loc[self.lowest_R2].values))
            
            ##plot graph for raw and unmixing data
            fig1, ([ax1, ax2],[ax3, ax4],[ax5, ax6]) = plt.subplots (3,2,figsize = (10,6))  #set up canvas
            
            ax1.plot(self.rawdata_reduce.iloc[n], 'k-')   #plot the first time point sampel trace
            self._format_ax(ax1, 'Wavelength [nm]', 'Intensity [Photon count]', 'Sample recording spectrum', showLegend=False)
            
            ax2.plot(self.raw_G, 'g-', label = self.raw_G.name)  #plot raw green and red data
            ax2.plot(self.raw_R, 'r-', label = self.raw_R.name)
            self._format_ax(ax2, 'Time [s]', 'Intensity [Photon count]', 'Raw time series recording', showLegend=True)
            
            ax3.plot(self.rawdata_reduce.iloc[n],'k.', label='raw')  #plot unmixing first time point sample trace
            ax3.plot(x_n, y_n,'b-',label=r'$unmixing, R^{2} = %5.3f$'%self.unmixing_result['R-squared'][n])
            self._format_ax(ax3, 'Wavelength [nm]', 'Intensity [Photon count]', 'Unmixing fitting', showLegend=True)
            
            ax4.plot(self.unmixing_result['Coeff of Gs'], 'g-', label = self.unmixing_result['Coeff of Gs'].name)
            ax4.plot(self.unmixing_result['Coeff of td'], 'r-', label = self.unmixing_result['Coeff of td'].name)
            self._format_ax(ax4, 'Time [s]', 'Coeffiency', 'Unmixing time series recording', showLegend=True)
            
            ax5.plot(self.rawdata_reduce.loc[round(self.highest_R2,3)], 'k.', label='raw') #plot highest R2 unmixing trace
            ax5.plot(x_highR, y_highR,'b-', label='unmixing, '+r'$R{2} = %5.3f$'%self.unmixing_result['R-squared'][self.highest_R2])
            self._format_ax(ax5, 'Wavelength [nm]', 'Intensity [Photon count]', 'Unmixing fitting: Highest R2', showLegend=True)
            
            ax6.plot(self.rawdata_reduce.loc[round(self.lowest_R2,3)], 'k.', label='raw') #plot lowest R2 unmixing trace
            ax6.plot(x_lowR, y_lowR,'b-', label='unmixing, '+r'$R{2} = %5.3f$'%self.unmixing_result['R-squared'][self.lowest_R2])
            self._format_ax(ax6, 'Wavelength [nm]', 'Intensity [Photon count]', 'Unmixing fitting: Lowest R2', showLegend=True)
            
            plt.show()
            
            return fig1
            
    def event_analysis (self, event_window, events, beh_data = pd.DataFrame(), **kwards):        
        
        get_kwards = lambda x, y: y if x == None else x
        expand = get_kwards(kwards.get('expand'), 5)
        bins = get_kwards(kwards.get('bind'), 10)
        shift = get_kwards(kwards.get('shift'), 0)
        beh_target = get_kwards (kwards.get('beh_target'), '')
        beh_event_idx = get_kwards (kwards.get('beh_event_idx'), '')
        blhist_display = get_kwards (kwards.get('blhist_display'), True)
        
        ratio = self.unmixing_result['Coeff of Gs']/self.unmixing_result['Coeff of td']
        
        event_s, event_e = event_window
        expand_s, expand_e = event_s - expand, event_e + expand
        
        start = [('', expand_s,'w')]
        end = [('', event_e,'w')]
        expand_end = [('', expand_e,'w')]
        
        self.events_list = start + events + end + expand_end
        
        event_data  = ratio[(ratio.index >= expand_s) & (ratio.index <= expand_e)] #recording event data
        hist, bin_edges = np.histogram(event_data, bins = bins)
        bl_thers = bin_edges[hist.argmax()]
        bl = event_data[event_data<=bl_thers]
        self.norm_EtOH = (event_data - np.average(bl))/(np.average(bl))*100 #normalized recording event data
        
        beh_event_data = beh_data[(beh_data.index >= expand_s) & (beh_data.index <= expand_e)]
        
        if blhist_display:
            self.blhistfig, (ax1, ax2) = plt.subplots (1,2,figsize = (8,3))  #set up canvas for bl select and raw data
            ax1.plot(event_data,'k')
            ax1.plot(bl, 'b.')
            ax2.hist(event_data, bins= bins)
            self._format_ax(ax1, 'Time [s]', 'CoG/CoT', set_xtick=False, set_ytick = False, showLegend= False)
            self._format_ax(ax2, 'CoG/CoT', 'Num of data', set_xtick=False, set_ytick = False, set_ylim = False, showLegend= False)
        
        self.eventAnalysis_fig = plt.figure (figsize = (10,6)) #set up canvas for analysis
        if len(beh_data) == 0:
            gs = mpl.gridspec.GridSpec (3, 1, height_ratios = [3,1,1])
            ax1, ax2, ax3 = plt.subplot(gs[0]),plt.subplot(gs[1]),plt.subplot(gs[2])
        else:
            gs = mpl.gridspec.GridSpec (4, 1, height_ratios = [3, 1, 1, 1])
            ax1, ax2, ax3, ax4 = plt.subplot(gs[0]), plt.subplot(gs[2]), plt.subplot(gs[3]), plt.subplot(gs[1])
            ax4.plot(beh_event_data[beh_target],'r|', lw= 0.5)  # add behavior event time stamp
        
        ax1.plot(self.norm_EtOH,'b')
        xmin, xmax = ax1.get_xlim()
        xlen = xmax-xmin
        self.event_auc = pd.DataFrame(columns=['event','AUC','AUC/t'])
        
        for n in range(1, len(self.events_list)):
            xloc = ((self.events_list[n-1][1]+self.events_list[n][1]-shift)/2 - xmin)/xlen
            self._auc_result (ax1, ax2, ax3, n, self.event_auc, self.events_list, self.norm_EtOH, xloc, offset = shift, bar_color = self.events_list[n-1][2])
            
        self._format_ax(ax1, 'Time [s]', 'CoG/CoT', '%s Drinking event-%s [%s s]'%(self.filename,beh_event_idx, round(event_e-event_s,2)), set_xtick=False, set_ytick = False, showLegend= False)
        
        ax2.set_xlim(0,1)
        ax2.tick_params(axis='x',which='both',bottom=False, top=False, labelbottom=False)
        ax2.set_ylabel('AUC')
        
        ax3.set_xlim(0,1)
        ax3.tick_params(axis='x',which='both',bottom=False, top=False, labelbottom=False)
        ax3.set_ylabel('AUC/second')
        
        if len(beh_data) != 0:
            ax4.set_ylim(0.99,1.01)
            ax4.set_xlim(xmin, xmax) #get the same time scale as recording data
            ax4.tick_params(axis='y', which = 'both', left = False, right = False, labelleft = False )
            ax4.set_ylabel (beh_target, rotation = 90)
        
        plt.show()
        
        return self.event_auc
    
    def save_eventAnalysisResult (self, path):
        self.event_auc.to_csv(os.path.join(path, '%s_auc.csv'%self.filename), sep=',')
        print ('Export %s unmixing data to the csv'%self.filename)
    
    def get_RecEventSample (self, col_num, thershold):
        sample_trace = self.norm_EtOH.to_frame()
        sample_trace.reset_index(inplace = True)
        sample_trace.rename(columns = {0:col_num+1}, inplace = True)
        sample_trace ['Beh_timestamp'] = sample_trace['time(s)'].apply(lambda x: self._getBehtimeStamp(x, thershold))
        return sample_trace
        
    def _getBehtimeStamp (self, time, thershold):
        for event in self.events_list:
            if abs(event[1] - time) < thershold:
                return event[0]
        
    def _auc_result (self, ax1, ax2, ax3, n, event_auc, action, norm_EtOH, xloc, offset = 0, bar_width = 0.02, bar_color ='r', transparent = 0.5):
        ax1.axvspan (action[n-1][1]-offset, action[n][1]-offset, alpha = 0.3, color = bar_color)
        ax1.text(xloc, 0.95, action[n-1][0], ha='center', va='center', transform=ax1.transAxes)
        event_auc.at[n-1, 'event'] = action[n-1][0]
        event_auc.at[n-1, 'AUC'] = np.trapz(norm_EtOH[(norm_EtOH.index> (action[n-1][1]-offset))& (norm_EtOH.index < (action[n][1]-offset))])
        event_auc.at[n-1, 'AUC/t'] = np.trapz(norm_EtOH[(norm_EtOH.index> (action[n-1][1]-offset))& (norm_EtOH.index < (action[n][1]-offset))])/(action[n][1]-action[n-1][1])
        ax2.bar(xloc, event_auc['AUC'][n-1], bar_width, color = bar_color, alpha = transparent)
        ax3.bar(xloc, event_auc['AUC/t'][n-1], bar_width, color = bar_color, alpha = transparent)
    
    def _format_ax (self, ax, xlabel=None, ylabel=None, figuretitle=None, set_xtick = True, set_ytick = True, set_ylim = True, showLegend=True):
        ax.set_title (figuretitle)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ymax = ymax+0.2*(ymax-ymin)
        ax.set_xlim(xmin,xmax)
        if set_ylim:
            ax.set_ylim(-(ymax/2),ymax)
        if set_xtick:
            ax.set_xticks([xmin,(xmax-xmin)/3+xmin,(xmax-xmin)/3*2+xmin,xmax])
        if set_ytick:
            ax.set_yticks([0,ymax/3,ymax/3*2,ymax])
        if showLegend:
            ax.legend()
    
    def _setupFigure(self):
        #### set up ploting parameters
        #print (plt.style.available)
        mpl.rcParams['lines.linewidth'] = self.lw
        mpl.rcParams['axes.spines.top'] = False
        mpl.rcParams['axes.spines.right'] = False
        mpl.rcParams['axes.titlesize'] = self.titlesize
        mpl.rcParams['axes.titlepad'] = self.titlepad
        mpl.rcParams['axes.labelsize'] = self.labelsize
        mpl.rcParams['axes.labelpad'] = self.labelpad
        mpl.rcParams['xtick.major.width'] = self.lw
        mpl.rcParams['xtick.labelsize'] = self.labelsize
        mpl.rcParams['ytick.major.width'] = self.lw
        mpl.rcParams['ytick.labelsize'] = self.labelsize
        mpl.rcParams['legend.frameon'] = False
        mpl.rcParams['figure.dpi'] = self.dpi
        mpl.rcParams['figure.autolayout'] = self.autolayout
