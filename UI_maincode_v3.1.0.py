
# coding: utf-8

import sys
sys.path.append(r'D:\Python Projects\Fiber photometry analysis project\Functions')
sys.path.append(r'D:\Python Projects\Behavior analysis project\Functions')
import re  #for word processing
import pandas as pd    #for data loading and manipulation
from fprecording import fprecording #import customized class for analysis fiber photometry signal
from PyQt5 import QtWidgets, uic  #QtCore, QtGui, 

#from detect_peaks import detect_peaks
qtCreatorFile = r'D:\Python Projects\Fiber photometry analysis app project\fiber photometry analysis app_3.1.1.ui'

Ui_MainWindow, QtBaseClass = uic.loadUiType (qtCreatorFile)

class MyApp (QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        
        self.load_datafile.clicked.connect(self.loadfile)   
        self.load_ref_csv.clicked.connect(self.load_ref)
        self.analysis.clicked.connect(self.getdata)
        self.save.clicked.connect(self.save_data)
        

    def loadfile (self):
        options = QtWidgets.QFileDialog.Options()
        #options = QtWidgets.QFileDialog.DontUseNativeDialog
        self.datafile, _ = QtWidgets.QFileDialog.getOpenFileName(self,"Load data file", "","Text Files (*.txt);;All Files (*)", options=options)
        if self.datafile == "":
            pass
        else:
            self.Output.append ('Loaded data file: \n%s'%re.split('/',self.datafile)[-1])
            
        
    
    def load_ref (self):
        options = QtWidgets.QFileDialog.Options()
        #options = QtWidgets.QFileDialog.DontUseNativeDialog
        self.reference, _ = QtWidgets.QFileDialog.getOpenFileName(self,"Load reference csv file", "","csv Files (*.csv);;All Files (*)", options=options)
        if self.reference == "":
            pass
        else:
            self.Output.append ('Loaded data file: \n%s'%re.split('/',self.reference)[-1])
                 
    
    def getdata(self, plot_feature = True):
        #get input from UI
        Green_range = [int(self.G_start.text()),int(self.G_end.text())]
        red_range = [int(self.R_start.text()),int(self.R_end.text())]
        
        self.rat_rec = fprecording (datafile = self.datafile, ref_file = self.reference, 
                                    auto_unmixing = self.unmixing_check.isChecked(),
                                    plot_unmixing = self.Plotting_check.isChecked(),
                                    G_range = Green_range, R_range = red_range)
        
        self.Output.append ('\nExtracting Completed!')
        
        recording_properties = "\nThis recording length is ~%s (s);\nEach recording frame is ~%s (ms);\nRecording frequency is ~%s (Hz)"%(self.rat_rec.rec_len, self.rat_rec.rec_window, 1000/self.rat_rec.rec_window)
                
        self.Output.append (recording_properties)

    
    def save_data (self):
        #get save options from UI before select save file path
        save_rawdata = self.save_rawdata_check.isChecked()
        save_green = self.save_rawgreen_check.isChecked()
        save_red = self.save_rawred_chcek.isChecked()
        save_unmixing = self.save_unmixing_check.isChecked() and self.unmixing_check.isChecked()
        
        #get save file name and path
        options = QtWidgets.QFileDialog.Options()
        #options = QtWidgets.QFileDialog.DontUseNativeDialog
        savefile, _ = QtWidgets.QFileDialog.getSaveFileName(self,"Save", "","Excel Files (*.xlsx);;All Files (*)", options=options)
        
        #start to write 
        with pd.ExcelWriter (savefile) as writer:
            self.Output.append ('\nStart writing...')
            
            if save_rawdata:
                self.rat_rec.rawdata.to_excel (writer, sheet_name = 'raw data')
                
            
            if save_green:
                self.rat_rec.raw_G.to_excel (writer, sheet_name = 'Green data')
            
            
            if save_red:
                self.rat_rec.raw_R.to_excel (writer, sheet_name = 'Red data')
        
            if save_unmixing:
                self.rat_rec.unmixing_result.to_excel (writer, sheet_name = 'Unmixing')

        self.Output.append('Exported data to Excel!')



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_()) 



