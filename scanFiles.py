# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 23:32:37 2018

@author: cyfho
"""

from __future__ import division, print_function


__author__ = "Yifen Cheng"
__version__ = "1.2.0"
__license__ = "NA"


try:
    import os
except ImportError:
    print('pandas are not available. Run \'pip install os\' in the terminal')

try:
    import re
except ImportError:
    print('pandas are not available. Run \'pip install re\' in the terminal')
    
try:
    import numpy as np
except ImportError:
    print('pandas are not available. Run \'pip install re\' in the terminal')


class scanFiles:
    
    def __init__ (self, rootdir, *extension):
        self.rootdir = rootdir
        self.ext = tuple(extension)
        self.get(display=False)
        
    def get (self, date_range = (0,99999999), display = True):
        #ext = tuple (extension)
        allFile_l = []
        if len(date_range) == 2:
            start, end = date_range
        else:
            start = date_range[0]
            end = start
        
        for subdir, dirs, files in os.walk(self.rootdir):
            for file in files:
                if file.endswith(self.ext):
                    if subdir.split('\\')[-1] >= str(start) and subdir.split('\\')[-1] <= str(end):
                        allFile_l.append(os.path.join(subdir,file))
        if display:
            print ('Get %s files'%len(allFile_l))
        else:
            print ('Found %s %s files'%(len(allFile_l),self.ext))
        self.allFile_l = allFile_l
        return allFile_l
        
    def head(self, n = 5):
        #num_files = len(self.allFile_l)
        count= 0
        for num, f in enumerate (self.allFile_l, 1):
            print (num,":",re.split('\\\\',f)[-2]+'\\'+re.split('\\\\',f)[-1])
            count += 1
            if count == n:
                break
