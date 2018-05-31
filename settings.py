# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 16:43:10 2017

@author: KHZS7716
"""
import os
import re
import sys
import time

version = 'G0R3'
verbose = False
debug = True
runtime = time.strftime("%Y%m%d%H%M%S")

if debug :
    root_path = '.'
    resultpath = 'results'
    out = sys.stdout
else :
    root_path = os.environ['rootpath']
    root_path = re.sub(r"\\","/",root_path)
    root_path = re.sub(r"\"","",root_path)
    resultpath = '/'.join((root_path,'results'))
    out = open('/'.join((resultpath,'run_'+runtime+'.log')),'w')

def init():
    global runtime
    global out
    global debug


