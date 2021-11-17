#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 15:36:44 2018

@author: jiahuei
"""
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

'''
for d in tqdm(sorted(dirs)):
    if os.path.isfile(d): continue
    model_files = []
    compact_files = []
    for f in sorted(os.listdir(d)):
        if 'model_compact-' in f:
            compact_files.append(pjoin(d, f))
        elif 'model-' in f:
            model_files.append(pjoin(d, f))
        else:
            pass
    compact_files.sort(key=natural_keys)
    model_files.sort(key=natural_keys)
    for f in compact_files[:-3]:
        os.remove(f)
    for f in model_files[:-3]:
        os.remove(f)
   ''' 



