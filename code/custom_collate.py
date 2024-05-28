# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:22:04 2024

@author: DAM1
"""

def custom_collate(data): # modify to generate confidence map and append and then train faster-rCNN

# this just allows you to return the datasets as is. 
# because we won't always have the same number of bounding boxes..
    return data # 