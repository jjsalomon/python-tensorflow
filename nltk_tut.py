# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 23:17:53 2017

@author: azkei
"""

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines= 10000000
