# Author: Sukhwinder Ahluwalia
# Purpose: To download data and run code to see if a trade is a good idea
# Date: 8 November 2021

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn' for the Order one
#pd.set_option('display.max_colwidth', -1)

import numpy as np #Numbers (not used)
import plotly.graph_objs as go #Charts
import glob as g #For filepath saving

import Runs as r #Run code

import DK_play as dk #DK play automate

import winsound #For sound

import os ##Filepath creation

import datetime 
from datetime import date

import Currency_downloader as d #Currencies in PPstone
d.pyautogui.hotkey('alt', 'tab')
import AM002 as cd #the Daily plays

##if (date.today().weekday() > 0) & (date.today().weekday() <= 5):
##        import Currency_downloader as d #Currencies in PPstone
##        d.pyautogui.hotkey('alt', 'tab')
##
##if (date.today().weekday() >= 0) & (date.today().weekday() < 5):
##        import AM002 as cd #the Daily plays
y##
##elif (date.today().weekday() == 6):
##        import Checker_weekly as cd #the Weekly plays
