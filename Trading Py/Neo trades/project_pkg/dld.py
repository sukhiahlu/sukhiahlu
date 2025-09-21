#Download the data from PPT terminal
#ie the Forex and Metals tickers
# Import the libraries
import pyautogui
import time
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import os
import itertools

from datetime import date

pd.set_option('display.max_columns', None)
def dld(Curr):
    #! python3
    # mouseNow.py - Displays the mouse cursor's current position.
    #Mouse coordinates. = x = (0,1919) y = (0,1079)

    pyautogui.PAUSE = 4

    #Get to Pepperstone app
    pyautogui.click(25, 900)
    # pyautogui.PAUSE = 1
    pyautogui.click(350, 200)
    #pyautogui.hotkey('alt', 'tab')
    pyautogui.PAUSE = 2

    #Get to currency page
    pyautogui.hotkey('ctrl', 'u')

    #Get to Bars
    pyautogui.click(500, 235)
    #Download AUDUSD
    pyautogui.typewrite("audusd")
    pyautogui.hotkey('enter')
    pyautogui.hotkey('tab')
    pyautogui.typewrite('d') #Daily
    pyautogui.click(1000, 265) #Request
    pyautogui.click(480, 640) #Export
    #time.sleep(2)
    pyautogui.click(700, 215) #Change the directory for these
    pyautogui.typewrite("C:\\Users\\nitis\\Documents\\Forex\\Data\\New data") #Use directory
    pyautogui.hotkey('enter')
    pyautogui.click(700, 550) #Change the name
    pyautogui.typewrite('audusd.csv') #Save file as aud
    pyautogui.hotkey('enter')
    pyautogui.typewrite('y')

    #Loop through all currencies, len(daily)
    for i in range(0,len(Curr)):
        pyautogui.click(490, 265) #Currency type
        pyautogui.typewrite(Curr[i])
        pyautogui.hotkey('enter')
        pyautogui.click(1000, 265) #Request
        pyautogui.click(1000, 265) #Request again
        pyautogui.click(480, 640) #Export
    #    time.sleep(2)
        nm = fr"{Curr[i]}.csv" #Name of file
        pyautogui.typewrite(nm) #Save file 
        pyautogui.hotkey('enter')
        pyautogui.typewrite('y')

    #Exit currency page
    pyautogui.hotkey('esc')
    
# #Idiom to 
# Keeps the module importable without side effects.
# Lets you run the file directly for debugging or as a script.
if __name__ == "__main__":
    dld()