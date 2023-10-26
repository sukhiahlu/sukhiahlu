# Author: Sukhwinder Ahluwalia
# Purpose: Execuite a trade at a certain specified price, volume etc.
# Date: 3rd July 2021

import pyautogui
import pandas as pd

def DK(df,Curr,ty,f):

    #ALL DK

    #Get to order page
    pyautogui.hotkey('f9')

    pyautogui.PAUSE = .1

    #Currency
    pyautogui.hotkey('ctrl','a')
    pyautogui.typewrite(Curr)
    pyautogui.hotkey('enter')
    pyautogui.hotkey('tab')
    #Pending order
    pyautogui.hotkey('down')
    pyautogui.hotkey('tab')

    #Type of play
    #If fade
    if f == 1:
        if ty == 2: #Buy limit is a skipper
            #Sell limit (one s)            
            pyautogui.hotkey('s')

    #If normal
    elif f == 2:
        if ty == 1:
            #Buy stop (one b)
            pyautogui.hotkey('b')
        else:
            #Sell stop (two s)
            pyautogui.hotkey('s')
            pyautogui.hotkey('s')
            
    #If BSL/SSL
    elif f == 3:
        if ty == 1:
            #Buy stop limit (two b)
            pyautogui.hotkey('b')
            pyautogui.hotkey('b')
        else:
            #Sell stop limit (three s)
            pyautogui.hotkey('s')
            pyautogui.hotkey('s')
            pyautogui.hotkey('s')
            
    pyautogui.hotkey('tab')
    #Volume (1 as always for now)
    pyautogui.hotkey('tab')
    #Price
    pyautogui.typewrite(str(df['DK'][0]))
    pyautogui.hotkey('tab')
    #If BSL/SSL, Limit Price
    if f == 3:
        pyautogui.typewrite(str(df['DK_lp'][0]))
        pyautogui.hotkey('tab')

##Notes for price checks (but)
##        Curr Price < Price
##        Limit < Price
##        Stop Loss < Limit

    #Stop Loss
    pyautogui.typewrite(str(df['DK_sl'][0]))
    pyautogui.hotkey('tab')
    #TP
    pyautogui.typewrite(str(df['DK_tp'][0]))
    pyautogui.hotkey('tab')
    #Closure (Today)
    pyautogui.hotkey('down')
    pyautogui.hotkey('tab')
    #Place order
    pyautogui.PAUSE = .5
    pyautogui.click(600,420)
