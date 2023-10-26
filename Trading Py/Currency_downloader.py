# Author: Sukhwinder Ahluwalia
# Purpose: Automating data download from a trading data download source (Pepperstone terminal)
# Date: 3rd September 2020

#! python3
# mouseNow.py - Displays the mouse cursor's current position.
import pyautogui
import time
#Mouse coordinates. = x = (0,1919) y = (0,1079)

#Get to Pepperstone app
pyautogui.click(525, 900)
#pyautogui.hotkey('alt', 'tab')
#pyautogui.PAUSE = 2

#Get to currency page
pyautogui.hotkey('ctrl', 'u')

#Get to Bars
pyautogui.click(500, 235)
pyautogui.PAUSE = 1
#Download AUDUSD
pyautogui.typewrite('audusd')
pyautogui.hotkey('enter')
pyautogui.hotkey('tab')
pyautogui.typewrite('d') #Daily
pyautogui.click(1000, 265) #Request
pyautogui.click(480, 640) #Export
##time.sleep(2)
pyautogui.click(700, 230) #Change the directory for these
pyautogui.typewrite("C:\\Users\\nitis\\Documents\\Forex\\Data\\New data") #Use directory
pyautogui.hotkey('enter')
pyautogui.click(700, 550) #Change the name
pyautogui.typewrite('audusd') #Save file as aud
pyautogui.hotkey('enter')
pyautogui.typewrite('y')


#Download other curencies
#Currency runs
Curr = ['EURAUD','EURGBP','EURUSD','GBPUSD','GBPAUD','USDCAD','USDJPY','US30','AUDCAD','USDCHF',
        'AUS200','US500','UK100','NAS100','EUSTX50','SPA35','JPN225','USDX','EURX','XAUUSD']

#Loop through all currencies, len(daily)
for i in range(0,len(Curr)):
    pyautogui.click(490, 265) #Currency type
    pyautogui.typewrite(Curr[i])
    pyautogui.hotkey('enter')
    pyautogui.click(1000, 265) #Request
    pyautogui.click(480, 640) #Export
##    time.sleep(2)
    pyautogui.typewrite(Curr[i]) #Save file 
    pyautogui.hotkey('enter')
    pyautogui.typewrite('y')

#Exit currency page
pyautogui.hotkey('esc')
