# Author: Sukhwinder Ahluwalia
# Purpose: To download data and run code to see if a trade is a good idea - the high level for the other scripts
# Date: 3rd September 2021

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn' for the Order one
#pd.set_option('display.max_colwidth', -1)

import numpy as np #Numbers (not used)
import plotly.graph_objs as go #Charts
import glob as g #For filepath saving

import Runs_simple as r #Run code

import DK_play as dk #DK play automate

import winsound #For sound

import os ##Filepath creation

import datetime 
from datetime import date

#Currency runs
Curr = ['AUDCAD','AUDUSD','AUS200','EURAUD','EURGBP','EURUSD','EURX','EUSTX50','GBPAUD','GBPUSD',
        'JPN225','NAS100','SPA35','UK100','US30','US500','USDCAD','USDCHF','USDJPY','USDX','XAUUSD',]

#Checker begins
#Get data file names for Daily (Prev, Step 1)
path = r'C:\Users\nitis\Documents\Forex\Data'
daily = g.glob(path + "/*.csv")

#Get data file names for Daily (New, Step 2)
path2 = r'C:\Users\nitis\Documents\Forex\Data\New data'
daily2 = g.glob(path2 + "/*.csv")

#Empty dataframe to save stuff
chts = list()

#Loop through all currencies, len(daily)
for i in range(0,len(daily)):

        #Empty dataframe to save stuff
        conc = list()

        #Step 1. Load all prev data
        dl = pd.read_csv(daily[i],sep='\t')
        #dl = dl.loc[:, ~dl.columns.str.contains('^Unnamed')]
        conc.append(dl)

        #Step 2. Load new down. data
        nl = pd.read_csv(daily2[i],sep='\t')
        if (date.today().weekday() > 0) & (date.today().weekday() <= 4):
                nl = nl[:-1] #Remove last row which would be the day of the extract

        #Check if Jan 4 2021 data matches (so that incorrect data not loaded)
        d = dl.loc[dl['<DATE>'] == '2021.06.04']
        n = nl.loc[nl['<DATE>'] == '2021.06.04']
        d.reset_index(drop=True, inplace=True)
        n.reset_index(drop=True, inplace=True)
        
        if n['<CLOSE>'][0] <= 2: #For (most) currencies
                if round(n['<CLOSE>'][0],5) != round(d['<CLOSE>'][0],5):
                        input('Wrong entry')
                        print(n['<CLOSE>'][0])
                        print(d['<CLOSE>'][0])
                        print(Curr[i])
                        break
        else:
                if int(n['<CLOSE>'][0]) != int(d['<CLOSE>'][0]):
                        input('Wrong entry')
                        print(n['<CLOSE>'][0])
                        print(d['<CLOSE>'][0])
                        print(Curr[i])
                        break           

        conc.append(nl)

        #Step 3. Concactenate all data for all currencies
        conc = pd.concat(conc)
        conc.drop_duplicates(subset='<DATE>', keep='first', inplace=True)

        #Resetting index for concacentated
        conc.reset_index(drop=True, inplace=True)

        #Step 4. Save final concatenated data into a csvs to load next time
        s = pd.Series([Curr[i],'.csv'])
        output_file = os.path.join(path,s.str.cat(sep=''))
        conc.to_csv(output_file,sep='\t',index=False)

        #Step 5. Run for concatenated. Running for Daily (1)
        ch = r.Run(conc,Curr[i],1)
        cht= ch.tail(10) #Can include more values if req, filters to last 10 here

        #Appending all cht dfs to view later if needed
        chts.append(cht)
        cht.reset_index(drop=True, inplace=True)
        
        #Loop through all dates for plays
        y = 0 #Counters to only show one date
        z = 0
        for x in range(0,len(cht)):           
            #200 MA Cross            
            if ((cht['200_MA_Cross'][x] != '0- No cross') & (y == 0)):
                    print('Currency: ', end =" ")
                    print(Curr[i])

                    print('Date: ', end =" ")
                    print(cht['<DATE>'][x])

                    print('What happened?', end =" ")
                    print(cht['200_MA_Cross'][x])
                    
                    print('Play category / type: ', "'Close v 200 MA'", sep =" ")
                    y = 1
                    
            #50 v 200 MA Cross
##            if ((cht['MA_Cross_2'][x] != '0- No crosses2') & (z == 0)):
##                    print('Currency: ', end =" ")
##                    print(Curr[i])
##
##                    print('Date: ', end =" ")
##                    print(cht['<DATE>'][x])
##
##                    print('What happened?', end =" ")
##                    print(cht['MA_Cross_2'][x])
##                    
##                    print('Play category / type: ', "'50 v 200 MA'", sep =" ")
##                    z = 1

        #Print new line only when there was a play
        if ((y == 1)| (z == 1)): 
                print('\n')

#Getting all data in one spot
chts = pd.concat(chts)

#For further research
def Res():
        for i in range(0,len(Curr)):
                print(i,Curr[i],sep = " ")
        Cur = int(input('Press option for Currency: '))
        return Cur

#Further research parameters
def Yes(i,o,Curr):
        df = chts[['Currency','Type',
                          'Pivot_Lev',
                          'Single',
                          'Double',
                          'Triple',
                          'MA_Cross', #10 v 50
                          'MA_Cross_2', #50 v 200. Other (and this) MA Crosses considered but too few to take note
                          '200_MA_Cross',
                          'Max',
                          'Min',
                          #'Close_3',
                          #'Close_5',
                          '<CLOSE>'
                    ]].loc[chts['Currency'] == Curr]
        print(df)

        #DK Trades, Long and Short
        df['DK_l'] = df['<CLOSE>'] + 0.003 #(20 pips for Calendar; 50 pips for DK)
        df['DK_l_tp'] = df['DK_l'] + 0.001 #Optimised for Perc
        df['DK_l_sl'] = df['DK_l'] - 0.003 #Optimised for Perc

        df['DK_s'] = df['<CLOSE>'] - 0.003
        df['DK_s_tp'] = df['DK_s'] - 0.001 #Optimised for Perc
        df['DK_s_sl'] = df['DK_s'] + 0.003 #Optimised for Perc

        #DK (Normal) print and play
        print('DK With trades are as follows:')
        df2 = df[['<CLOSE>','Pivot_Lev','DK_l','DK_l_tp','DK_l_sl','DK_s','DK_s_tp','DK_s_sl']].tail(1)
        df2.reset_index(drop=True, inplace=True)
        print(df2)

        #Check if the play should be fade play- Fade if Dk_l or dk_s is within 0.2 and 0.8 pips
        fade = 0
        
        if (((100*df2['DK_l'][0] - int(100*df2['DK_l'][0]) < 0.2) or
            (100*df2['DK_l'][0] - int(100*df2['DK_l'][0]) > 0.8)) or
            ((100*df2['DK_s'][0] - int(100*df2['DK_s'][0]) < 0.2) or
            (100*df2['DK_s'][0] - int(100*df2['DK_s'][0]) > 0.8))):
                fade = 1
        else:
                fade = 0

        #DK fade Trades, Long and Short
        df_fade = df.copy()
        df_fade['DK_l'] = df_fade['<CLOSE>'] - 0.003 #(20 pips for Calendar; 50 pips for DK)
        df_fade['DK_l_tp'] = df_fade['DK_l'] + 0.001 #Optimised for Perc
        df_fade['DK_l_sl'] = df_fade['DK_l'] - 0.003 #Optimised for Perc

        df_fade['DK_s'] = df_fade['<CLOSE>'] + 0.003
        df_fade['DK_s_tp'] = df_fade['DK_s'] - 0.001 #Optimised for Perc
        df_fade['DK_s_sl'] = df_fade['DK_s'] + 0.003 #Optimised for Perc

        #DK (Normal) print and play
        print('DK Fade trades are as follows:')
        df2_fade = df_fade[['<CLOSE>','Pivot_Lev','DK_l','DK_l_tp','DK_l_sl','DK_s','DK_s_tp','DK_s_sl']].tail(1)
        df2_fade.reset_index(drop=True, inplace=True)
        print(df2_fade)

        if o == 0: #Do DK trade
                r.Candle(chts.loc[chts['Currency'] == Curr],Curr)

        if (o == 1 and fade == 0): #Do DK With trade
                print('Yay')
                #Set DK Buy
                df3 = df2.copy()
                df3['DK'] = df2['DK_l']
                df3['DK_sl'] = df2['DK_l_sl']
                df3['DK_tp'] = df2['DK_l_tp']
                
                #Set DK Sell
                df4 = df2.copy()
                df4['DK'] = df2['DK_s']
                df4['DK_sl'] = df2['DK_s_sl']
                df4['DK_tp'] = df2['DK_s_tp']                
                
                #Get to Pepperstone app and do DK
                dk.pyautogui.click(600, 900)
                dk.DK(df3,Curr,1,2)
                dk.DK(df4,Curr,2,2)
                
        if (o == 1 and fade == 1): #No With play
                print('No with play here matey')

        if (o == 2 and fade == 0): #No fade play
                print('No fade play here matey')

        if (o == 2 and fade == 1): #Do DK fade play for both
                df3_fade = df2_fade.copy()
                df3_fade['DK'] = df2_fade['DK_l']
                df3_fade['DK_sl'] = df2_fade['DK_l_sl']
                df3_fade['DK_tp'] = df2_fade['DK_l_tp']
                
                #Set DK Sell
                df4_fade = df2_fade.copy()
                df4_fade['DK'] = df2_fade['DK_s']
                df4_fade['DK_sl'] = df2_fade['DK_s_sl']
                df4_fade['DK_tp'] = df2_fade['DK_s_tp']                
                
                #Get to Pepperstone app and do DK
                dk.pyautogui.click(600, 900)
                dk.DK(df3_fade,Curr,1,1)
                dk.DK(df4_fade,Curr,2,1)

cnt = 0

#Check loop
for j in range(0,30):
    #Ch = int(input('For curr. check press 1: '))
    Ch = 1
    if Ch == 1:
            i = Res()
##            i = j
            if (i >= 0) & (i <= 30):
                    Yes(i,0,Curr[i])
##                    x = int(input('Do trade for both? 1 for yes'))
##                    if (x == 1) | (x == 2):
##                            cnt = 1
##                            Yes(i,x,Curr[i])
            else:
                    print('Try again')
    else:
            input('Press any key. ')
            break

#For no DK plays selected
if cnt == 0:
    if (date.today().weekday() > 0) & (date.today().weekday() < 5):
            #Check loop
            for j in range(0,len(Curr)):
                    Yes(j,1,Curr[j])
                    dk.pyautogui.hotkey('alt', 'tab')

input('Bye ')
