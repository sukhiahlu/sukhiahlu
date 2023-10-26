# Author: Sukhwinder Ahluwalia
# Purpose: Do a simple trade for unknown up or down in the day
# Date: 3rd September 2021

#Sims for the daily data

import pandas as pd #Where would i be without this?
pd.options.mode.chained_assignment = None  # default='warn' for the Order one

import numpy as np #Numbers (not used)
import plotly.graph_objs as go #Charts
import glob as g #For filepath saving

import Runs as r #Other code

import winsound #For sound

import datetime

#Another way to do Date time is by Concatenate:
#pd.to_datetime(df['Date'] + ' ' + df['Time'])

#Loop through dates to add 0 times
##for b in range(0, len(dfEURSing.index)):                
##    dfEURSing['Time'][b] = pd.to_datetime(pd.Timestamp(dfEURSing.index[b]), format = '%Y-%m-%d %H:%M:$S')

#Regex
#.str.contains(r'(?!$)Bull(?!$)') == True:
#import re
#regex = re.compile('Bull')
#re.match(regex, candle)

##Filepath creation (CHANGE THIS BEFORE RUN!!!!!!!!!)
import os
path_o = r'C:\Users\nitis\Documents\Forex\Data\Output data'
output_file = os.path.join(path_o,'New_AUD_MA.csv')

#Applying Apply
#leng.apply(lambda row: print(row.name), axis=1)

#Simulation function
def Sim (df, date, action, candle, col, sl, tp):
    #Assume 10k transaction and we buy at market of chosen date and time
    df2 = df.loc[date:]
    df2['Order'] = 10000*df2['<OPEN>']

    #Left join back to df to get all Order prices
    df = df.merge(df2['Order'],left_index=True, right_index=True, how='left')

    #Compare price to original price of transaction
    df['Price_Change'] = df['Order'] - df['Order'].loc[date]
    df = df.dropna()

    #For No cross
    if (action == '0'):
        return date - date, df['<OPEN>'].loc[date], df['<OPEN>'].loc[date], col, 0, 'Buy', candle
        
    #For Buy transactions
    #Go by action if a lot of plays, but specific checks for plays of concern need regex
    elif (action == '1'):    
            #Loop through all transactions to find the first condition to break (Stop Loss or Target)
            for a in range(1, len(df.index)):
                #Legacy columns: df['<OPEN>'][a], df['<OPEN>'].loc[date]
                #Stop Loss: 40
                if df['Price_Change'][a] <= sl:                    
                     return df.index[a] - date, sl, tp, col, df['Price_Change'][a], 'Buy', candle, df['Pivot_Lev'].loc[date]
                     break
                #Target: 100
                elif df['Price_Change'][a] >= tp:
                     return df.index[a] - date, sl, tp, col, df['Price_Change'][a], 'Buy', candle, df['Pivot_Lev'].loc[date]
                     break

    #For Sell transactions
    elif (action == '2'):    
            #Loop through all transactions to find the first condition to break (Stop Loss or Target)
            for a in range(1, len(df.index)):
                #print(df['Price_Change'][a])
                #Stop Loss: 40
                if df['Price_Change'][a] >= -1*sl:                    
                     return df.index[a] - date, sl, tp, col, -1*df['Price_Change'][a], 'Sell', candle, df['Pivot_Lev'].loc[date]
                     break
                #Target: 100
                elif df['Price_Change'][a] <= -1*tp:
                     return df.index[a] - date, sl, tp, col, -1*df['Price_Change'][a], 'Sell', candle, df['Pivot_Lev'].loc[date]
                     break

#Checking function for different types
#Merging on df (with time) created above
def Test (dfs, dfl, col, sl, tp):    
    check = dfs.merge(dfl.dropna(),left_index=True, right_index=True, how='left')
    check[['Action','Candle_type']] = check[col].str.split("- ",expand = True)    

    #Length for loop = when no NA in data and if action = 0
    leng = check.dropna()
    leng = leng.loc[leng['Action']!='0']

    #Empty tuple list to save stuff
    df2 = list()

    #Loop through desired range and save values to loop
    for b in range(0, len(leng.index)):
        df2.append(Sim(check, leng.index[b]
                       , leng['Action'][b], leng['Candle_type'][b], col, sl, tp))

    #Saving tuple list into a dataframe (legacy columns 'Tran_Price', 'Orig_Price')
    df3 = pd.DataFrame(df2, columns =['Num', 'Stop_Loss', 'Target', 'Type', 'Price_Change', 'Action', 'Candle', 'Comparer'])
    return df3

#Function to get summart stats for a currency run
def Summ(dfs, dfl):
    #Empty dataframe to save stuff in the loops
    df_emp = list()
    #Potential options to consider for SL and TP:
    #SL: -40, -50, -100, -200, -300, -500, -1000
    #TP: 100, 200, 300, 500, 1000
    #R-R ratios: 2.5-1, 2-1, 1-1
    for sl in (-40, -50, -100, -200, -300, -500, -1000):
        for tp in (100, 200, 300, 500, 1000):
            if -1*(tp/sl) in (2.5, 2, 1):
                #Loop through all columns of interest. Options commented
                for b in [
##                          'Single',
##                          'Double',
##                          'Triple',
##                          'Pivot_Sell',
##                          'Pivot_Buy',
                          'MA_Cross',
                          'MA_Cross_2',
                          '200_MA_Cross',
                          'Recent_MA_Cross',
##                          'Close_3',
##                          'Close_5',
##                          'Close_10',
##                          'Close_20'
                          ]:                      
                    df_emp.append(Test(dfs, dfl[[b,
                                                 'Pivot_Lev'
                                                 #'200_MA_Comp'                                                 
                                                ]], b, sl, tp))
                print("SL =")
                print(sl)
                #print(b)
            #else:
                #break #continue

    #Save and run the different stuff, remove the no cross checks
    df_emp = pd.concat(df_emp)    

    #Summary stats for all
    #Taken out from index:
            #'Type' for different types of patterns (Single, Double etc)
            #'Comparer' for Pivot/MA crosses
    piv = df_emp.pivot_table(index=[#'Type',
                                    'Candle'
                                    ,'Stop_Loss'
                                    ,'Target'
                                     #,'Comparer'
                                     ]
                              ,aggfunc=['count'
                                        ,'mean'
                                        #,'min'
                                        #,'max'
                                        ,'median'
                                        ,'sum']
                              ,values=['Price_Change'])

    #Renaming columns
    piv.columns = ['count','mean','median','sum']

    #Number of positive (True) / negative (False) runs
    df_emp['PC_ch'] = df_emp['Price_Change'].apply(lambda x: x > 0)

    #Use the positives only
    df_emp3 = df_emp[df_emp['PC_ch'] == True]
    df_emp3['Num'] = df_emp3['Num'].dt.days #The avg days solution was supposed to work without this but meh

    #Count over the positives and get avg days to Target
    grp = df_emp3.groupby([#'Type',
                           'Candle', 'Stop_Loss', 'Target'])

    #Count over the positives
    grp2 = grp.count()
    grp2.rename(columns={"Price_Change": "Pos_count"}, inplace=True)

    #and get avg days to Target
    grp2['Pos_avg_days'] = grp['Num'].sum() / grp['Num'].count()
    
    #Join on pinot result + count positives
    res = piv.merge(grp2[['Pos_count','Pos_avg_days']],left_index=True, right_index=True, how='left')
    
    #Getting Percentage of positives and adding SL/TP
    res['Perc'] = res['Pos_count']/res['count']
    return res

#Get data file names for Daily (Remove/Add "\Data rep\New curr" for old/new currenices)
path = r'C:\Users\nitis\Documents\Forex\Data'
daily = g.glob(path + "/*.csv")

#Get data file names for H1/M1
path2 = r'C:\Users\nitis\Documents\Forex\Data\H1 data'
hourly = g.glob(path2 + "/*.csv")

#Currency runs
Curr = ['AUDUSD','EURAUD','EURGBP','EURUSD','GBPAUD','GBPUSD']

#Empty dataframe to save stuff
summs = list()
#Loop through all currencies, len(daily)
for i in range(0,len(daily)):
        #Daily runs
        dl = pd.read_csv(daily[i],sep='\t')
        dlr= r.Run(dl,Curr[i],1)
        dlr.set_index(['<DATE>'], inplace=True, drop=True)

        #Hourly runs
        hr = pd.read_csv(hourly[i],sep='\t', parse_dates=[['<DATE>', '<TIME>']])
        hr.set_index(['<DATE>_<TIME>'], inplace=True, drop=True)

        #Sims
        print(Curr[i])
        ch = Summ(hr, dlr)        
        ch['Curr'] = Curr[i]
        
        summs.append(ch)

summs = pd.concat(summs)
print(summs)

#Change name before run (above)!
summs.to_csv(output_file)

#Sound alert!
duration = 1000  # milliseconds
freq = 1000  # Hz
winsound.Beep(freq, duration)

#Check for run file and remove the sucky plays
summs = pd.read_csv(output_file)
summs = summs.loc[summs['count']>30]
summs = summs.loc[(summs['sum']>20000) | (summs['Perc']>0.6)]

#Select best SL/TP
#By percentage of rights
summs2 = summs.sort_values(['Curr','Candle','Perc'], ascending = (True, True, False))
#By total money earnt
summs3 = summs.sort_values(['Curr','Candle','sum'], ascending = (True, True, False))
#By avg money earnt
summs4 = summs.sort_values(['Curr','Candle','mean'], ascending = (True, True, False))

#Select the first (n=0) row but can be tweaked for any as needed
print('Perc')
print(summs2.groupby(['Curr','Candle']).nth(0))
print(summs2.groupby(['Curr','Candle'])[['Stop_Loss','Target']].nth(0).to_string(index=False))

print('Sum')
print(summs3.groupby(['Curr','Candle']).nth(0))
print(summs3.groupby(['Curr','Candle'])[['Stop_Loss','Target']].nth(0).to_string(index=False))

print('Mean')
print(summs4.groupby(['Curr','Candle']).nth(0))
print(summs4.groupby(['Curr','Candle'])[['Stop_Loss','Target']].nth(0).to_string(index=False))

#DONE
#Maybe need to filter first to desired date apply the function and then map back to riginal data for sim.
#Finding a loss threshold or profit targets
#Collect the results of Sim into a dataframe
#Simpler than below, combine the two scripts 
#Feeding in different dates not just a fixed one: loop implemented but needs to be specific to dates and transaction types
#Datetime to Date checks and how to use df and dfEUR in the loop
#Some checks on certain candle types, MAs etc.
#1 HR data?
#Get more data. More granular time periods maybe?
#Simulation of trades ^ using granulr time to see how much $$ make for some plays
#I think Pivots are a good way to monitor potential moves (sims using say m1 data maybe but arent a check criteria). #TO DO: , 'Pivot_Lev'
#Chck for >100 runs (a loop in those conditions
#Sound alert for when done
#DONE Sim checks: Double on currencies, stop loss and target requirements- Document these!
#NOT WORTH IT Sim checks: Triple  on currencies, stop loss and target requirements- Document these!#Sim checks: 200_MA on currencies, stop loss and target requirements- Document these!
#Sim checks: Single on currencies, stop loss and target requirements- Document these!
#Also Sims for MA Cross (10/20 had some decents, 20/50 sucked!)
#Other MA options besides 200 (50, 100, 150)
#Exciting times for automation script
