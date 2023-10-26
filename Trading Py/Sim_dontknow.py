# Author: Sukhwinder Ahluwalia
# Purpose: Do a simple trade for unknown up or down in the day
# Date: 3rd September 2021

#Sims for the hourly data to check what random trades (likely bread and butter ones)

import pandas as pd #Where would i be without this?
pd.options.mode.chained_assignment = None  # default='warn' for the Order one

import numpy as np #Numbers (not used)
import plotly.graph_objs as go #Charts
import glob as g #For filepath saving

import Runs as r #Other code

import winsound #For sound

import datetime

import os ##Filepath creation

#The Dont know sim 
def DK_Sim(df, sl, tp, j):
           #Counter to break
           z = 0

           #Counter to return 0 values
           zz = 0
           
           #Check all next hours if they have a higher/lower close than Long/Short trades than close at 0:00
           for k in range(1,len(df)):
               if (df['<CLOSE>'][k] >= df['Long_trade_close'][0]): #Higher than Long
                   
                   #Create new df for all time after k
                   df2 = df.loc[df.index[k]:]
                   df2['Price_Change'] = df2['Order'] - df2['Order'][0]
                   
                   #Loop through all transactions to find the first condition to break (Stop Loss or Target)
                   for a in range(1, len(df2.index)):
                       #Stop Loss: 40
                       if df2['Price_Change'][a] <= sl:
##                           print('Long: Stopped mate')
##                               print(df.index[a])
                           z = 1
                           zz = 1
                           return j, a, sl, tp, df2['Price_Change'][a], 'Long'
                           break
                         
                       #Stop Loss: 40
                       elif df2['Price_Change'][a] >= tp:
                           z = 1
                           zz = 1
                           return j, a, sl, tp, df2['Price_Change'][a], 'Long'
                           break
                        
               elif (df['<CLOSE>'][k] <= df['Short_trade_close'][0]): #Lower than short

                   #Create new df for all time after k
                   df2 = df.loc[df.index[k]:]
                   df2['Price_Change'] = df['Order'] - df2['Order'][0]
                   
                   #Loop through all transactions to find the first condition to break (Stop Loss or Target)
                   for a in range(1, len(df2.index)):
                       #Stop Loss: 40
                       if df2['Price_Change'][a] >= -1*sl:
                           z = 1
                           zz = 1
                           return j, a, sl, tp, -1*df2['Price_Change'][a], 'Short'
                           break
                       #Stop Loss: 40
                       elif df2['Price_Change'][a] <= -1*tp:
                           z = 1
                           zz = 1
                           return j, a, sl, tp, -1*df2['Price_Change'][a], 'Short'
                           break
                                            
               if z != 0: #Break the loop if there is a TP/SL               
                    break

           if zz == 0: #Condition for not done              
                return j, -1, sl, tp, 0, 'None'

#Get data file names for H1/M1
path2 = r'C:\Users\nitis\Documents\Forex\Data\H1 data'
hourly = g.glob(path2 + "/*.csv")

#Currency runs
Curr = ['AUDUSD','EURAUD','EURGBP','EURUSD','GBPAUD','GBPUSD']

#At 't ' am/pm run dont_know with +-x pips
def run_DK(x,t):

    #Empty list
    save = list()

    #Loop through all currencies, len(hourly)
    for i in range(0,len(hourly)):
    
        #Hourly runs
        hr = pd.read_csv(hourly[i],sep='\t', parse_dates=[['<DATE>', '<TIME>']])
        hr.set_index(['<DATE>_<TIME>'], inplace=True, drop=True)
        hr['Time'] = hr.index.time #For time based operations
        hr['Date'] = hr.index.date #For date based operations
        hr['Diff_Date'] = hr['Date'] - hr['Date'].shift(1)
        hr['Order'] = 10000*hr['<OPEN>'] #Assume 10k transaction

        #Long and Short trades (at 20 pips from open/close)
##        hr.loc[(hr['Type'] == 'Bull'), 'Long_trade_oc'] = hr['<CLOSE>'] + 0.002
##        hr.loc[(hr['Type'] == 'Bear'), 'Long_trade_oc'] = hr['<OPEN>'] + 0.002
##        hr.loc[(hr['Type'] == 'Bull'), 'Short_trade_oc'] = hr['<OPEN>'] - 0.002
##        hr.loc[(hr['Type'] == 'Bear'), 'Short_trade_oc'] = hr['<CLOSE>'] - 0.002

        #Long and Short trades (at x pips from open/close)
        hr['Long_trade_close'] = hr['<CLOSE>'] + x #x pips
        hr['Short_trade_close'] = hr['<CLOSE>'] - x
        #Sims
        print(Curr[i])
        
        #Empty dataframe to save stuff
        summs = list()

        #Check for many SL and TP
##        for sl in (-20, -40, -50, -100):
##            for tp in (20, 30, 50, 100, 150, 200):
        for sl in (-5, -10, -15):
            for tp in (5, 10, 15, 20, 30):
                if -1*(tp/sl) in (3, 2.5, 2, 1.5):
                #if -1*(tp/sl) ==10:

                    print("SL =")
                    print(sl)
                            
                    #Run Sims for start of data only
                    #For any t time do the Sim. Monday filter for week run optional (note Monday runs need t = 0
                    for j in range(0,len(hr)):
                       if (hr['Time'][j] == datetime.time(t, 0)): #& (hr['Diff_Date'][j] == datetime.timedelta(days=3)):
                    
                            #Empty dataframe to save stuff
                            df_emp = list()
                            
                            #Create new df for all time after j
                            df = hr.loc[hr.index[j]:]
                            df['Price_Change'] = df['Order'] - df['Order'][0]
            
                            df_emp.append(DK_Sim(df, sl, tp, j))
                            
                            #Saving tuple list into a dataframe (legacy columns 'Tran_Price', 'Orig_Price')
                            df_emp = pd.DataFrame(df_emp, columns =['Index', 'Num', 'Stop_Loss', 'Target', 'Price_Change', 'Action'])

                            #Append to summs
                            summs.append(df_emp)

        #Print summs for a Currency
        summs = pd.concat(summs)
        summs = summs.loc[summs['Action']!='None']

        #Output pivots etc.
        piv = summs.pivot_table(index=['Action','Stop_Loss','Target']
                  ,aggfunc=['count'
                            ,'mean'
                            #,'min'
                            #,'max'
                            ,'median'
                            ,'sum']
                  ,values=['Price_Change'])

        #Renaming columns
        piv.columns = ['count','mean','median','sum']

        #Get perc positive
        #Number of positive (True) / negative (False) runs
        summs['PC_ch'] = summs['Price_Change'].apply(lambda x: x > 0)

        #Use the positives only
        df_emp3 = summs[summs['PC_ch'] == True]

        #Count over the positives and get avg days to Target
        grp = df_emp3.groupby(['Action', 'Stop_Loss', 'Target']).count()
        grp.rename(columns={"Price_Change": "Pos_count"}, inplace=True)
        
        #Join on pinot result + count positives
        res = piv.merge(grp[['Pos_count']],left_index=True, right_index=True, how='left')
        
        #Getting Percentage of positives and adding SL/TP
        res['Perc'] = res['Pos_count']/res['count']
        res['Curr'] = Curr[i]
        save.append(res)

    #Returning all saved files
    save = pd.concat(save)
    return save

#Output file paramters
path_o = r'C:\Users\nitis\Documents\Forex\Data\Output data\Dont_know'
##name =  0#File name

##Define x and t in loops
##for x in (0.002, 0.005, 0.01): #0.001
##    #for t in (6, 9): #Run for 12?
##    t = 0
##    save = run_DK(x,t)
##
##    #Change name before run
##    name = name+1
##    print(name) #File number
##    o = 'Small_' + str(name) + '.csv'
##    output_file = os.path.join(path_o,o)
##    save['Num_pips'] = x*10000
##    save['Time'] = t
##    save.to_csv(output_file)
##
##    #Sound alert!
##    duration = 1000  # milliseconds
##    freq = 1000  # Hz
##    winsound.Beep(freq, duration)

#For Manual runs
##yo = run_DK(0.005, 0)
##output_file = os.path.join(path_o,'File2_Mon.csv')
##yo['Num_pips'] = 50
##yo['Time'] = 0
##yo.to_csv(output_file)

#Empty list
emp = list()

dk = g.glob(path_o + "/*Small*.csv")
#emp = pd.read_csv(dk[16])

#Analyse them files        
for z in range(0,len(dk)):
        save = pd.read_csv(dk[z])
        emp.append(save)

emp = pd.concat(emp)

emp.reset_index(inplace = True)
#print(emp)

#Hypothesis: Time doesnt make a difference to any of the variables
#This means for the same currency pair, SL, TP, Action, Num pips. The output for different times should be the same
piv = emp.pivot_table(index='Curr'#Curr; ['Action','Stop_Loss','Target', 'Curr', 'Num_pips']
                  ,aggfunc=['mean'
                            ,'min'
                            ,'max'
                            ,'median']
                  ,values=['mean','count','median', 'sum', 'Perc', 'Pos_count'])
print(piv)

#Another option for pivot is 'Cur' which is also similar
#Result: Later in the day seems to have higher returns. Might try more later hours

#Optimal pips (x) for the same currency pair, Action
##emp = emp.loc[emp['Num_pips']==20]
emp = emp.loc[emp['Num_pips']!=-200]

#By percentage of rights
summs2 = emp.sort_values(['Curr','Action','Perc'], ascending = (True, True, False))
###By total money earnt
summs3 = emp.sort_values(['Curr','Action','sum'], ascending = (True, True, False))
###By avg money earnt
summs4 = emp.sort_values(['Curr','Action','mean'], ascending = (True, True, False))
##
###Select the first (n=0) row but can be tweaked for any as needed
print('Perc')
print(summs2.groupby(['Curr','Action']).nth(0))
print(summs2.groupby(['Curr','Action'])[['Stop_Loss','Target','Num_pips']].nth(0).to_string(index=False))
##
print('Sum')
print(summs3.groupby(['Curr','Action']).nth(0))
print(summs3.groupby(['Curr','Action'])[['Stop_Loss','Target','Num_pips']].nth(0).to_string(index=False))

##print('Mean')
##print(summs4.groupby(['Curr','Action']).nth(0))
##print(summs4.groupby(['Curr','Action'])[['Stop_Loss','Target','Num_pips']].nth(0).to_string(index=False))

##piv2 = emp.pivot_table(index=['Action','Stop_Loss','Target', 'Curr', 'Num_pips']
##                  ,aggfunc=['mean'
##                            ,'min'
##                            ,'max'
##                            ,'median']
##                  ,values=['mean', 'sum', 'Perc'])
