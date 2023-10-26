# Author: Sukhwinder Ahluwalia
# Purpose: To check for different possible trades, indicators in the data
# Date: 31 March 2021

#######Simpler runs code to add indicators (MA, Candle types, Pivot)

import pandas as pd
import numpy as np
import plotly.graph_objs as go

from datetime import date

pd.set_option('display.max_columns', None)

#Run function. put df, name of currency and run type (7 = Weekly, any other number, prefer 1 = daily)
def Run(df,name,ch):

    #Currency and dates
    df['Currency'] = name
    df['<DATE>'] = pd.to_datetime(df['<DATE>'], format='%Y.%m.%d')

    #For weekly runs if needed        
    if (ch == 7):        
        df['Ch'] = np.floor(pd.to_numeric((df['<DATE>'] - df['<DATE>'].loc[0]).dt.days, downcast='integer')/ch)+1
        df = df.groupby('Ch').aggregate({'<DATE>':'first','<OPEN>':'first', '<CLOSE>':'last', '<LOW>':'min', '<HIGH>':'max'})
        df['Currency'] = name

    #Create new candle columns
    df.loc[df['<OPEN>'] > df['<CLOSE>'], 'Type'] = 'Bear'
    df.loc[df['<OPEN>'] <= df['<CLOSE>'], 'Type'] = 'Bull'
    df['Body'] = df['<CLOSE>'] - df['<OPEN>'] 
    df['Length'] = df['<HIGH>'] - df['<LOW>'] 

    #Upwick
    df.loc[df['<OPEN>'] > df['<CLOSE>'], 'Upwick'] = df['<HIGH>'] - df['<OPEN>']
    df.loc[df['<OPEN>'] <= df['<CLOSE>'], 'Upwick'] = df['<HIGH>'] - df['<CLOSE>']

    #Tail
    df.loc[df['<OPEN>'] > df['<CLOSE>'], 'Tail'] = df['<CLOSE>'] - df['<LOW>']
    df.loc[df['<OPEN>'] <= df['<CLOSE>'], 'Tail'] = df['<OPEN>'] - df['<LOW>']

    ##################SINGLE##########################################

    #V2 BULLISH - More than 95 percentile(~100 pip or 0.01) Body irrespective of rest
    df.loc[(df['Body'] > np.percentile(df['Body'], 95)) & (df['Type'] == 'Bull'), 'Single'] = '1- Large Body'

    #V2 BEARISH - Less than 5 percentile(~100 pip or -0.01) Body irrespective of rest
    df.loc[(df['Body'] < np.percentile(df['Body'], 5)) & (df['Type'] == 'Bear'), 'Single'] = '2- Large Body'

    #V2 BULLISH - More than 95 percentile tail (~50 pip .005) Tail irrespective of rest. Note some crowding out by Hammers
    df.loc[(df['Tail'] > np.percentile(df['Tail'], 95)) & (df['Type'] == 'Bull'), 'Single'] = '1- Large Tail Bull'
    df.loc[(df['Tail'] > np.percentile(df['Tail'], 95)) & (df['Type'] == 'Bear'), 'Single'] = '1- Large Tail Bear'

    #V2 BEARISH - More than 95 percentile upwick (~50 pip 0.005) Upwick irrespective of rest (95 percentile upwick). Note some crowding out by Hammers
    df.loc[(df['Upwick'] > np.percentile(df['Upwick'], 95)) & (df['Type'] == 'Bear'), 'Single'] = '2- Large Upwick Bear'
    df.loc[(df['Upwick'] > np.percentile(df['Upwick'], 95)) & (df['Type'] == 'Bull'), 'Single'] = '2- Large Upwick Bull'
    
    #BULLISH - Hammer / Hanging man = Small Upwick, Tail > 2*Body
    df.loc[(df['Upwick'] < 0.001) & (df['Type'] == 'Bull') & (df['Tail'] > 2*df['Body']), 'Single'] = '1- Bull Hammer'
    df.loc[(df['Upwick'] < 0.001) & (df['Type'] == 'Bear') & (df['Tail'] > -2*df['Body']), 'Single'] = '1- Bear Hammer'

    #BEARISH - Inverted Hammer / Shooting Star = Small Tail, Upwick > 2*Body
    df.loc[(df['Tail'] < 0.001) & (df['Type'] == 'Bull') & (df['Upwick'] > 2*df['Body']), 'Single'] = '2- Bull Inv. Hammer'
    df.loc[(df['Tail'] < 0.001) & (df['Type'] == 'Bear') & (df['Upwick'] > -2*df['Body']), 'Single'] = '2- Bear Inv. Hammer'

    #BULLISH - Bull Full Bar  = Small Upwick, Small Tail, Bull
    df.loc[(df['Upwick'] < 0.001) & (df['Type'] == 'Bull') & (df['Tail'] < 0.001), 'Single'] = '1- Bull Full Bar'

    #BEARISH - Bear Full Bar  = Small Upwick, Small Tail, Bear
    df.loc[(df['Upwick'] < 0.001) & (df['Type'] == 'Bear') & (df['Tail'] < 0.001), 'Single'] = '2- Bear Full Bar'

    #V2 BULLISH - Bull Full Bar2  = Small Upwick, Small Tail, Bull, More than 85 percentile body (50 pip ish for all 4)
    df.loc[(df['Upwick'] < 0.001) & (df['Type'] == 'Bull') & (df['Tail'] < 0.001)
           & (df['Body'] > np.percentile(df['Body'], 85)), 'Single'] = '1- Bull Full Bar2'

    #V2 BEARISH - Bear Full Bar2  = Small Upwick, Small Tail, Bear, More than 15 percentile (50 pip body ish for all 4)
    df.loc[(df['Upwick'] < 0.001) & (df['Type'] == 'Bear') & (df['Tail'] < 0.001)
           & (df['Body'] < np.percentile(df['Body'], 15)), 'Single'] = '2- Bear Full Bar2'

    ##################DOUBLE##########################################
    
    #BULLISH - Engulfing-Bull = Bear then larger body Bull. Note some crowding out by Tweezers
    df.loc[(df['Type'].shift(1) == 'Bear') & (df['Type'] == 'Bull') & (-1*df['Body'].shift(1) < df['Body']), 'Double'] = '1- Bull Engulf'

    #BEARISH - Engulfing-Bear = Bull then larger body Bear. Note some crowding out by Tweezers
    df.loc[(df['Type'].shift(1) == 'Bull') & (df['Type'] == 'Bear') & (df['Body'].shift(1) < -1*df['Body']), 'Double'] = '2- Bear Engulf'

    #BULLISH - Tweezer bottom = Bear then similar low and close Bull
    df.loc[(df['Type'].shift(1) == 'Bear') & (df['Type'] == 'Bull') & (df['<LOW>'].shift(1) > 0.998*df['<LOW>'])
           & (df['<LOW>'].shift(1) < 1.002*df['<LOW>']) & (df['<OPEN>'].shift(1) > 0.998*df['<CLOSE>'])
           & (df['<OPEN>'].shift(1) < 1.002*df['<CLOSE>']), 'Double'] = '1- Tweezer bottom'

    #BEARISH - Tweezer top = Bull then similar high and close Bear
    df.loc[(df['Type'].shift(1) == 'Bull') & (df['Type'] == 'Bear') & (df['<HIGH>'].shift(1) > 0.998*df['<HIGH>'])
           & (df['<HIGH>'].shift(1) < 1.002*df['<HIGH>']) & (df['<OPEN>'].shift(1) > 0.998*df['<CLOSE>'])
           & (df['<OPEN>'].shift(1) < 1.002*df['<CLOSE>']), 'Double'] = '2- Tweezer top'

    #V2 BULLISH - Continuing-Bull = Bull then larger body Bull
    df.loc[(df['Type'].shift(1) == 'Bull') & (df['Type'] == 'Bull') & (df['Body'].shift(1) < df['Body']), 'Double'] = '1- Bull Cont'

    #V2 BEARISH - Continuing-Bear = Bull then larger body Bear
    df.loc[(df['Type'].shift(1) == 'Bear') & (df['Type'] == 'Bear') & (df['Body'].shift(1) > df['Body']), 'Double'] = '2- Bear Cont'

    ##################TRIPLE##########################################
    
    #BEARISH - Evening Star = Bull then Meh bar then large Bear (more than half of original Bull)
    #Note: Second bar's body size is between 40 (-7 pips) and 60 (10 pips) th percentiles 
    df.loc[(df['Type'].shift(2) == 'Bull') & (df['Type'] == 'Bear') & (df['Body'].shift(1) < 0.001) &
           (df['Body'].shift(1) > -0.001) & (df['<CLOSE>'] < ((df['<OPEN>'].shift(2) + df['<CLOSE>'].shift(2))/2)) , 'Triple'] = '2- Evening Star'

    #BULLISH - Morning Star = Bear then Meh bar then large Bull (more than half of original Bear)
    #Note: Second bar's body size is between 40 (-7 pips) and 60 (10 pips) th percentiles 
    df.loc[(df['Type'].shift(2) == 'Bear') & (df['Type'] == 'Bull') & (df['Body'].shift(1) < 0.001) &
           (df['Body'].shift(1) > -0.001) & (df['<CLOSE>'] > ((df['<OPEN>'].shift(2) + df['<CLOSE>'].shift(2))/2)) , 'Triple'] = '1- Morning Star'

    #BULLISH - Three White Soldiers = Three bulls with second bull's body > first bull and second and third bulls having little upwick
    df.loc[(df['Type'].shift(2) == 'Bull') & (df['Type'].shift(1) == 'Bull') & (df['Type'] == 'Bull') & (df['Body'].shift(2) < df['Body'].shift(1)) &
           (df['Upwick'].shift(1) < 0.001) & (df['Upwick'] < 0.001) , 'Triple'] = '1- Three White Soldiers'

    #BEARISH - Three Black Crows = Three bears with second bear's body > first bears (note Body is negative in this case so condition is '>')
    #and second and third bears having little tails
    df.loc[(df['Type'].shift(2) == 'Bear') & (df['Type'].shift(1) == 'Bear') & (df['Type'] == 'Bear') & (df['Body'].shift(2) > df['Body'].shift(1)) &
           (df['Tail'].shift(1) < 0.001) & (df['Tail'] < 0.001) , 'Triple'] = '2- Three Black Crows'

    #V2 BEARISH - Cont Denied = Bull then Bull then large Bear (bigger than original Bull). First 2 bulls bigger than 70th perc (~20 pips)
    df.loc[(df['Type'].shift(2) == 'Bull') & (df['Type'].shift(1) == 'Bull') & (df['Type'] == 'Bear') & (df['Body'].shift(2) > np.percentile(df['Body'], 70))
           & (df['Body'].shift(1) > np.percentile(df['Body'], 70)) & (df['Body'] > -1*(df['Body'].shift(2))) , 'Triple'] = '2- Bull Cont Deny'

    #V2 BULLISH - Cont Denied = Bear then Bear then large Bull bigger than original Bear). First 2 bulls smaller than 30th perc (~20 pips)
    df.loc[(df['Type'].shift(2) == 'Bear') & (df['Type'].shift(1) == 'Bear') & (df['Type'] == 'Bull') & (df['Body'].shift(2) < np.percentile(df['Body'], 30))
           & (df['Body'].shift(1) < np.percentile(df['Body'], 30)) & (df['Body'] > -1*(df['Body'].shift(2))) , 'Triple'] = '1- Bear Cont Deny'

    ##################MA CROSS############################################
    
    #MA values
    df['MA_10'] = df['<CLOSE>'].rolling(10).mean()
    #df['MA_20'] = df['<CLOSE>'].rolling(20).mean()
    df['MA_50'] = df['<CLOSE>'].rolling(50).mean()
    df['MA_200'] = df['<CLOSE>'].rolling(200).mean()

    #10 v 50 MA base conditions
    df.loc[(df['MA_50'] < 0.998*df['MA_10']), 'MA_Comp'] = '10 Higher'
    df.loc[(df['MA_50'] > 1.002*df['MA_10']), 'MA_Comp'] = '50 Higher'
    df.loc[(df['MA_50'] <= 1.002*df['MA_10']) & (df['MA_50'] >= 0.998*df['MA_10']), 'MA_Comp'] = 'Same'

    #10 v 50 MA cross conditions
    df.loc[(df['MA_Comp'].shift(1) == '10 Higher') & (df['MA_Comp'] == 'Same'), 'MA_Cross'] = '2- From aboves'
    df.loc[(df['MA_Comp'].shift(1) == '50 Higher') & (df['MA_Comp'] == 'Same'), 'MA_Cross'] = '1- From belows'
    df.loc[((df['MA_Comp'].shift(1) == 'Same') & (df['MA_Comp'] != 'Same') | (df['MA_Comp'].shift(1) == df['MA_Comp'])), 'MA_Cross'] = '0- No crosses'

    #50 v 200 MA base conditions
    df.loc[(df['MA_200'] < 0.998*df['MA_50']), 'MA_Comp_2'] = '50 Higher'
    df.loc[(df['MA_200'] > 1.002*df['MA_50']), 'MA_Comp_2'] = '200 Higher'
    df.loc[(df['MA_200'] <= 1.002*df['MA_50']) & (df['MA_200'] >= 0.998*df['MA_50']), 'MA_Comp_2'] = 'Same'

    #50 v 200 MA cross conditions
    df.loc[(df['MA_Comp_2'].shift(1) == '50 Higher') & (df['MA_Comp_2'] == 'Same'), 'MA_Cross_2'] = '2- From aboves2'
    df.loc[(df['MA_Comp_2'].shift(1) == '200 Higher') & (df['MA_Comp_2'] == 'Same'), 'MA_Cross_2'] = '1- From belows2'
    df.loc[((df['MA_Comp_2'].shift(1) == 'Same') & (df['MA_Comp_2'] != 'Same') | (df['MA_Comp_2'].shift(1) == df['MA_Comp_2'])), 'MA_Cross_2'] = '0- No crosses2'

    ##################MA CLOSE############################################

    #10 MA v Close conditions
    df.loc[(df['MA_10'] < 0.998*df['<CLOSE>']), '10_MA_Comp'] = 'Close Higher'
    df.loc[(df['MA_10'] > 1.002*df['<CLOSE>']), '10_MA_Comp'] = '10 Higher'
    df.loc[(df['MA_10'] <= 1.002*df['<CLOSE>']) & (df['MA_10'] >= 0.998*df['<CLOSE>']), '10_MA_Comp'] = 'Same'

    #10 MA v Close cross conditions
    df.loc[(df['10_MA_Comp'].shift(1) == 'Close Higher') & (df['10_MA_Comp'] != 'Close Higher'), '10_MA_Cross'] = '2- From above'
    df.loc[(df['10_MA_Comp'].shift(1) == '10 Higher') & (df['10_MA_Comp'] != '10_MA_Comp'), '10_MA_Cross'] = '1- From below'
    df.loc[((df['10_MA_Comp'].shift(1) == 'Same') & (df['10_MA_Comp'] != 'Same') |
            (df['10_MA_Comp'].shift(1) == df['10_MA_Comp'])), '10_MA_Cross'] = '0- No cross'
    
    #50 MA v Close conditions
    df.loc[(df['MA_50'] < 0.998*df['<CLOSE>']), '50_MA_Comp'] = 'Close Higher'
    df.loc[(df['MA_50'] > 1.002*df['<CLOSE>']), '50_MA_Comp'] = '50 Higher'
    df.loc[(df['MA_50'] <= 1.002*df['<CLOSE>']) & (df['MA_50'] >= 0.998*df['<CLOSE>']), '50_MA_Comp'] = 'Same'

    #50 MA v Close cross conditions
    df.loc[(df['50_MA_Comp'].shift(1) == 'Close Higher') & (df['50_MA_Comp'] != 'Close Higher'), '50_MA_Cross'] = '2- From above'
    df.loc[(df['50_MA_Comp'].shift(1) == '50 Higher') & (df['50_MA_Comp'] != '50_MA_Comp'), '50_MA_Cross'] = '1- From below'
    df.loc[((df['50_MA_Comp'].shift(1) == 'Same') & (df['50_MA_Comp'] != 'Same') |
            (df['50_MA_Comp'].shift(1) == df['50_MA_Comp'])), '50_MA_Cross'] = '0- No cross'

    #200 MA v Close conditions
    df.loc[(df['MA_200'] < 0.99*df['<CLOSE>']), '200_MA_Comp'] = 'Close Higher'
    df.loc[(df['MA_200'] > 1.01*df['<CLOSE>']), '200_MA_Comp'] = '200 Higher'
    df.loc[(df['MA_200'] <= 1.01*df['<CLOSE>']) & (df['MA_200'] >= 0.99*df['<CLOSE>']), '200_MA_Comp'] = 'Same'

    #200 MA v Close cross conditions
    df.loc[(df['200_MA_Comp'].shift(1) == 'Close Higher') & (df['200_MA_Comp'] != 'Close Higher'), '200_MA_Cross'] = '2- From above'
    df.loc[(df['200_MA_Comp'].shift(1) == '200 Higher') & (df['200_MA_Comp'] != '200 Higher'), '200_MA_Cross'] = '1- From below'
    df.loc[((df['200_MA_Comp'].shift(1) == 'Same') & (df['200_MA_Comp'] != 'Same') |
            (df['200_MA_Comp'].shift(1) == df['200_MA_Comp'])), '200_MA_Cross'] = '0- No cross'

    #Recent MA cross apply to after cross event + 5 days
    #df.loc[(df['MA_Cross'].shift(5) != 'No cross'), 'Recent_MA_Cross'] = df['MA_Cross'].shift(5)
    #df.loc[(df['MA_Cross'].shift(4) != 'No cross'), 'Recent_MA_Cross'] = df['MA_Cross'].shift(4)
    df.loc[(df['MA_Cross'].shift(3) != '0- No cross'), 'Recent_MA_Cross'] = df['MA_Cross'].shift(3)
    df.loc[(df['MA_Cross'].shift(2) != '0- No cross'), 'Recent_MA_Cross'] = df['MA_Cross'].shift(2)
    df.loc[(df['MA_Cross'].shift(1) != '0- No cross'), 'Recent_MA_Cross'] = df['MA_Cross'].shift(1)
    df.loc[(df['Recent_MA_Cross'] != '2- From above') & (df['Recent_MA_Cross'] != '1- From below'), 'Recent_MA_Cross'] = '0- No cross'

    df['RSI'] = RSI(df['<CLOSE>'])


    ##################PIVOT############################################
    

    #One higher up identifier
    #For monthly pivots
    if ch == 7:
        df['Id'] = np.floor((df.index - 0)/4)+1
        df['PrevId'] = df['Id'] - 1
    #Weekly pivots
    else:
        df['Id'] = np.floor(pd.to_numeric((df['<DATE>'] - df['<DATE>'].loc[0]).dt.days, downcast='integer')/7)+1
        df['PrevId'] = df['Id'] - 1  

    dfId = df.groupby('Id').aggregate({'<OPEN>':'first', '<CLOSE>':'last', '<LOW>':'min', '<HIGH>':'max'})
    
    #Pivot
    dfId['PP'] = (dfId['<CLOSE>'] + dfId['<HIGH>'] + dfId['<LOW>'])/3
    #Resistance
    dfId['R1'] = 2*dfId['PP'] - dfId['<LOW>']
    dfId['R2'] =   dfId['PP'] +(dfId['<HIGH>'] - dfId['<LOW>'])
    dfId['R3'] =   dfId['R1'] +(dfId['<HIGH>'] - dfId['<LOW>'])
    #Support
    dfId['S1'] = 2*dfId['PP'] - dfId['<HIGH>']
    dfId['S2'] =   dfId['PP'] -(dfId['<HIGH>'] - dfId['<LOW>'])
    dfId['S3'] =   dfId['S1'] -(dfId['<HIGH>'] - dfId['<LOW>'])

    #Left join back to df
    df = df.merge(dfId[['PP','R1','R2','R3','S1','S2','S3']], left_on='PrevId', right_on='Id', how='left')

    #Checking the close against the different pivot levels
    df.loc[(df['<CLOSE>'] >= df['R1']), 'Pivot_Lev'] = 'Above R1'
    df.loc[(df['<CLOSE>'] >= df['R2']), 'Pivot_Lev'] = 'Above R2'
    df.loc[(df['<CLOSE>'] >= df['R3']), 'Pivot_Lev'] = 'Above R3'
    df.loc[(df['<CLOSE>'] <= df['S1']), 'Pivot_Lev'] = 'Below S1'
    df.loc[(df['<CLOSE>'] <= df['S2']), 'Pivot_Lev'] = 'Below S2'
    df.loc[(df['<CLOSE>'] <= df['S3']), 'Pivot_Lev'] = 'Below S3'
    df.loc[(df['<CLOSE>'] > df['S1']) & (df['<CLOSE>'] < df['PP']), 'Pivot_Lev'] = 'Pivot & S1'
    df.loc[(df['<CLOSE>'] < df['R1']) & (df['<CLOSE>'] > df['PP']), 'Pivot_Lev'] = 'Pivot & R1'
    df.loc[(df['<CLOSE>'] < 1.002*df['PP']) & (df['<CLOSE>'] > 1.002*df['PP']), 'Pivot_Lev'] = 'Near Pivot'

    #Sell for all pivot levels
    df.loc[(df['Pivot_Lev'] != 'Pivot & R1') & (df['Pivot_Lev'] != 'Near Pivot')
           & (df['Pivot_Lev'] != 'Pivot & S1'), 'Pivot_Sell'] = '2- ' + df['Pivot_Lev']
    df.loc[(df['Pivot_Lev'] == 'Pivot & R1') | (df['Pivot_Lev'] == 'Near Pivot')
           | (df['Pivot_Lev'] == 'Pivot & S1'), 'Pivot_Sell'] = '0- ' + df['Pivot_Lev']

    #Buy for all pivot levels
    df.loc[(df['Pivot_Lev'] != 'Pivot & R1') & (df['Pivot_Lev'] != 'Near Pivot')
           & (df['Pivot_Lev'] != 'Pivot & S1'), 'Pivot_Buy'] = '1- ' + df['Pivot_Lev']
    df.loc[(df['Pivot_Lev'] == 'Pivot & R1') | (df['Pivot_Lev'] == 'Near Pivot')
           | (df['Pivot_Lev'] == 'Pivot & S1'), 'Pivot_Buy'] = '0- ' + df['Pivot_Lev']

    ##################SUMMARY STATS############################################
    
    df['Max'] = df['<CLOSE>'].max()
    df['Min'] = df['<CLOSE>'].min()

    return df
    
#RSI: Purpose is to understand how much buy pressure and sale pressure exists
#Note- exact values don't matter as such its more about its relative change with time, trends, over 50, under 50 with
#RSI formula = 100 - (100/(1+RS)). RS = Avg Gains / Avg Loss over last 14 days. 
def RSI(series):
     delta = series.diff().dropna()
     u = delta * 0
     d = u.copy()
     u[delta > 0] = delta[delta > 0]
     d[delta < 0] = -delta[delta < 0]
     up = u.shift(1).ewm(13).mean() + (u/14)
     down = d.shift(1).ewm(13).mean() + (d/14)
     rs = up/down
     return 100 - (100 / (1 + rs))

#Median, mean, max, min and count checks 
def Check(df,index):
    print(df.pivot_table(index=index,aggfunc=['count','mean','min','max','median'],values=['Close+1','Close+2','Close+3']))

#Candle and MA lines
def Candle(df,title):
    #Building the chart
    data=[go.Candlestick(x = df['<DATE>'],
                        open  = df['<OPEN>'],
                        high  = df['<HIGH>'],
                        low   = df['<LOW>'],
                        close = df['<CLOSE>'],
                        name="Candles"), 
                    #go.Scatter(x = df['<DATE>'], y = df['MA_50'], line=dict(color='blue', width=3),name="MA-10"),
                    go.Scatter(x = df['<DATE>'], y = df['MA_200'], line=dict(color='purple', width=3),name="MA-200")]
    
    figSignal = go.Figure(data = data
                          , layout = go.Layout(title = title))
    figSignal.show()
    
###More historical runs
###EURUSD
dfh1 = pd.read_csv (r'C:\Users\nitis\Documents\Forex\Data\EURUSD.csv',sep='\t')
dfhEUR = Run(dfh1,'EURUSD',1)
##Check(dfhEUR,['Single'])
##Check(dfhEUR,['Triple'])
##
###GBPUSD
dfh2 = pd.read_csv (r'C:\Users\nitis\Documents\Forex\Data\GBPUSD.csv',sep='\t')
dfhGBP = Run(dfh2,'GBPUSD',1)
##
###AUDUSD
dfh3 = pd.read_csv (r'C:\Users\nitis\Documents\Forex\Data\AUDUSD.csv',sep='\t')
dfhAUD = Run(dfh3,'AUDUSD',1)
##Candle(dfhAUD,'AUDUSD')

##
###EURGBP
dfh4 = pd.read_csv (r'C:\Users\nitis\Documents\Forex\Data\EURGBP.csv',sep='\t')
dfhEURG = Run(dfh4,'EURGBP',7)
