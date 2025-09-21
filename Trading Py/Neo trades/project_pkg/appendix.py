#Storing here again JIC
# #! python3
# # mouseNow.py - Displays the mouse cursor's current position.
# import pyautogui
# import time
# #Mouse coordinates. = x = (0,1919) y = (0,1079)

# pyautogui.PAUSE = 4

# #Get to Pepperstone app
# pyautogui.click(25, 900)
# # pyautogui.PAUSE = 1
# pyautogui.click(350, 200)
# #pyautogui.hotkey('alt', 'tab')
# pyautogui.PAUSE = 2

# #Get to currency page
# pyautogui.hotkey('ctrl', 'u')

# #Get to Bars
# pyautogui.click(500, 235)
# #Download AUDUSD
# pyautogui.typewrite("audusd")
# pyautogui.hotkey('enter')
# pyautogui.hotkey('tab')
# pyautogui.typewrite('d') #Daily
# pyautogui.click(1000, 265) #Request
# pyautogui.click(480, 640) #Export
# #time.sleep(2)
# pyautogui.click(700, 215) #Change the directory for these
# pyautogui.typewrite("C:\\Users\\nitis\\Documents\\Forex\\Data\\New data") #Use directory
# pyautogui.hotkey('enter')
# pyautogui.click(700, 550) #Change the name
# pyautogui.typewrite('audusd') #Save file as aud
# pyautogui.hotkey('enter')
# pyautogui.typewrite('y')

# #Loop through all currencies, len(daily)
# for i in range(0,len(Curr)):
#     pyautogui.click(490, 265) #Currency type
#     pyautogui.typewrite(Curr[i])
#     pyautogui.hotkey('enter')
#     pyautogui.click(1000, 265) #Request
#     pyautogui.click(480, 640) #Export
# #    time.sleep(2)
#     pyautogui.typewrite(Curr[i]) #Save file 
#     pyautogui.hotkey('enter')
#     pyautogui.typewrite('y')

# #Exit currency page
# pyautogui.hotkey('esc')


# # Check on how to split data better
# base_path_hist = r"C:\Users\nitis\Documents\Forex\Data\New data\Hist data"
# #  store non-empty dfs
# collected_st = []
# collected_wk = []

# from io import StringIO

# #Loop through all currencies, len(daily)
# for i in range(0,len(Curr)):
#     path = fr"{base_path_hist}\{Curr[i]}.csv"    
#     with open(path, 'r', encoding='utf-8') as f:
#         text = f.read()
#     # remove one pair of surrounding quotes if present
#     if text.startswith('"') and text.count('\n')>0:
#         text = text.lstrip('"').rstrip('\n')
#         # also remove a trailing closing quote on the header line if present
#         first_line, rest = text.split('\n', 1)
#         if first_line.endswith('"'):
#             first_line = first_line.rstrip('"')
#         text = first_line + '\n' + rest
#     df = pd.read_csv(StringIO(text))
#     print(df)

# #     with open(path, 'rb') as f:
# #         first = f.readline()
# #         print(first)          # shows raw bytes; look for b'\\t' vs b'\t'
# #         print(first.decode('utf-8'))  # shows literal \t if present

#Save "historic" file
# #Join on to historical datasets
# # Analysis on the datasets
# base_path = r"C:\Users\nitis\Documents\Forex\Data\New data"
# base_path_hist = r"C:\Users\nitis\Documents\Forex\Data\New data\Hist data"

# #Loop through all currencies, len(daily)
# for i in range(0,len(Curr)):
#     #Load new data
#     path = fr"{base_path}\{Curr[i]}.csv"
#     df = pd.read_csv(path, sep='\t')
#     #Load hist data
#     path_hist = fr"{base_path_hist}\{Curr[i]}.csv"
#     df_hist = pd.read_csv(path_hist, sep='\t')
#     #Save total data in hist
#     combined = pd.concat([df_hist, df], ignore_index=True)
#     combined = combined.drop_duplicates()
#     combined.to_csv(path_hist, index=False)

#simple step get crosses pair
# # make DATE a datetime index (recommended)
# for df in (eur_df, cad_df):
#     df["<DATE>"] = pd.to_datetime(df["<DATE>"])
#     df.set_index("<DATE>", inplace=True)

# # align on DATE and multiply numeric columns elementwise
# # method: reindex union of dates then multiply (NaNs remain if a date missing in one df)
# common_index = eur_df.index.union(cad_df.index)
# left = eur_df.reindex(common_index)
# right = cad_df.reindex(common_index)

# # multiply elementwise for the four columns
# cols = ["<OPEN>","<HIGH>","<LOW>","<CLOSE>"] #Open and Close seem to be nearer, High/Low are wilder
# result = left[cols].multiply(right[cols])

# # optional: if you prefer to drop rows with any NaN (dates not present in both)
# result_dropped = result.dropna(how="any")

# print(result)

# #Download other Forex and Metals tickers
# Curr = [
#     #Currencies
# #     "AUDCAD", "AUDCHF",
# #     "AUDJPY", "AUDNZD", "AUDSGD", 
# #     "CADCHF", "CADJPY", "CHFJPY", "CHFSGD",
# #     "EURAUD", "EURCAD", "EURCHF", "EURCZK", "EURGBP", "EURHUF", "EURJPY",
# #     "EURMXN", "EURNOK", "EURNZD", "EURPLN", "EURSEK", "EURSGD", "EURTRY", "EURZAR",
# #     "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPMXN", "GBPNOK", "GBPNZD", "GBPSEK", "GBPSGD", "GBPTRY", 
# #     "NOKJPY", "NOKSEK", "NZDCAD", "NZDCHF", "NZDJPY", "SEKJPY", "SGDJPY",
#     "USDCAD", "USDCHF", "USDCNH", "USDCZK", "USDHUF", "USDJPY", "USDMXN", "USDNOK", "USDPLN", "USDSEK",
#     "USDSGD", "USDTHB", "USDTRY", "USDZAR", "USDIDR", "USDINR", 
#     'USDX.a','EURX',
#     "EURUSD", "GBPUSD","NZDUSD", "AUDUSD", 
    
# #     #Metals
# #     "XAGAUD", "XAGEUR",  "XAUAUD", "XAUCHF", "XAUEUR",
# #     "XAUGBP", "XAUJPY", 
#     "XAGUSD.a","XAUUSD.a", "XPTUSD.a",
    
# #     #ETFs
#     'AUS200.a','US30.a','US500.a','UK100.a','NAS100.a','EUSTX50.a','SPA35.a','JPN225.a', 'GER40.a','HK50.a',
    
# #     # US Shares tickers
#     "AMD.US-24", "BABA.US-24", "GOOG.US-24", "AMZN.US-24", "AAPL.US-24", "BAC.US-24", "CAT.US-24",
#     "CVX.US-24", "C.US-24", "XOM.US-24", "F.US-24", "GM.US-24", "HPQ.US-24", "IBM.US-24", "INTC.US-24",
#     "JPM.US-24", "JNJ.US-24", "MCD.US-24", "META.US-24", "MSFT.US-24", "NKE.US-24", "NVDA.US-24",
#     "NFLX.US-24", "ORCL.US-24", "PFE.US-24", "PG.US-24", "SLB.US-24", "SNAP.US-24", "TSLA.US-24",
#     "BA.US-24", "KO.US-24", "DIS.US-24", "UNH.US-24", "VZ.US-24", "RTX.US-24", "V.US-24", "WMT.US-24"

# ]