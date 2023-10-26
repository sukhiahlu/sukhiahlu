# Author: Sukhwinder Ahluwalia
# Purpose: To measure who is the best Tennis player of all time
# Date: 3rd September 2023
# This code is built using ChatGPT. My prompts are included in the file called prompts 

# Big thanks to Jeff Sackman for creating this rich repository of tennis data. 
# Tennis databases, files, and algorithms by Jeff Sackmann / 
# Tennis Abstract is licensed under Creative Commons 
# You can find his work here: https://github.com/JeffSackmann.

#Load packages
library(tidyverse)
library(lubridate)
library(profvis) # to monitor performance
library(tableHTML)

# Load the surface filter code
source("C:/Users/nitis/Documents/GitHub/Funn_Scripts/R/Data Analysis/Tennis/Surface.R")

# Load the descriptive stats code
source("C:/Users/nitis/Documents/GitHub/Funn_Scripts/R/Data Analysis/Tennis/Desc_stat.R")

# Load the ratio test code
source("C:/Users/nitis/Documents/GitHub/Funn_Scripts/R/Data Analysis/Tennis/Ratio_test.R")

#Load the data
# Specify the directory where your CSV files are located
csv_directory <- "C:/Users/nitis/Documents/GitHub/tennis_atp"

# Get a list of all CSV files in the directory
csv_files <- list.files(path = csv_directory, pattern = "^atp_matches_\\d{4}\\.csv$", full.names = TRUE)

# Read and combine the CSV files into a single dataset
tennis_data <- bind_rows(lapply(csv_files, read.csv))

# The analysis has 3, 5, 7, 9 year windows to make things less noisy and efficient
# raw data to analyse, year ranges for the loop and surface filter, if at all
# Surfaces tried: All (no argument passed), Clay, Grass, Carpet, Hard, "" (blank)
# Note with Grass because there are such few matches, we need to change a filter for total matches played..
# We ignore Carpet as very few matches in some years tagged for that surface

#If we need to do the Profile audit
# profvis({
#   data<-surface(tennis_data,2003,2003) 
# })

#No audit
data<-surface(tennis_data,1968,2020)

# output of prev function, ratio to analyse, window of focus
# All ratio columns to analyse
#analyse function to make it easier to run all windows of interest
analyse <- function(dataset,x,i) {
  for (j in c(3, 5, 7, 9)) {
    dataset_name <- paste("new_", i, j, sep = "_")
    assign(dataset_name,desc_stat(dataset,{{x}},j))
    #dynamic creation of View. Use Sprintf to create the required character string
    #parse to make it an R expression and eval to evaluate. output is dataset with the right identifier uptop
    eval(parse(text = sprintf('View(%s)', dataset_name)))
  }
}
#Run it for all columns of interest:
analyse(data,Match_Win_Loss_Ratio,'MWLR') # Match_Win_Loss_Ratio, 
#Sets analysis
analyse(data,Sets_Win_Loss_Ratio,'SWLR') # Sets_Win_Loss_Ratio, 
analyse(data,Sets_Win_Loss_In_Wins,'SWLiW') # Sets_Win_Loss_In_Wins, 
analyse(data,Sets_Win_Loss_In_Losses,'SWLiL') # Sets_Win_Loss_In_Losses, 
#Games analysis
analyse(data,Games_Win_Loss_Ratio,'GWLR') # Games_Win_Loss_Ratio, 
analyse(data,Games_Win_Loss_In_Wins,'GWLiW') # Games_Win_Loss_In_Wins, 
analyse(data,Games_Win_Loss_In_Losses,'GWLiL') # Games_Win_Loss_In_Losses
#Straight Sets analysis
analyse(data,Straights_Wins,'SW') # Straights_Wins
analyse(data,Straights_Wins_Tot,'SWT') # Straights_Wins_Tot
analyse(data,Straights_Losses,'SL') # Straights_Losses
analyse(data,Straights_Losses_Tot,'SLT') # Straights_Losses_Tot

# For Findings, see provided doc. Also see Prompts for ChatGPT stuff