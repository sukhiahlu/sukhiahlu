# Author: Sukhwinder Ahluwalia
# Purpose: Analyse tennis data
# How the matches get played — ratio of sets won to lost, most common types of sets won/lost minutes played
# The premise is that a great player also wins sets comprehensively. Fewer sets lost, fewer games lost etc.
# Also a shorter match, shorter rallies. Some players are notorious for longer matches so they get penalised here haha

# Big thanks to Jeff Sackman for creating this rich repository of tennis data. 
# Tennis databases, files, and algorithms by Jeff Sackmann / 
# Tennis Abstract is licensed under Creative Commons 
# You can find his work here: https://github.com/JeffSackmann.

# Load the data.table package if not already loaded
if (!require(data.table)) {
  install.packages("data.table")
  library(data.table)
}
# Function to define year ranges
get_ratio <- function(tennis_data,x,y) {
  # Define the range of years you want to filter for
  start_year <- x  # Replace with the desired start year
  end_year <- y   # Replace with the desired end year
  
  # Filter data for the specified year range
  tennis_data <- tennis_data %>%
    filter(year(parse_date_time(tourney_date, orders = "%Y%m%d")) %in% start_year:end_year)

  # Only select columns of interest
  tennis_data[,c("winner_id","winner_name","loser_id","loser_name","score"
                 ,"tourney_date","tourney_id")]
  # tennis_data = head(tennis_data,10000)
  
  #Data cleaning for: ABD (Abandoned), W/O (Walkover), RET (Retired), Tiebreak scores (Brackets), UNK (Unknown)
  tennis_data$score <- gsub("ABD|W/O|UNK|RET|\\([^)]+\\)|\\s*$", "", tennis_data$score)
  
  # Convert to data.table
  setDT(tennis_data)
  
  # Define the maximum number of sets
  max_sets <- 5
  
  # Split the score string into individual set scores
  set_scores_list <- strsplit(tennis_data$score, " ")
  
  # Assign the actual set scores and create columns for each set
  for (i in 1:length(set_scores_list)) {
    set_scores <- set_scores_list[[i]]
      for (j in 1:max_sets) {
      col_name <- paste0(j, "set_score")
      tennis_data[i, (col_name) := set_scores[j]]
    }
  }
  
  # Initialize matrices for game won and lost for each set
  game_won <- matrix(NA, nrow = nrow(tennis_data), ncol = max_sets)
  colnames(game_won) <- paste0("game_won_", 1:max_sets)
  game_lost <- matrix(NA, nrow = nrow(tennis_data), ncol = max_sets)
  colnames(game_lost) <- paste0("game_lost_", 1:max_sets)
  
  # Split each set score into individual games won and lost
  for (i in 1:max_sets) {
    col_name <- paste0(i, "set_score")
    set_scores <- tennis_data[[col_name]]
    games <- strsplit(set_scores, "-")
    game_won[, i] <- sapply(games, function(x) as.integer(x[1]))
    game_lost[, i] <- sapply(games, function(x) as.integer(x[2]))
  }
  
  # Create columns for set winners
  set_winners <- matrix(NA, nrow = nrow(tennis_data), ncol = max_sets)
  colnames(set_winners) <- paste0("set_", 1:max_sets, "_won")
  
  # 1 for set won
  # 0 for set lost or sets with not played (eg retired etc)
  for (i in 1:max_sets) {
    set_winners[, i] <- ifelse(
      is.na(game_won[, i]) | is.na(game_lost[, i]), 
      NA, 
      ifelse(game_won[, i] > game_lost[, i], 1, 0)
    )
  }
  
  # Create columns for set losers
  set_losers <- matrix(NA, nrow = nrow(tennis_data), ncol = max_sets)
  colnames(set_losers) <- paste0("set_", 1:max_sets, "_lost")
  
  # 1 for set won
  # 0 for set lost or sets with not played (eg retired etc)
  for (i in 1:max_sets) {
    set_losers[, i] <- ifelse(
      is.na(game_won[, i]) | is.na(game_lost[, i]), 
      NA, 
      ifelse(game_won[, i] < game_lost[, i], 1,  0)
    )
  }
  
  # Combine the matrices with your original data
  result_data <- cbind(tennis_data, game_won, game_lost, set_winners, set_losers)
  
  # Calculate the sum of sets won, sets lost, games won, and games lost
  sum_sets_won <- rowSums(result_data[, c("set_1_won", "set_2_won", "set_3_won", 
                                          "set_4_won", "set_5_won")], na.rm = TRUE)
  sum_sets_lost <- rowSums(result_data[, c("set_1_lost", "set_2_lost", "set_3_lost", 
                                           "set_4_lost", "set_5_lost")], na.rm = TRUE)
  sum_games_won <- rowSums(result_data[, c("game_won_1", "game_won_2", "game_won_3", 
                                           "game_won_4", "game_won_5")], na.rm = TRUE)
  sum_games_lost <- rowSums(result_data[, c("game_lost_1", "game_lost_2", "game_lost_3", 
                                            "game_lost_4", "game_lost_5")], na.rm = TRUE)
  
  # Create a data frame with the sums
  sums_df <- data.frame(
    "Sets_Won" = sum_sets_won,
    "Sets_Lost" = sum_sets_lost,
    "Games_Won" = sum_games_won,
    "Games_Lost" = sum_games_lost
  )
  
  # Append the sums to tennis_data
  result_data <- cbind(result_data, sums_df)
  
  # Only select columns of interest
  result_data = result_data[,c("winner_id","winner_name","loser_id","loser_name",
                               "score","Sets_Won","Sets_Lost", "Games_Won", "Games_Lost")]
  
  # Create a new column "Straight_Set_Win" based on Sets_Lost
  result_data$Straight_Sets <- ifelse(result_data$Sets_Lost == 0, 1, 0)
  
  # Create a new column "Type of Set" based on Games_Lost -> 
  # Letting this go for now, not v interesting and hard to do at match level, to be done at set level
  # 0 -> Bagel
  # 1 -> Breadstick
  # Other for all other (if both are 0)
  # result_data$Bagel_Set <- ifelse(result_data$Games_Lost == 0, 1, 0)
  # result_data$Bread_Set <- ifelse(result_data$Games_Lost == 1, 1, 0)
  
  #Calculate summary stats for winner_id
  winner_data <- result_data %>%
    group_by(winner_id) %>%
    summarize(
      Win_Sets_Won = sum(Sets_Won),
      Win_Sets_Lost = sum(Sets_Lost),
      Win_Games_Won = sum(Games_Won),
      Win_Games_Lost = sum(Games_Lost),
      Win_Straights = sum(Straight_Sets),
      Wins = n()
    )
  
  #Calculate summary stats for loser_id
  loser_data <- result_data %>%
    group_by(loser_id) %>%
    summarize(
      Lose_Sets_Won = sum(Sets_Lost), #The opponent won this match, so this player "won" the lost sets / games..
      Lose_Sets_Lost = sum(Sets_Won),
      Lose_Games_Won = sum(Games_Lost),
      Lose_Games_Lost = sum(Games_Won),
      Lose_Straights = sum(Straight_Sets),
      Losses = n()
    )
  
  # Merge the dataframes based on winner_id/loser_id
  result_summary <- merge(winner_data, loser_data, by.x = "winner_id", by.y = "loser_id", all.x = TRUE)

  # Renaming NAs as 0s:
  # 1. Identify numeric columns
  numeric_cols <- sapply(result_summary, is.numeric)
  # 2. Replace NAs with 0 for numeric columns only
  result_summary[numeric_cols] <- lapply(result_summary[numeric_cols], function(x) ifelse(is.na(x), 0, x))
  
  # Calculate the ratios:
  # win/loss for matches
  # sets win/loss for matches
  # games win/loss for matches
  # sets win/lost in wins
  # sets win/lost in losses
  # games  win/lost in wins
  # games win/lost in losses
  
  #This bit helps supersede the WinLoss script
  ratios_data <- result_summary %>%
    mutate(Total_matches_played = Wins + Losses,
          Match_Win_Loss_Ratio = Wins / (Wins + Losses),
          Sets_Win_Loss_Ratio = (Win_Sets_Won + Lose_Sets_Won) / (Win_Sets_Won + Lose_Sets_Won + Win_Sets_Lost + Lose_Sets_Lost),
          Games_Win_Loss_Ratio = (Win_Games_Won + Lose_Games_Won) / (Win_Games_Won + Lose_Games_Won + Win_Games_Lost + Lose_Games_Lost),
          Sets_Win_Loss_In_Wins = Win_Sets_Won / (Win_Sets_Won + Win_Sets_Lost),
          Sets_Win_Loss_In_Losses = Lose_Sets_Won / (Lose_Sets_Won + Lose_Sets_Lost),
          Games_Win_Loss_In_Wins = Win_Games_Won / (Win_Games_Won + Win_Games_Lost),
          Games_Win_Loss_In_Losses = Lose_Games_Won / (Lose_Games_Won + Lose_Games_Lost),
          Straights_Wins = Win_Straights / Wins,
          Straights_Losses = Lose_Straights / Losses,
          Straights_Wins_Tot = Win_Straights / (Wins + Losses),
          Straights_Losses_Tot = Lose_Straights / (Wins + Losses)
    )
  
  # Renaming NAs as 0s:
  # 1. Identify numeric columns
  numeric_cols <- sapply(ratios_data, is.numeric)
  # 2. Replace NAs with 0 for numeric columns only
  ratios_data[numeric_cols] <- lapply(ratios_data[numeric_cols], function(x) ifelse(is.na(x), 0, x))
  
  # Adding all relevant columns, removing dupes and doing a filter for matches played
  summary_data <- tennis_data %>%
    select(winner_id, winner_name) %>% 
    distinct() %>%
    right_join(ratios_data, by = "winner_id") %>%
    filter(Total_matches_played >= 50) %>%
    mutate(start_year = x) %>%
    mutate(end_year = y)

  # Return final dataset
  return(summary_data)
}