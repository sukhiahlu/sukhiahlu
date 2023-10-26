# Author: Sukhwinder Ahluwalia
# Purpose: Analyse tennis data
# Win/loss ratio - the premise is that the best player will have a good win/loss ratio across a few years.
# We capture the results and then look at the 3 year windows over time to analyse again

# Big thanks to Jeff Sackman for creating this rich repository of tennis data. 
# Tennis databases, files, and algorithms by Jeff Sackmann / 
# Tennis Abstract is licensed under aCreative Commons 
# You can find his work here: https://github.com/JeffSackmann.

# This code is built using ChatGPT. My prompts are included in another file

# Function to define year ranges
get_winloss <- function(tennis_data,x,y,z = NULL) {
  # Define the range of years you want to filter for
  start_year <- x  # Replace with the desired start year
  end_year <- y   # Replace with the desired end year
  
  if (!is.null(z)) {
    # If a column to filter is specified, apply the filter
    tennis_data <- tennis_data[tennis_data$surface == z, ]
  } else {
    # If no column to filter is specified, return the data as is
    tennis_data <- tennis_data
  }
  
  # Filter data for the specified year range
  tennis_data <- tennis_data %>%
    filter(year(parse_date_time(tourney_date, orders = "%Y%m%d")) %in% start_year:end_year)
  
  # Group by 'winner_id' and calculate wins and losses
  win_loss_data <- tennis_data %>%
    group_by(winner_id) %>%
    summarize(wins = n())
  
  # Using 'loser_id' column, we calculate losses
  win_loss_data <- win_loss_data %>%
    left_join(tennis_data %>% group_by(loser_id) %>% summarize(losses = n()), by = c("winner_id" = "loser_id")) %>%
    replace_na(list(losses = 0))
  
  # Calculate the win/loss ratio
  win_loss_data <- win_loss_data %>%
    mutate(win_loss_ratio = wins / (wins + losses)) %>%
    mutate(total_matches_played = wins + losses)
  
  # Adding all relevant columns, removing dupes and doing a filter for matches played
  win_loss_data <- win_loss_data %>%
    left_join(tennis_data %>% select(winner_id, winner_name), by = "winner_id") %>%
    select(winner_id, winner_name, wins, losses, total_matches_played, win_loss_ratio) %>%
    distinct() %>%
    filter(total_matches_played >= 50) %>%
    arrange(desc(win_loss_ratio)) %>%
    mutate(rank = row_number()) %>%
    mutate(start_year = x) %>%
    mutate(end_year = y)
  
  # View the resulting win/loss ratio data
  return(win_loss_data)
}