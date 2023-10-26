# Author: Sukhwinder Ahluwalia
# Purpose: Get descriptive stats for columns passed through
# Big thanks to Jeff Sackman for creating this rich repository of tennis data. 

desc_stat <- function(data,col,w) {

# Filtering on window
  data <- data %>%
    filter(window == w)
  
  #Creating the rank column
  rank <- data %>%
    group_by(start_year) %>%
    arrange(desc({{col}})) %>%
    mutate(rank = row_number()) %>%
    select(winner_id, start_year, rank) # Select only ID, Year, and Rank columns
    
  # Merge the ranks back to the original data based on ID and Year
  data2 <- merge(data, rank, by = c("winner_id", "start_year"))
  
  # Getting basic desc stats on this data
  descriptive_stats <- data2 %>%
    group_by(winner_name) %>%
    summarize(
      avg_ratio = round(mean({{col}}), 2), #Average ratio
      median_ratio = round(median({{col}}), 2), #Median ratio
      max_ratio = round(max({{col}}), 2), #Max ratio
      min_ratio = round(min({{col}}), 2), #Min ratio
      min_rank = max(rank), #Min rank
      max_rank = min(rank), #Max rank
      mode_rank1 = as.numeric(names(sort(table(rank), decreasing = TRUE)[1])), #Mode / most common rank
      count = n(), #Number of n-year windows appeared in
    )  %>%
    arrange(desc(median_ratio), mode_rank1,desc(count))
  
  return(descriptive_stats)
}