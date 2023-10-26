# Author: Sukhwinder Ahluwalia
# Purpose: Get descriptive stats for columns passed through
# Big thanks to Jeff Sackman for creating this rich repository of tennis data. 

surface <- function(tennis_data,x,y,z = NULL) {
  #Filter on surfce if neded
  if (!is.null(z)) {
    # If a column to filter is specified, apply the filter
    tennis_data <- tennis_data[tennis_data$surface == z, ]
  } else {
    # If no column to filter is specified, return the data as is
    tennis_data <- tennis_data
  }
  
  # Create a shell dataset to combine all the output data
  combined_data <- data.frame()
  
  #Consider different year range windows. Interested in these ones to give less noisy results
  for (j in c(3, 5, 7, 9)) {
    
    # Run the loops for different ranges
    for (i in x:y) {
      # What we are running
      cat("The year is: ", i, "and the window is", j, "years. \n")
      start_time <- Sys.time()  # Record the start time
      #do the magic
      output = get_ratio(tennis_data,i,i+j) #get_winloss is other option
      output$window = j #Define the window column
      combined_data <- rbind(combined_data, output) #Combine with all previous runs
      # Measure the execution time of the function
      end_time <- Sys.time()  # Record the end time
      execution_time <- end_time - start_time  # Calculate execution time
      cat("Iteration execution Time:", execution_time, "seconds\n") #Print result
    }
  }
  return(combined_data)
}