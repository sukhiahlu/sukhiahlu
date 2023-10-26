#The goal is to create a simple Chess game in R
#Author - Sukwinder Ahluwalia

#Install required packages
#install.packages('rchess')

#LoAding packages
library(rchess)

# Create a new game:
chess_game <- Chess$new()

#Bot 1: Random!

# Define function for Random move generator for a given position:
get_random_move <- function(chess_game) {
  return(sample(chess_game$moves(), size = 1))
}

# Perform random legal moves until the game ends in mate or draw
while (!chess_game$game_over()) 
  {
  chess_game$move(get_random_move(chess_game))
}

# Plot the final position
plot(chess_game)

#Get all moves in game
chess_game$fen()
