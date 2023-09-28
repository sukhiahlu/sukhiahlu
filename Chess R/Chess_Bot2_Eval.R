#The goal is to create a simple Chess game in R
#Author - Nitish Arora

#Install required packages
#install.packages('rchess')

#Loading packages
library(rchess)

# Create a new game:
chess_game <- Chess$new()

# RChess functions:
# $moves()- set of all possible moves given current game situation incl w and b moves
# $new()- is for a new game
# $in_checkmate() is for checking checkmate
# $turn() to check whose move is it, == "w" or "b"
# $game_over() to check if game is over
# $fen() to get where each position sits
# $in_check() to check if king in check 


#Bot 2: Eval function

# Retaining Bot1: Random move generator for a given position
get_random_move <- function(chess_game) {
  return(sample(chess_game$moves(), size = 1)) 
}

# Position evaluation function
evaluate_position <- function(position) {
  # Test if black won
  if (position$in_checkmate() & (position$turn() == "w")) {
    return(-1000)
  }
  # Test if white won
  if (position$in_checkmate() & (position$turn() == "b")) {
    return(1000)
  }
  # Test if game ended in a draw
  if (position$game_over()) {
    return(0)
  }
  # Compute material advantage
  position_fen <- strsplit(strsplit(position$fen(), split = " ")[[1]][1], split = "")[[1]]
  white_score <- length(which(position_fen == "Q")) * 9 + length(which(position_fen == "R")) * 5 + length(which(position_fen == "B")) * 3 + length(which(position_fen == "N")) * 3 + length(which(position_fen == "P"))
  black_score <- length(which(position_fen == "q")) * 9 + length(which(position_fen == "r")) * 5 + length(which(position_fen == "b")) * 3 + length(which(position_fen == "n")) * 3 + length(which(position_fen == "p"))
  # Evaluate king safety
  check_score <- 0
  if (position$in_check() & (position$turn() == "w")) check_score <- -1
  if (position$in_check() & (position$turn() == "b")) check_score <- 1
  # Return final position score
  return(white_score - black_score + check_score)
}

# Score position via minimax strategy
minimax_scoring <- function(chess_game, depth) {
  # If the game is already over or the depth limit is reached
  # then return the heuristic evaluation of the position
  if (depth == 0 | chess_game$game_over()) {
    return(evaluate_position(chess_game))
  }
  
  # Run the minimax scoring recursively on every legal next move, making sure the search depth is not exceeded
  next_moves <- chess_game$moves()
  next_move_scores <- vector(length = length(next_moves))
  for (i in 1:length(next_moves)) {
    chess_game$move(next_moves[i])
    next_move_scores[i] <- minimax_scoring(chess_game, depth - 1) 
    #Use above function call to basically include depth, recursive call till depth = 0 to get to the 
    #eval_pos function to get the score for that stage
    chess_game$undo()
  }
  
  # White will select the move that maximizes the position score
  # Black will select the move that minimizes the position score
  if (chess_game$turn() == "w") {
    return(max(next_move_scores))
  } else {
    return(min(next_move_scores))
  }
}

# Select the next move based on the minimax scoring
get_minimax_move <- function(chess_game) {
  # Score all next moves via minimax
  next_moves <- chess_game$moves()
  next_move_scores <- vector(length = length(next_moves))
  for (i in 1:length(next_moves)) {
    chess_game$move(next_moves[i])
    # To ensure fast execution of the minimax function we select a depth of 1
    # This depth can be increased to enable stronger play at the expense of longer runtime
    next_move_scores[i] <- minimax_scoring(chess_game, 1)
    chess_game$undo()
  }
  
  # For white return the move with maximum score
  # For black return the move with minimum score
  # If the optimal score is achieved by multiple moves, select one at random
  # This random selection from the optimal moves adds some variability to the play
  if (chess_game$turn() == "w") {
    return(sample(next_moves[which(next_move_scores == max(next_move_scores))], size = 1))
  } else {
    return(sample(next_moves[which(next_move_scores == min(next_move_scores))], size = 1))
  }
}

# Function that takes a side as input ("w" or "b") and plays 10 games
# The selected side will choose moves based on the minimax algorithm
# The opponent will use the random move generator
play_10_games <- function(minimax_player) {
  game_results <- vector(length = 5)
  for (i in 1:5) {
    chess_game <- Chess$new()
    while (!chess_game$game_over()) {
      if (chess_game$turn() == minimax_player) {
        # Selected player uses the minimax strategy
        chess_game$move(get_minimax_move(chess_game))
      } else {
        # Opponent uses the random move generator
        chess_game$move(get_random_move(chess_game))
      }
    }
    # Record the result of the current finished game
    # If mate: the losing player is recorded
    # If draw: record a 0
    if (chess_game$in_checkmate()) {
      game_results[i] <- chess_game$turn()
    } else {
      game_results[i] <- "0"
    }
  }
  
  # Print the outcome of the 10 games
  print(table(game_results))
}

play_10_games("w")