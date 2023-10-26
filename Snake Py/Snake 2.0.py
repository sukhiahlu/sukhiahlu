#Author - Sukwinder Ahluwalia
#Purpose: Second change to after first edit. Key difference is now snake can enter from the otherside
#Date: 31 Dec 2020

# Thanks to aqeelanwar for this OG script from here: https://github.com/aqeelanwar/Snake-And-Apple/blob/master/main.py
# I modified this here
# Icons: https://www.flaticon.com/authors/freepik
from tkinter import *
import random
import time
import numpy as np
import pyautogui
##from PIL import ImageTk,Image

# Define useful parameters
size_of_board = 600
rows = 10
cols = 10
DELAY = 100
snake_initial_length = 3
symbol_size = (size_of_board / 3 - size_of_board / 8) / 2
symbol_thickness = 2
RED_COLOR = "red"
BLUE_COLOR = "greenyellow"
Green_color = "green"

BLUE_COLOR_LIGHT = 'blue'
RED_COLOR_LIGHT = 'bisque'

class SnakeAndApple:
    # ------------------------------------------------------------------
    # Initialization Functions:
    # ------------------------------------------------------------------
    def __init__(self):
        self.window = Tk()
        self.window.title("Snake-and-Apple")
        self.window.configure(bg='dimgray')
        self.canvas = Canvas(self.window, bg="black", width=size_of_board, height=size_of_board)
        self.canvas.pack()
        # Input from user in form of clicks and keyboard
        self.window.bind("<Key>", self.key_input)
        self.window.bind("<0>", self.mouse_input)
        self.window.bind("<.>", self.pause)
##        self.window.bind("<o>", self.unpause)        
            
        self.play_again()
        self.begin = False

    def initialize_board(self):
        self.board = []
        self.apple_obj = []
        self.old_apple_cell = []

        for i in range(rows):
            for j in range(cols):
                self.board.append((i, j))

        for i in range(rows):
            self.canvas.create_line(
                i * size_of_board / rows, 0, i * size_of_board / rows, size_of_board,
            )

        for i in range(cols):
            self.canvas.create_line(
                0, i * size_of_board / cols, size_of_board, i * size_of_board / cols,
            )

    def initialize_snake(self):
        self.snake = []
        self.crashed = False
        self.snake_heading = "Right"
        self.last_key = self.snake_heading
        self.forbidden_actions = {}
        self.forbidden_actions["Right"] = "Left"
        self.forbidden_actions["Left"] = "Right"
        self.forbidden_actions["Up"] = "Down"
        self.forbidden_actions["Down"] = "Up"
        self.snake_objects = []
        for i in range(snake_initial_length):
            self.snake.append((i, 0))

    def play_again(self):
        self.canvas.delete("all")
        self.initialize_board()
        self.initialize_snake()
        self.place_apple()
        self.display_snake(mode="complete")
        self.begin_time = time.time()

    def mainloop(self):
        while True:
            self.window.update()
            if self.begin:
                if not self.crashed:
                    self.window.after(DELAY, self.update_snake(self.last_key))
                else:
                    self.begin = False
                    self.display_gameover()

    # ------------------------------------------------------------------
    # Drawing Functions:
    # The modules required to draw required game based object on canvas
    # ------------------------------------------------------------------
    def display_gameover(self):
        score = len(self.snake) - 3
        self.canvas.delete("all")
        score_text = "Scores \n"

        # put gif image on canvas
        # pic's upper left corner (NW) on the canvas is at x=50 y=10

        self.canvas.create_text(
            size_of_board / 2,
            3 * size_of_board / 8,
            font="cmr 40 bold",
            fill=Green_color,
            text=score_text,
        )
        score_text = str(score)
        self.canvas.create_text(
            size_of_board / 2,
            1 * size_of_board / 2,
            font="cmr 50 bold",
            fill=BLUE_COLOR,
            text=score_text,
        )
        time_spent = str(np.round(time.time() - self.begin_time, 1)) + 'sec'
        self.canvas.create_text(
            size_of_board / 2,
            3 * size_of_board / 4,
            font="cmr 20 bold",
            fill=BLUE_COLOR,
            text=time_spent,
        )
        score_text = "Hit 0 to play again \n"
        self.canvas.create_text(
            size_of_board / 2,
            15 * size_of_board / 16,
            font="cmr 20 bold",
            fill="gray",
            text=score_text,
        )

    def place_apple(self):
        # Place apple randomly anywhere except at the cells occupied by snake
        unoccupied_cels = set(self.board) - set(self.snake)
        self.apple_cell = random.choice(list(unoccupied_cels))
        row_h = int(size_of_board / rows)
        col_w = int(size_of_board / cols)
        x1 = self.apple_cell[0] * row_h
        y1 = self.apple_cell[1] * col_w
        x2 = x1 + row_h
        y2 = y1 + col_w
        self.apple_obj = self.canvas.create_rectangle(
            x1, y1, x2, y2, fill=RED_COLOR_LIGHT, outline=BLUE_COLOR,
        )

    def display_snake(self, mode=""):
        # Remove tail from display if it exists
        if self.snake_objects != []:
            self.canvas.delete(self.snake_objects.pop(0))
        if mode == "complete":
            for i, cell in enumerate(self.snake):
                # print(cell)
                row_h = int(size_of_board / rows)
                col_w = int(size_of_board / cols)
                x1 = cell[0] * row_h
                y1 = cell[1] * col_w
                x2 = x1 + row_h
                y2 = y1 + col_w
                self.snake_objects.append(
                    self.canvas.create_rectangle(
                        x1, y1, x2, y2, fill=BLUE_COLOR, outline=BLUE_COLOR,
                    )
                )
        else:
            # only update head
            cell = self.snake[-1]
            row_h = int(size_of_board / rows)
            col_w = int(size_of_board / cols)
            x1 = cell[0] * row_h
            y1 = cell[1] * col_w
            x2 = x1 + row_h
            y2 = y1 + col_w
            self.snake_objects.append(
                self.canvas.create_rectangle(
                    x1, y1, x2, y2, fill=BLUE_COLOR, outline=RED_COLOR,
                )
            )
            if self.snake[-1] == self.old_apple_cell:
                self.snake.insert(0, self.old_apple_cell)
                self.old_apple_cell = []
                tail = self.snake[0]
                row_h = int(size_of_board / rows)
                col_w = int(size_of_board / cols)
                x1 = tail[0] * row_h
                y1 = tail[1] * col_w
                x2 = x1 + row_h
                y2 = y1 + col_w
                self.snake_objects.insert(
                    0,
                    self.canvas.create_rectangle(
                        x1, y1, x2, y2, fill=BLUE_COLOR, outline=RED_COLOR
                    ),
                )
            self.window.update()

    # ------------------------------------------------------------------
    # Logical Functions:
    # The modules required to carry out game logic
    # ------------------------------------------------------------------
    def update_snake(self, key):
        # Check if it hit the wall or its own body
        tail = self.snake[0]
        head = self.snake[-1]
        if tail != self.old_apple_cell:
            self.snake.pop(0)

        #Change key
        if head[0] < 0: #Left of field
            key = "Right"            
        elif head[0] > cols - 1: #Right of field
            key = "Left"
        elif head[1] < 0: #Up of field
            key = "Down"
        elif head[1] > rows - 1: #Down of field
            key = "Up"
        else:
            key = key
            
        #Actions
        if key == "Left":
            self.snake.append((head[0] - 1, head[1]))
        elif key == "Right":
            self.snake.append((head[0] + 1, head[1]))
        elif key == "Up":
            self.snake.append((head[0], head[1] - 1))
        elif key == "Down":
            self.snake.append((head[0], head[1] + 1))

        head = self.snake[-1]
        #print('Head',head,sep=':')
        score_text=str(len(self.snake)-3)
        self.canvas.delete("rect")
        self.canvas.delete("time")
        self.canvas.delete("pause")
        self.canvas.create_text(
            size_of_board / 50,
            size_of_board / 50,
            font="cmr 15 bold",
            fill=BLUE_COLOR,
            text=score_text,
            tags="rect"
        )
        self.canvas.create_text(
            2*size_of_board / 50,
            3*size_of_board / 50,
            font="cmr 15 bold",
            fill=BLUE_COLOR,
            text=str(np.round(time.time() - self.begin_time, 1)),
            tags="time"
        )
        self.canvas.create_text(
            4*size_of_board / 50,
            5*size_of_board / 50,
            font="cmr 15 bold",
            fill=BLUE_COLOR,
            text=".=Pause",
            tags="pause"
        )     
##        print('Tail',tail,sep=':')
        if len(set(self.snake)) != len(self.snake): #Hit on body            
            self.crashed = True                           
        elif self.apple_cell == head: # Got the apple            
            self.old_apple_cell = self.apple_cell
            self.canvas.delete(self.apple_obj)
            self.place_apple()
            self.display_snake()
        elif head[0] < 0: #Left of field
            self.snake[-1] = (cols,head[1])            
        elif head[0] > cols - 1: #Right of field
            self.snake[-1] = (-1,head[1])
        elif head[1] < 0: #Up of field
            self.snake[-1] = (head[0],rows)
        elif head[1] > rows - 1: #Down of field
            self.snake[-1] = (head[0],-1)           
        else:
            self.snake_heading = key
            self.display_snake()

    def check_if_key_valid(self, key):
        valid_keys = ["Up", "Down", "Left", "Right"]
        if key in valid_keys and self.forbidden_actions[self.snake_heading] != key:
            return True
        else:
            return False

    def mouse_input(self, event):
        self.play_again()

    def pause(self, event):        
            time.sleep(1)
            pyautogui.alert('This displays some text with an OK button.')
        
    def key_input(self, event):
        if not self.crashed:
            key_pressed = event.keysym
            # Check if the pressed key is a valid key
            if self.check_if_key_valid(key_pressed):
                # print(key_pressed)
                self.begin = True
                self.last_key = key_pressed


game_instance = SnakeAndApple()
game_instance.mainloop()
