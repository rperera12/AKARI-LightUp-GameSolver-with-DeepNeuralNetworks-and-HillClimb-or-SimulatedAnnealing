
from numpy import number
import processing_py as psi
import setup as stp

# one cell width, pixel
CELL_WIDTH = 100
TEXT_SIZE = 24

class DrawBoard:

    def __init__(self, board):
        # set width for one cell on the board
        self.width = CELL_WIDTH
        # set row number
        self.row = len(board)
        # initialize graph board
        self.app = psi.App(self.row * self.width, self.row * self.width)
        # set text size
        self.text_size = TEXT_SIZE
        self.app.textSize(TEXT_SIZE)
        # get the board information
        self.board = board
    
    # draw a bulb
    def bulb(self, x, y):
        w = self.width
        self.app.fill(255,128,0)
        self.app.ellipse(x+int(w/2),y+int(w/2),int(w/2),int(w/2))

    
    # draw a black block, with number
    def black(self, x, y, number):
        # draw black cell first
        w = self.width
        self.app.fill(0,0,0)
        self.app.rect(x, y, w, w)
        # add text as white color
        if number < stp.CELL_BLACK_FIVE:
            self.app.fill(255)
            self.app.text(str(number), 
                            x + int(w/2) - int(self.text_size/2), 
                            y + int(w/2) + int(self.text_size/2))

    # show plain white cell
    def white(self, x, y):
        w = self.width
        self.app.fill(255)
        self.app.rect(x, y, w, w)

    
    # show a cell is lighted
    def light(self, x, y):
        w = self.width
        # yellow color as be lighted
        self.app.fill(255, 255, 153)
        self.app.rect(x, y, w, w)

    # show the cell banned bulb
    def ban(self, x, y):
        w = self.width
        # light blue color as be banned
        self.app.fill(102, 102, 255)
        self.app.rect(x, y, w, w)

    # draw a black block, with number
    def four_digit(self, x, y, number):
        # draw black cell first
        w = self.width
        # add text as white color
        if number < stp.CELL_BLACK_FIVE:
            self.app.fill(66, 96, 245)
            self.app.text(str(number), 
                            x + int(w/2) - int(self.text_size), 
                            y + int(w/2) + int(self.text_size))


    # show the game board
    def show(self, probs):
        x, y = 0, 0
        for i in range(len(self.board)):            
            for j in range(len(self.board[i])):
                v = self.board[i][j]
                if v <= stp.CELL_BLACK_FIVE:
                    self.black(x, y, v)
                else:
                    # white cell
                    self.white(x, y)
                    # cell has a bulb and lighted
                    if v == stp.CELL_BULB:
                        self.light(x, y)
                        self.bulb(x, y)
                    # lighted cells
                    elif v == stp.CELL_LIGHT:
                        self.light(x, y)
                    # bulb banned cells
                    elif v == stp.CELL_BULB_BAN:
                        self.ban(x, y)
                    
                    else:                      

                        index = probs["axis"].index([i,j])
                        prob = probs["prob"][index]
                        self.four_digit(x, y, prob)

            
                # move to right
                x += self.width

            # move down
            y += self.width 
            # reset to left edge
            x = 0 
        
        # show the frame
        self.app.redraw()


    