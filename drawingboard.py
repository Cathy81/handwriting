import pygame
import numpy as np
import recognition as hwr
import dataprep as prep

SIZE=84
TEXTFIELD=50
RED = (255, 0, 0)
#size=126
digitSize=int(SIZE / 3) # 28
data = np.zeros((SIZE, SIZE), dtype=np.uint8)
digit=np.zeros((int(SIZE / 3), int(SIZE / 3)), dtype=np.uint8)
screen = pygame.display.set_mode((SIZE, SIZE))

beginDrawing = False
last_pos = (0, 0)
color = (255, 255, 0)
radius = 1

def drawline(surf, start, end, color, radius=2):
    dx = end[0]-start[0]
    dy = end[1]-start[1]
    distance = max(abs(dx), abs(dy))
    for i in range(distance):
        x = int( start[0]+float(i)/distance*dx)
        y = int( start[1]+float(i)/distance*dy)
        pygame.draw.circle(surf, color, (x, y), radius)
        if y>=SIZE:
            y= SIZE - 1
        elif x>=SIZE:
            x= SIZE - 1
        elif x<0:
            x=0
        elif y<0:
            y=0
        data[y][x]=250
        for i in range(-5,5):
            if x + i < SIZE and x+i>=0:
                for j in range(-5,5):
                    if y+j <SIZE and y+j>=0:
                        data[y+j][x+i] = 250
try:
    while True:
        event = pygame.event.wait()
        if event.type == pygame.QUIT:
            raise StopIteration
        if event.type == pygame.MOUSEBUTTONDOWN:
            pygame.draw.circle(screen, color, event.pos, radius)
            beginDrawing = True
        if event.type == pygame.MOUSEBUTTONUP:
            beginDrawing = False
        if event.type == pygame.MOUSEMOTION:
            if beginDrawing:
               # pygame.draw.circle(screen, color, event.pos, radius)
                drawline(screen, event.pos, last_pos,  color, radius)
            last_pos = event.pos
        pygame.display.flip()
except StopIteration:
#step1: recoganize the handwriting digit by reshaping the array data, and call the function
# digitRecog(digit) to print the output, whiich is the recognized digit.
    digit=data.reshape([digitSize, SIZE // digitSize, digitSize, SIZE // digitSize]).mean(3).mean(1)
    print(digit)
    print(digit.shape)
    prep.showImg(digit)
    hwr.digitRecog(digit)
#"*** YOUR CODE HERE ***"


pygame.quit()