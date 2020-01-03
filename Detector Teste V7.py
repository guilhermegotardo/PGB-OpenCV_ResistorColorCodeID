
# Python program for Detection of a  
# specific color(blue here) using OpenCV with Python 
import cv2 
import numpy as np  
import matplotlib.pyplot as plt
# Webcamera no 0 is used to capture the frames 

# This drives the program into an infinite loop. 
       
    # Captures the live stream frame-by-frame 
    #IMG_9563
    #IMG_9570
    #IMG_9571
sizeImagex = 300
sizeImagey = 350

frame = cv2.imread('IMG_9563.JPG')

#frame = cv2.resize(frame,(1024,768), interpolation = cv2.INTER_CUBIC)
frame = cv2.resize(frame,(sizeImagey,sizeImagex), interpolation = cv2.INTER_CUBIC)
    # Converts images from BGR to HSV 
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 

lower_yellow = np.array([23,170,117]) 
upper_yellow = np.array([25,216,145])

lower_brown = np.array([4,120,41]) 
upper_brown = np.array([12,181,70])  
  
lower_black = np.array([14,0,15]) 
upper_black = np.array([170,175,42])  

lower_orange = np.array([8,168,115]) 
upper_orange = np.array([14,225,200])

lower_red = np.array([0,168,85]) 
upper_red = np.array([5,240,120])

lower_green = np.array([53,139,49]) 
upper_green = np.array([70,255,86])

lower_violet = np.array([131,107,56]) 
upper_violet = np.array([134,159,107])

lower_gray = np.array([109,18,68]) 
upper_gray = np.array([175,70,83])

# Here we are defining range of bluecolor in HSV 
# This creates a mask of blue coloured  
# objects found in the frame. 


#mask = cv2.inRange(hsv, lower_yellow, upper_yellow) 
#mask = cv2.inRange(hsv, lower_brown, upper_brown) 
#mask = cv2.inRange(hsv, lower_black, upper_black) 
#mask = cv2.inRange(hsv, lower_orange, upper_orange)
#mask = cv2.inRange(hsv, lower_red, upper_red)
#mask = cv2.inRange(hsv, lower_green, upper_green)
#mask = cv2.inRange(hsv, lower_violet, upper_violet)
#mask = cv2.inRange(hsv, lower_gray, upper_gray)
codigoCorPos = np.zeros(4)
codigoCor = np.zeros(4)
auxCodigoCor = 0

#IDENTIFICANDO PRETO
mask = cv2.inRange(hsv, lower_black, upper_black) 
res = cv2.bitwise_and(frame,frame, mask = mask) 
aux = 0
auxPosicao = np.zeros(1000)
auxPosicao2 = 0

for i in range(300):
    for j in range(350):
        if mask[i][j] == 255:
            aux+=1
            auxPosicao[aux] = i
            if aux>150:
                break

for i in range(aux):
    auxPosicao2 += auxPosicao[aux]
    
if aux > 0:
    auxPosicao2 = auxPosicao2/aux

if aux > 100:
    print("Possui PRETO em : ") 
    print(auxPosicao2)
    codigoCor[auxCodigoCor] = 0
    codigoCorPos[auxCodigoCor] = auxPosicao2
    auxCodigoCor+=auxCodigoCor+1

  #IDENTIFICANDO MARROM
mask = cv2.inRange(hsv, lower_brown, upper_brown) 
res = cv2.bitwise_and(frame,frame, mask = mask) 
aux = 0
auxPosicao = np.zeros(1000)
auxPosicao2 = 0

for i in range(sizeImagex):
    for j in range(sizeImagey):
        if mask[i][j] == 255:
            aux+=1
            auxPosicao[aux] = i
            if aux>150:
                break

for i in range(aux):
    auxPosicao2 += auxPosicao[aux]

if aux > 0:    
    auxPosicao2 = auxPosicao2/aux

if aux > 100:
    print("Possui MARROM em : ") 
    print(auxPosicao2)  
    codigoCor[auxCodigoCor] = 1
    codigoCorPos[auxCodigoCor] = auxPosicao2
    auxCodigoCor+=auxCodigoCor+1


  #IDENTIFICANDO VERMELHO
mask = cv2.inRange(hsv, lower_red, upper_red)
res = cv2.bitwise_and(frame,frame, mask = mask) 
aux = 0
auxPosicao = np.zeros(1000)
auxPosicao2 = 0

for i in range(sizeImagex):
    for j in range(sizeImagey):
        if mask[i][j] == 255:
            aux+=1
            auxPosicao[aux] = i
            if aux>150:
                break

for i in range(aux):
    auxPosicao2 += auxPosicao[aux]
 
if aux > 0:    
    auxPosicao2 = auxPosicao2/aux

if aux > 100:
    print("Possui VERMELHO em : ") 
    print(auxPosicao2)  
    codigoCor[auxCodigoCor] = 2
    codigoCorPos[auxCodigoCor] = auxPosicao2
    auxCodigoCor+=auxCodigoCor+1


  #IDENTIFICANDO LARANJA
mask = cv2.inRange(hsv, lower_orange, upper_orange)
res = cv2.bitwise_and(frame,frame, mask = mask) 
aux = 0
auxPosicao = np.zeros(1000)
auxPosicao2 = 0

for i in range(sizeImagex):
    for j in range(sizeImagey):
        if mask[i][j] == 255:
            aux+=1
            auxPosicao[aux] = i
            if aux>150:
                break

for i in range(aux):
    auxPosicao2 += auxPosicao[aux]
    
if aux > 0:
    auxPosicao2 = auxPosicao2/aux

if aux > 100:
    print("Possui LARANJA em : ") 
    print(auxPosicao2)  
    codigoCor[auxCodigoCor] = 3
    codigoCorPos[auxCodigoCor] = auxPosicao2
    auxCodigoCor+=auxCodigoCor+1

  #IDENTIFICANDO AMARELO
mask = cv2.inRange(hsv, lower_yellow, upper_yellow) 
res = cv2.bitwise_and(frame,frame, mask = mask) 
aux = 0
auxPosicao = np.zeros(1000)
auxPosicao2 = 0

for i in range(sizeImagex):
    for j in range(sizeImagey):
        if mask[i][j] == 255:
            aux+=1
            auxPosicao[aux] = i
            if aux>150:
                break

for i in range(aux):
    auxPosicao2 += auxPosicao[aux]

if aux > 0:
    auxPosicao2 = auxPosicao2/aux

if aux > 100:
    print("Possui AMARELO em : ") 
    print(auxPosicao2)  
    codigoCor[auxCodigoCor] = 4
    codigoCorPos[auxCodigoCor] = auxPosicao2
    auxCodigoCor+=auxCodigoCor+1



  #IDENTIFICANDO VERDE
mask = cv2.inRange(hsv, lower_green, upper_green)
res = cv2.bitwise_and(frame,frame, mask = mask) 
aux = 0
auxPosicao = np.zeros(1000)
auxPosicao2 = 0

for i in range(sizeImagex):
    for j in range(sizeImagey):
        if mask[i][j] == 255:
            aux+=1
            auxPosicao[aux] = i
            if aux>150:
                break

for i in range(aux):
    auxPosicao2 += auxPosicao[aux]

if aux > 0:
    auxPosicao2 = auxPosicao2/aux

if aux > 100:
    print("Possui VERDE em : ") 
    print(auxPosicao2)  
    codigoCor[auxCodigoCor] = 5
    codigoCorPos[auxCodigoCor] = auxPosicao2
    auxCodigoCor+=auxCodigoCor+1


  #IDENTIFICANDO ROXO
mask = cv2.inRange(hsv, lower_violet, upper_violet) 
res = cv2.bitwise_and(frame,frame, mask = mask) 
aux = 0
auxPosicao = np.zeros(1000)
auxPosicao2 = 0

for i in range(sizeImagex):
    for j in range(sizeImagey):
        if mask[i][j] == 255:
            aux+=1
            auxPosicao[aux] = i
            if aux>150:
                break

for i in range(aux):
    auxPosicao2 += auxPosicao[aux]

if aux > 0:
    auxPosicao2 = auxPosicao2/aux

if aux > 100:
    print("Possui ROXO em : ") 
    print(auxPosicao2)  
    codigoCor[auxCodigoCor] = 7
    codigoCorPos[auxCodigoCor] = auxPosicao2
    auxCodigoCor+=auxCodigoCor+1



  #IDENTIFICANDO CINZA
mask = cv2.inRange(hsv, lower_gray, upper_gray)
res = cv2.bitwise_and(frame,frame, mask = mask) 
aux = 0
auxPosicao = np.zeros(1000)
auxPosicao2 = 0

for i in range(sizeImagex):
    for j in range(sizeImagey):
        if mask[i][j] == 255:
            aux+=1
            auxPosicao[aux] = i
            if aux>150:
                break

for i in range(aux):
    auxPosicao2 += auxPosicao[aux]

if aux > 0:
    auxPosicao2 = auxPosicao2/aux

if aux > 100:
    print("Possui CINZA em : ") 
    print(auxPosicao2)  
    codigoCor[auxCodigoCor] = 8
    codigoCorPos[auxCodigoCor] = auxPosicao2
    auxCodigoCor+=auxCodigoCor+1

codigoOrdenado = np.zeros(4)

if  codigoCorPos[0] > codigoCorPos[1] and codigoCorPos[0] > codigoCorPos[2] and codigoCorPos[1] > codigoCorPos[2]:
    codigoOrdenado[0] = codigoCor[0] 
    codigoOrdenado[1] = codigoCor[1] 
    codigoOrdenado[2] = codigoCor[2]

elif codigoCorPos[0] > codigoCorPos[1] and codigoCorPos[0] > codigoCorPos[2] and codigoCorPos[2] > codigoCorPos[1]:
    codigoOrdenado[0] = codigoCor[0] 
    codigoOrdenado[1] = codigoCor[2] 
    codigoOrdenado[2] = codigoCor[1]

elif codigoCorPos[1] > codigoCorPos[0] and codigoCorPos[1] > codigoCorPos[2] and codigoCorPos[0] > codigoCorPos[2]:
    codigoOrdenado[0] = codigoCor[1] 
    codigoOrdenado[1] = codigoCor[0] 
    codigoOrdenado[2] = codigoCor[2]
    
 elif codigoCorPos[1] > codigoCorPos[0] and codigoCorPos[1] > codigoCorPos[2] and codigoCorPos[2] > codigoCorPos[0]:
    codigoOrdenado[0] = codigoCor[1] 
    codigoOrdenado[1] = codigoCor[0] 
    codigoOrdenado[2] = codigoCor[2]


cv2.imshow('frame',frame) 
cv2.imshow('mask',mask) 
cv2.imshow('res',res) 

#cv2.waitKey(0); cv2.destroyAllWindows(); 





#cv2.waitKey(1)

  


