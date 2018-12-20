import numpy as np
import math

def NearestNeigthborInterpolation(x, y, image):    
    i = math.floor((x - image.Origin[0] ) / image.Spacing[0] + 0.5 )
    j = math.floor((y - image.Origin[1] ) / image.Spacing[1] + 0.5 )
    val = image.Values[int(i), int(j)]
    return val

def bi(x, y, image):    
    i1 = math.floor((x - image.Origin[0] ) / image.Spacing[0])
    x1 = i1 * image.Spacing[0] + image.Origin[0]
    i2 = i1 + 1
    x2 = i2 * image.Spacing[0] + image.Origin[0]
    j1 = math.floor((y - image.Origin[1] ) / image.Spacing[1])
    y1 = j1 * image.Spacing[1] + image.Origin[1]
    j2 = j1 + 1
    y2 = j2 * image.Spacing[1] + image.Origin[1]
    if i1 >= image.Values.shape[0]:
        i1 = image.Values.shape[0] - 1
    if i2 >= image.Values.shape[0]:
        i2 = image.Values.shape[0] - 1
    if j1 >= image.Values.shape[1]:
        j1 = image.Values.shape[1] - 1
    if j2 >= image.Values.shape[1]:
        j2 = image.Values.shape[1] - 1
    
    #print("X: {0}, X1: {1}, X2: {2}".format(x, x1, x2))
    #print("Y: {0}, Y1: {1}, Y2: {2}".format(y, y1, y2))
    #print("I1: {0}, I2: {1}".format(i1, i2))
    #print("J1: {0}, J2: {1}".format(j1, j2))
    #print("xShape: {0}, yShape: {1}".format(image.Values.shape[0], image.Values.shape[1]))
    f11 = image.Values[i1, j1]
    f12 = image.Values[i1, j2]
    f21 = image.Values[i2, j1]
    f22 = image.Values[i2, j2]
    #print("V_i1j1: {0}, V_i1j2: {1}, V_i2j1 : {2}, V_i1j2: {3}".format(f11, f12, f21, f22))
    Q = np.array([[f11, f12], [f21, f22]], dtype=float)
    xdiff = np.array([x2-x, x-x1], dtype=float)
    ydiff = np.array([y2-y, y-y1], dtype=float)
    fxy = 1/((x2 - x1)*(y2-y1)) * xdiff.dot(Q).dot(ydiff)
    #if fxy == np.Inf:
    #    print("ERROR: fxy:{0},x2:{1},x1:{2},y2:{3},y1:{4},i1:{5},i2:{6},j1:{7},j2:{8},x:{9},y:{10}".format(fxy, x2, x1, y2, y1, i1, i2, j1, j2, x, y))    
    return fxy    

def BilinearInterpolate(x, y, image):
    i1 = math.floor((x - image.Origin[0] ) / image.Spacing[0])
    x1 = i1 * image.Spacing[0] + image.Origin[0]
    i2 = i1 + 1    
    j1 = math.floor((y - image.Origin[1] ) / image.Spacing[1])
    y1 = j1 * image.Spacing[1] + image.Origin[1]  
    j2 = j1 + 1    
    a = x - x1
    b = y - y1
    #print("X:{0},X1:{1},Y:{2},Y1:{3}".format(x, x1, y, y1))

    if i1 >= image.Values.shape[0]:
        i1 = image.Values.shape[0] - 1 
    if i2 >= image.Values.shape[0]:
        i2 = image.Values.shape[0] - 1
    if j1 >= image.Values.shape[1]:
        j1 = image.Values.shape[1] - 1
    if j2 >= image.Values.shape[1]:
        j2 = image.Values.shape[1] - 1

    vi1j1 = image.Values[i1,j1]
    if vi1j1 != vi1j1:
        vi1j1 = 0
    vi2j1 = image.Values[i2, j1]
    if vi2j1 != vi2j1:
        vi2j1 = 0
    vi2j2 = image.Values[i2,j2]
    if vi2j2 != vi2j2:
        vi2j2 = 0
    vi1j2 = image.Values[i1, j2]
    if vi1j2 != vi1j2:
        vi1j2 = 0
    return (1-a) * (1-b) * vi1j1 + a * (1 - b) * vi2j1 + a * b * vi2j2 + (1 - a) * b * vi1j2
    

def getExtremeValues(image):
    pixelsX = image.Values.shape[0]
    pixelsY = image.Values.shape[1]
    minX = image.Origin[0]
    minY = image.Origin[1]
    maxX = minX + pixelsX * image.Spacing[0] - 1
    maxY = minY + pixelsY * image.Spacing[1] - 1
    #print("MinX: {0}".format(minY))
    #print("MinY: {0}".format(minX))
    #print("MaxX: {0}".format(maxX))
    #print("MaxY: {0}".format(maxY))
    return minX, minY, maxX, maxY

def Transform(image, refImg, T, nullFill = True):        
    invT = np.linalg.inv(T) ## TODO change to own inverse transform
    
    if not nullFill:
        tImg = np.zeros(refImg.Values.shape, dtype=float) # TODO how should it be initilized
    else:    
        tImg = np.full(refImg.Values.shape, None, dtype=float) # TODO how should it be initilized    
    (minX, minY, maxX, maxY) = getExtremeValues(image)
    for i in range(tImg.shape[0]):
        for j in range(tImg.shape[1]):
            x = i * refImg.Spacing[0] + refImg.Origin[0]
            y = j * refImg.Spacing[1] + refImg.Origin[1]
            pos = np.dot(invT, np.array([x,y,1], dtype=float, copy=True))
            #print("x: {0}, y:{1}, T: {2}".format(x,y, invT))
            #print(pos)
            if not (minX <= pos[0] <= maxX and minY <= pos[1] <= maxY):                
                continue            
            val = BilinearInterpolate(pos[0], pos[1], image)
            tImg[i,j] = val
    return Image(tImg, refImg.Spacing, refImg.Origin)

class Image:
    def __init__(self, values, spacing, origin):
        self.Values = values
        self.Spacing = spacing
        self.Origin = origin