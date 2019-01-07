import numpy as np
import math

def NearestNeigthborInterpolation(x, y, image):    
    i = math.floor((x - image.Origin[0] ) / image.Spacing[0] + 0.5 )
    j = math.floor((y - image.Origin[1] ) / image.Spacing[1] + 0.5 )
    val = image.Values[int(i), int(j)]
    return val  

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
    return minX, minY, maxX, maxY

def Transform(B, A, T, nullFill = True):
    """Transform image B into image A cordinate system using affine transformation matrix T.
    Input:
        A - Reference image of type Image
        B - Moving Image of type Image
        T - [3x3] Affine transformation matrix
    Output: 
        Image A' transformed into Bs cordinate system.
    """  
    invT = np.linalg.inv(T) ## TODO change to own inverse transform
    
    if not nullFill:
        tImg = np.zeros(A.Values.shape, dtype=float) # TODO how should it be initilized
    else:    
        tImg = np.full(A.Values.shape, None, dtype=float) # TODO how should it be initilized    
    (minX, minY, maxX, maxY) = getExtremeValues(B)
    for i in range(tImg.shape[0]):
        for j in range(tImg.shape[1]):
            x = i * A.Spacing[0] + A.Origin[0]
            y = j * A.Spacing[1] + A.Origin[1]
            pos = np.dot(invT, np.array([x,y,1], dtype=float, copy=True))
            #print("x: {0}, y:{1}, T: {2}".format(x,y, invT))
            #print(pos)
            if not (minX <= pos[0] <= maxX and minY <= pos[1] <= maxY):                
                continue            
            val = BilinearInterpolate(pos[0], pos[1], B)
            tImg[i,j] = val
    return Image(tImg, A.Spacing, A.Origin)

# MSE
def MSE_residual(A, B):
    """Calculate the residual for the intensity values in image A and B on domain A
    Input:
        A - Reference image of type Image
        B - Moving Image of type Image
    Output: 
        an array with the residual values
    """
    residual = []
    for i in range(A.Values.shape[0]):
        for j in range(A.Values.shape[1]):
            if A.Values[i,j] != A.Values[i,j]: # Null check                
                continue                
            x = i * A.Spacing[0] + A.Origin[0]
            y = j * A.Spacing[1] + A.Origin[1]            
            f_r = BilinearInterpolate(x=x, y=y, image=B)
            residual.append(A.Values[i,j] - f_r)
    return np.array([residual], dtype=float).T

def MSE_cost(residual):
    """Calculate the mean square error (MSE) between intensity values of image A and B
    Input:
        residual - the residual in every point of the given domain (Ai - Bi)
    Output: 
        the cost
    """
    return np.sum(residual**2)/residual.size

# Normalized Cross correlation
def NCC(A, B):
    """Calculate Normalized Cross Correlation between to images A and B using Fast Fourier Transform
    Input:
        A - Reference image of type Image
        B - Moving Image of type Image
    Output: 
        (ncc, idxY, idxX, x, y)
        ncc - Normalized cross correlation
        i - index in Refrence image x-direction where the highest NCC appears
        j - index in Refrence image y-direction where the highest NCC appears
        x - x value in Refrence image where the highest NCC appears
        y - y value in Refrence image where the highest NCC appears
    """
    assert np.abs(np.mean(A.Values)) < 1e-10 # Assert mean has already been removed
    assert np.abs(np.mean(B.Values)) < 1e-10

    padWidthX = int((B.Values.shape[1]-1)/2)
    padWidthY = int((B.Values.shape[0]-1)/2)
    gPad = np.pad(A.Values, [(padWidthY, ), (padWidthX, )], mode='constant')
    G = np.fft.fft2(gPad)
    F = np.fft.fft2(B.Values, G.shape)    
    FG = np.conjugate(F)*G
    ncc = np.real(np.fft.ifft2(FG))
    j, i = np.unravel_index(np.argmax(ncc), ncc.shape)  # find the match
    x = i * A.Spacing[0] + A.Origin[0]
    y = j * A.Spacing[1] + A.Origin[1]
    return ncc, i, j, x, y


class Image:
    def __init__(self, values, spacing, origin):
        self.Values = values
        self.Spacing = spacing
        self.Origin = origin