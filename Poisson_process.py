import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
 
#Simulation window parameters


def calculate_area_rectangle(xMin, xMax, yMin, yMax):
    xDelta=xMax-xMin
    yDelta=yMax-yMin # rectangle dimensions
    areaTotal=xDelta*yDelta
    
    return areaTotal, xDelta, yDelta
    
def Poisson_point_process(lambda0=100, xMin=0, xMax=1, yMin=0, yMax=1, show=True):
    #intensity (ie mean density) of the Poisson process

    areaTotal, xDelta, yDelta=calculate_area_rectangle(xMin, xMax, yMin, yMax)


    # Simulate Poisson point process
    numbPoints = scipy.stats.poisson( lambda0*areaTotal ).rvs()  # Poisson number of points
    xx = xDelta*scipy.stats.uniform.rvs(0,1,((numbPoints,1)))+xMin  # x coordinates of Poisson points
    yy = yDelta*scipy.stats.uniform.rvs(0,1,((numbPoints,1)))+yMin  # y coordinates of Poisson points
    
    # Plotting
    if show:
        plt.scatter(xx,yy, edgecolor='b', facecolor='none', alpha=0.5 )
        plt.xlabel("x"); plt.ylabel("y")
        plt.show()
    
    return xx, yy


def Poisson_point_process_cross(N_c1, N_total, lambda0=100, xMin=0, xMax=1, yMin=0, yMax=1, show=True):
    #intensity (ie mean density) of the Poisson process

    areaTotal, xDelta, yDelta=calculate_area_rectangle(xMin, xMax, yMin, yMax)


    # Simulate Poisson point process
    numbPoints = scipy.stats.poisson( lambda0*areaTotal ).rvs()  # Poisson number of points
    xx = xDelta*scipy.stats.uniform.rvs(0,1,((numbPoints,1)))+xMin  # x coordinates of Poisson points
    yy = yDelta*scipy.stats.uniform.rvs(0,1,((numbPoints,1)))+yMin  # y coordinates of Poisson points
    

    # choose points to be part of class c1 with probability N_c1/N_total
    c1mask = np.random.choice(a=[False, True], size=(xx.shape[0],), p=[N_c1/N_total, 1-N_c1/N_total])
    
     # randomly select exactly N_C1 points to be in class c1, 
#     k = N_C1
#     booleans = np.zeros(shape=(xx.shape[0]), dtype=bool) # Array with N_total False
#     booleans[:int(k / 100 * N_total)] = True  # Set the first k% of the elements to True
#     np.random.shuffle(booleans)  # Shuffle the array
    
    # Plotting
    if show:
        plt.scatter(xx[c1mask],yy[c1mask], edgecolor='b', facecolor='none', s=4 )
        plt.scatter(xx[np.logical_not(c1mask)],yy[np.logical_not(c1mask)], edgecolor='r', s=4, facecolor='none')
        plt.xlabel("x"); plt.ylabel("y")
        plt.show()
    
    return xx, yy, c1mask





