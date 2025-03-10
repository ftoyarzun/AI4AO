#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 14:25:13 2025

@author: ptrouve


"""
import numpy as np
#%% Set general parameter


## WFS and Telescope parameters
WFSParams = dict(
{ "Nres": 50,                                                                      # Number of pixels in the aperture of the telescope
"sampling" : 4,                                                                    # Zero-padding factor (2 is Shannon)
"D" : 1.5 ,                                                                        # Telescope diameter (m)
"Nphotons" : [3, 7],                                                               # Log range of number of photons in measurement    
"RON" : [0, 2]    ,                                                                # Read-out noise in photons per pixel per frame
"Nzernike" : 50 ,                                                                  # Number of Zernike modes to reconstruct
"Nactuator" : 20,                                                                  # Number of actuators across the diameter of the DM
"useNoise" : True,
"InitParam" : [1.0, 1.0, 6.0],
"MaskType" : 'Zernike'
},

)  

                                                               
## Atmosphere parameters
AtmosParams = dict(
    {                                                                   
"r0" : [-1.2, -0.5],           # warning : min and max range taken in log10 space
"L0" : [20,30.0000],                                                                    # Inner scale (m)                                                                       # Outter scale (m)
"Nphases" : 64,
}
)                             



## Loop parameters
LoopParams = dict(
{
"loopFrequency" : 1000,
"delayFrames" : 1,
"windSpeedVector" : [5,10],
"levelOfCorrection" : [0.999, 1.0],
}
)

## Train parameters
TrainParams = dict(
{
"lro" : 0.01,
"lrn" :0.01,
"TrainRunNb" : 10000,
"TestRunNb" : 10,
}
)
