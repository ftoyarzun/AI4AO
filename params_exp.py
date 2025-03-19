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
"RON" : [0, 3]     ,                                                                # Read-out noise in photons per pixel per frame
"Nzernike" : 50 ,                                                                  # Number of Zernike modes to reconstruct
"Nactuator" : 10,                                                                  # Number of actuators across the diameter of the DM
"useNoise" : True,
"InitParam" : [0.78, 0.78],
"MaskType" : 'Free'
},

)  

                                                               
## Atmosphere parameters
AtmosParams = dict(
    {                                                                   
"r0" : [0.06, 0.12],           # warning : min and max range taken in log10 space
"L0" : [20,30.0000],                                                                    # Inner scale (m)                                                                       # Outter scale (m)
"Nphases" : 16,
}
)                             



## Loop parameters
LoopParams = dict(
{
"loopFrequency" : 1000,
"delayFrames" : 1,
"windSpeedVector" : [5,10],
"levelOfCorrection" : [0., 1.0],
}
)

## Train parameters
TrainParams = dict(
{
"lro" : 0.001,
"lrn" :0.001,
"TrainRunNb" : 1000,
"TestRunNb" : 10,
}
)
