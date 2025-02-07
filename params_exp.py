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
"sampling" : 6,                                                                    # Zero-padding factor (2 is Shannon)
"D" : 1.0 ,                                                                        # Telescope diameter (m)
"Nphotons" : 1e7  ,                                                                # Number of photons in measurement    
"RON" : 0    ,                                                                     # Read-out noise in photons per pixel per frame
"Nzernike" : 50 ,                                                                  # Number of Zernike modes to reconstruct
"Nactuator" : 10,                                                                  # Number of actuators across the diameter of the DM
"useNoise" : True,
},

)  

                                                               
## Atmosphere parameters
AtmosParams = dict(
    {                                                                   
"r0" : [-0.8239,-0.8239],           # warning : min and max range taken in log10 space
"L0" : [20,20.0001],                                                                    # Inner scale (m)                                                                       # Outter scale (m)
"Nphases" : 31,
}
)                             



## Loop parameters
LoopParams = dict(
{
"loopFrequency" : 1000,
"delayFrames" : 1,
"windSpeedVector" : [5,10],
"levelOfCorrection" : 1,
}
)

## Train parameters
TrainParams = dict(
{
"lro" : 0.01,
"lrn" :1,
"TrainRunNb" : 1000,
"TestRunNb" : 10,
}
)
