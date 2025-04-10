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
"sampling" : 2,                                                                    # Zero-padding factor (2 is Shannon)
"D" : 1.5 ,                                                                        # Telescope diameter (m)
"Nphotons" : [4, 6],                                                               # Log range of number of photons in measurement    
"RON" : [1, 2]     ,                                                                # Read-out noise in photons per pixel per frame
"Nzernike" : 50 ,                                                                  # Number of Zernike modes to reconstruct
"Nactuator" : 10,                                                                  # Number of actuators across the diameter of the DM
"useNoise" : True,
"InitParam" : [0.78*2, 0.78*2],
"MaskType" : 'FreePhaseTransmision'
},

)  

                                                               
## Atmosphere parameters
AtmosParams = dict(
    {                                                                   
"r0" : [0.05, 0.2],           
"L0" : [20,30.0000],                                                                    # Inner scale (m)                                                                       # Outter scale (m)
"Nphases" : 32,
"Layers" : 5
}
)                             



## Loop parameters
LoopParams = dict(
{
"loopFrequency" : 500,
"delayFrames" : 1,
"windSpeedVector" : [-10,10],
"levelOfCorrection" : [0., 1.],
}
)

## Train parameters
TrainParams = dict(
{
"lro" : 0.0001,
"lrn" : 0.0001,
"TrainRunNb" : 3000,
"TestRunNb" : 10,
}
)
