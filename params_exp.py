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
{ "Nres": 78, #78,                                                                      # Number of pixels in the aperture of the telescope
"sampling" : 240/78 ,#240/78,                                                                    # Zero-padding factor (2 is Shannon)
"D" : 1.52 ,                                                                        # Telescope diameter (m)
"Nphotons" : [4, 6],                                                               # Log range of number of photons in measurement    
"RON" : [0.6, 0.6]     ,                                                                # Read-out noise in photons per pixel per frame
"Nzernike" : 50 ,                                                                  # Number of Zernike modes to reconstruct
"Nactuator" : 17,                                                                  # Number of actuators across the diameter of the DM
"useNoise" : True,
"InitParam" : [1.2, 1.2, 1000],
"MaskType" : "BiOEdge",
"Reconstruction": "SimpleNet",
"FrameBufferLength": 1,
"beamSplitProportionForWFSDetector": 1.,
"ModalBasis" : "Zernike"
}
)  

                                                               
## Atmosphere parameters
AtmosParams = dict(
    {                                                                   
"r0" : [0.05, 0.2],           
"L0" : [20,30.0000],                                                                    # Inner scale (m)                                                                       # Outter scale (m)
"Nphases" : 16,
"Layers" : [1,2]
}
)                             



## Loop parameters
LoopParams = dict(
{
"loopFrequency" : 1000,
"delayFrames" : 1,
"windSpeedVector" : [-10,10],
"levelOfCorrection" : [0., 1.],
}
)

## Train parameters
TrainParams = dict(
{
"lro" : 1e-3,
"lrn" : 1e-3,
"TrainRunNb" : 5000,
"TestRunNb" : 10,
}
)
