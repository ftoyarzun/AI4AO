# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 15:28:47 2025

@author: foyarzun
"""

mask_types_list = ["FreePhase",
                   "FreePhaseTransmision",
                   "FreeTransmision",
                   "Pyramid",
                   "Zernike",
                   "ModalMask",
                   "FullyFreePhase",
                   "BiOEdge",
                   "IBiOEdge",
                   "FullyFreeTransmision",
                   "Papyrus",
                   "DoublePyramid",
                   "DoubleZernike"
                   ]

param_needed_mask_list = ["Pyramid", "Zernike", "FullyFreePhase", "BiOEdge", "Papyrus"]

double_transmision_masks = ["BiOEdge", 
                            "FullyFreeTransmision"]

reconstruction_types_list = ["Linear", "SimpleNet", "DataFusion", "Papyrus", "PapyrusPhase", "VGGNet", "UNet", "Transformer", 'Papyrus2ndstage']

basis_list = ["Zernike", "Papyrus_KL", "Papyrus_Zernike", "Papyrus_Zonal", "Oziriis_KL", "Oziriis_Zonal"]


