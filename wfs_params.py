WFSParams = dict(
        {
            "D": 7.92223,
            "c_obs": 2.59767,
            "nSubap": 32,
            "Nres": 320,
            "wavelength": 658e-9,
            "pixel_scale": 0.56,
            "Nphotons": [4.826, 6.128],
            "RON": [0.7, 2.],
            "useNoise": True,
            "beamSplitProportionForWFSDetector": 1.0,
            "lightRatio": 1.0,
            "binning_factor": 1,
            "shannon_sampling": False,
            "n_pixel_per_subap": 8,
            "half_pixel_shift": False,
            "Nzernike": 60,
            "Nactuator": 17,
            "ModalBasis": "Zernike",
            "Reconstruction": "Linear",
            "QE": 0.8,
            "FWC": 30000,
            "gain": 1.0,
            "bits": 16,
            "sensor": "CMOS",
            "darkCurrent": 0.006,
            "applyDigitalization": True,
            "exposureTime": 2e-3,
            "backgroundRange": [5e3, 5e4]
        }
    )

AtmosParams = dict(
        {
            "r0": [0.118, 0.118],
            "L0": [20., 20.],
            "Nphases": 1,
            "Layers": [7, 7],
            "f_slope": 11.0 / 3.0
        }
    )

LoopParams = dict(
        {
            "loopFrequency": 1000,
            "delayFrames": 1,
            "windSpeedVector": [-10, 10],
            "levelOfCorrection": [0.0, 1.0]
        }
    )
