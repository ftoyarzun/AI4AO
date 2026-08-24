import torch # type: ignore[import]
import torch.nn as nn # type: ignore[import]
import numpy as np
import torch.nn.functional as F # type: ignore[import]


class FramePreprocess:
    def __init__(self, wfsParams, wfs, device):
        self.reference = 1.0
        self.normalization = 1.0

        self.device = device
        
        self.Substract_reference = wfsParams["Substract_Reference"]
        self.bin_factor = wfsParams["Bin_factor"]
        self.Nres = wfsParams["Nres"]
        self.Centering_noise = wfsParams["Center_noise"] * wfsParams["Bin_factor"]
        self.Extract_pupils_pad = wfsParams["Extract_pupils_pad"] * wfsParams["Bin_factor"]
        self.wfs = wfs

        self.Ncrop = self.Nres + self.Extract_pupils_pad

        self.yy, self.xx = np.meshgrid(
            np.arange(self.Ncrop),
            np.arange(self.Ncrop),
            indexing="ij"
        )

        self.yy = self.yy[None, None]
        self.xx = self.xx[None, None]

    def ProcessReference(self, reference_frame):

        frame = torch.clone(reference_frame)
        frame = frame.unsqueeze(0)

        frame = self.GetPupils(frame, add_pupil_noise = False)

        if self.bin_factor != 1:
            frame = self.BinImage(frame) * self.bin_factor ** 2

        self.normalization = torch.std(frame, dim=(-2, -1), keepdim=True)
        self.reference = frame

    def GetPupilNormalizationFromDataset(self, dataset, wfs, n_iter = 1000):

        for i in range(n_iter):
            batch = dataset[0]
            phaseGT = batch["phase"]
            wfs_frames = wfs(phaseGT)
            if self.bin_factor != 1:
                wfs_frames = self.BinImage(wfs_frames) * self.bin_factor ** 2
            pupils = self.GetPupils(wfs_frames)
            if i == 0:
                normalization = torch.std(pupils, dim=(-2, -1), keepdim=True)
            else:
                normalization += torch.std(pupils, dim=(-2, -1), keepdim=True)

        self.normalization = normalization / n_iter

    def ProcessFrame(self, input_frame, add_pupil_noise = True):

        frame = torch.clone(input_frame)

        frame = self.GetPupils(frame, add_pupil_noise)

        if self.bin_factor != 1:
            frame = self.BinImage(frame) * self.bin_factor ** 2

        if self.Substract_reference:
            frame = frame - self.reference
            frame = frame / self.normalization
        else:
            frame = frame - frame.mean(dim=(-2, -1), keepdim=True)
            frame = frame / self.normalization

        return frame

    def GetPupils(self, images, add_pupil_noise = False):

        B = images.shape[0]
        batch = np.arange(B)[:, None, None, None]

        centers = np.copy(self.wfs.pupil_centers)[None,...]
        pupil_noise = np.random.randint(-self.Centering_noise, self.Centering_noise,(B,*centers.shape[-2:]))

        if add_pupil_noise:
            centers = centers + pupil_noise
        else:
            centers = centers + pupil_noise * 0
        

        yy = self.yy + centers[:,:, 0][..., None, None] - self.Ncrop//2   # [B,C,N,N]
        xx = self.xx + centers[:,:, 1][..., None, None] - self.Ncrop//2   # [B,C,N,N]

        patches = images[batch, yy, xx]
        return patches

    
    def BinImage(self, Image, bin_factor = None):
        
        if bin_factor is None: 
            bin_factor = bin_factor = self.bin_factor

        # output_size = int(Image.shape[-1] / bin_factor)
        out = F.interpolate(Image, scale_factor = 1 / bin_factor)
        return out
