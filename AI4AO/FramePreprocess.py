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
        self.Extract_pupils = wfsParams["Extract_pupils"]
        self.Extract_pupils_pad = wfsParams["Extract_pupils_pad"]
        self.bin_factor = wfsParams["Bin_factor"]
        self.Nres = wfsParams["Nres"]
        self.Centering_noise = wfsParams["Center_noise"]
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

        frame = self.GetPupils(frame, isReference = True)

        if int(self.bin_factor) != 1:
            frame = self.BinImage(frame) * self.bin_factor ** 2

        self.normalization = torch.std(frame, dim=(-2, -1), keepdim=True)
        self.reference = frame

    def ProcessFrame(self, input_frame):

        frame = torch.clone(input_frame)

        frame = self.GetPupils(frame)

        if int(self.bin_factor) != 1:
            frame = self.BinImage(frame) * self.bin_factor ** 2

        if self.Substract_reference:
            frame = frame - self.reference
            frame = frame / self.normalization
        else:
            frame = frame - frame.mean(dim=(-2, -1), keepdim=True)
            frame = frame / frame.std(dim=(-2, -1), keepdim=True)

        return frame

    def GetPupils(self, images, isReference = False):

        # Ncrop = self.Nres + int(self.Extract_pupils_pad * self.bin_factor)
        # centers = np.copy(self.wfs.pupil_centers)  # // self.Bin_factor

        # if self.Centering_noise > 0 and not isReference:
        #     centers += np.random.randint(
        #         -self.Centering_noise, self.Centering_noise, centers.shape
        #     ) * int(self.bin_factor)
            
        # out = torch.zeros(
        #     (images.shape[0], centers.shape[0], Ncrop, Ncrop), device=self.device
        # )

        # for i, center in enumerate(centers):
        #     out[:, i] = images[
        #         ...,
        #         center[0] - Ncrop // 2 : center[0] + Ncrop // 2,
        #         center[1] - Ncrop // 2 : center[1] + Ncrop // 2,
        #     ]
        # return out

        B = images.shape[0]
        batch = np.arange(B)[:, None, None, None]

        centers = np.copy(self.wfs.pupil_centers)[None,...]
        pupil_noise = np.random.randint(-self.Centering_noise, self.Centering_noise,(B,*centers.shape[-2:]))

        if isReference:
            centers = centers + pupil_noise * 0
        else:
            centers = centers + pupil_noise
        

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
