import torch # type: ignore[import]
import torch.nn as nn # type: ignore[import]
import numpy as np
import torch.nn.functional as F # type: ignore[import]


class FramePreprocess:
    def __init__(self, wfsParams, atmosParams, device, maskManager):
        self.reference = 1.0
        self.normalization = 1.0

        self.maskManager = maskManager

        self.device = device
        
        self.Substract_reference = wfsParams["Substract_Reference"]
        self.Extract_pupils = wfsParams["Extract_pupils"]
        self.bin_factor = wfsParams["Bin_factor"]
        self.bin_step = nn.AvgPool2d(self.bin_factor)
        self.Nres = wfsParams["Nres"]
        self.Centering_noise = wfsParams["Center_noise"]

    def ProcessReference(self, reference_frame):

        frame = torch.clone(reference_frame)
        frame = frame.unsqueeze(0)

        frame = self.GetPupils(frame, isReference = True)

        if self.bin_factor is not 1:
            frame = self.BinImage(frame) * self.bin_factor ** 2

        self.normalization = torch.std(frame, dim=(-2, -1), keepdim=True)
        self.reference = frame

    def ProcessFrame(self, input_frame):

        frame = torch.clone(input_frame)

        frame = self.GetPupils(frame)

        if self.bin_factor is not 1:
            frame = self.BinImage(frame) * self.bin_factor ** 2

        if self.Substract_reference:
            frame = frame - self.reference
            frame = frame / self.normalization
        else:
            frame = frame - frame.mean(dim=(-2, -1), keepdim=True)
            frame = frame / frame.std(dim=(-2, -1), keepdim=True)

        return frame

    def GetPupils(self, images, isReference = False):

        Ncrop = self.Nres + 2 * int(self.bin_factor)
        centers = np.copy(self.maskManager.pupil_centers)  # // self.Bin_factor

        if self.Centering_noise > 0 and not isReference:
            centers += np.random.randint(
                -self.Centering_noise, self.Centering_noise, centers.shape
            ) * int(self.bin_factor)
            
        out = torch.zeros(
            (images.shape[0], centers.shape[0], Ncrop, Ncrop), device=self.device
        )

        for i, center in enumerate(centers):
            out[:, i] = images[
                ...,
                center[0] - Ncrop // 2 : center[0] + Ncrop // 2,
                center[1] - Ncrop // 2 : center[1] + Ncrop // 2,
            ]
        return out
    
    def BinImage(self, Image, bin_factor = None):
        
        if bin_factor is None: 
            bin_factor = bin_factor = self.bin_factor

        # output_size = int(Image.shape[-1] / bin_factor)
        out = F.interpolate(Image, scale_factor = 1 / bin_factor)
        return out
