import torch
import numpy as np
import torch.nn.functional as F


class FramePreprocess:
    def __init__(self, wfsParams, atmosParams, device, maskManager):
        self.reference = 1.0
        self.normalization = 1.0

        self.maskManager = maskManager

        self.device = device
        self.Nres = wfsParams["Nres"]
        self.Substract_reference = wfsParams["Substract_Reference"]
        self.Extract_pupils = wfsParams["Extract_pupils"]
        self.Bin_factor = wfsParams["Bin_factor"]
        self.Centering_noise = wfsParams["Center_noise"]

    def ProcessReference(self, reference_frame):

        frame = torch.clone(reference_frame)
        frame = frame.unsqueeze(0)

        # if self.Bin_factor > 1:
        #     frame = self.bin_image(frame, self.Bin_factor)

        if self.Extract_pupils:
            pupils = self.GetPupils(frame)

        self.normalization = torch.std(pupils, dim=(-2, -1), keepdim=True)
        self.reference = frame

    def ProcessFrame(self, input_frame):

        frame = torch.clone(input_frame)
        frame = frame - self.reference

        if self.Bin_factor > 1:
            frame = self.bin_image(frame, self.Bin_factor)

        if self.Extract_pupils:
            frame = self.GetPupils(frame)

        frame = frame / self.normalization
        return frame

    def GetPupils(self, images=None, add_centering_noise=False):

        if images is None:
            images = self.Image

        Ncrop = self.Nres + 2  # // self.Bin_factor

        centers = np.copy(self.maskManager.pupil_centers)  # // self.Bin_factor
        if self.Centering_noise > 0:
            centers += np.random.randint(
                -self.Centering_noise, self.Centering_noise, centers.shape
            )

        np.random.rand
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

    def bin_image(self, image: torch.Tensor, bin_size: int) -> torch.Tensor:
        """
        Bins a 2D image (or batch of images) by summing over bin_size x bin_size regions.

        Args:
            image (torch.Tensor): shape (B, C, H, W)
            bin_size (int): binning factor

        Returns:
            torch.Tensor: binned image of shape (B, C, H//bin_size, W//bin_size)
        """
        B, H, W = image.shape
        image = image.unsqueeze(1)  # Add channel dimension: (B, 1, H, W)

        # Create uniform binning kernel
        kernel = torch.ones((1, 1, bin_size, bin_size), device=image.device)

        # Apply convolution with stride = bin_size
        binned = F.conv2d(image, kernel, stride=bin_size)

        return binned.squeeze(1)  # Remove channel dimension -> (B, H//bin, W//bin)
