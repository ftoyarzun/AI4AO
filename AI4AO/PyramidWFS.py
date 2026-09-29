import torch # type: ignore[import]
import torch.nn as nn # type: ignore[import]
import numpy as np
from .TorchPropagator import WFS

class PyramidWFS(WFS):
    def __init__(self, ParamsDict, device):
        super().__init__(ParamsDict, device)

        self.initialized = False

        self.mainSlope = nn.Parameter(torch.tensor(torch.pi / 2, device=self.device, dtype=torch.float32))
        self.maskShifts = nn.Parameter(torch.ones(4, 2, device=self.device, dtype=torch.float32))
        self.rooftop = nn.Parameter(torch.tensor(0, device=self.device, dtype=torch.float32))

        self.modulation = ParamsDict["Modulation"]
        self.maxModulationSteps = 32

        self.BuildMask()

        self.initialized = True

    def BuildMask(self):
        """Builds the (Nwavelength, Nsteps, H, W) phase mask (Nsteps = 1 when
        unmodulated) and passes it to SetMask's 4-D per-wavelength path.

        Modulation and rooftop are given in lambda_c/D (lambda_c: centre of the
        band, see WFS.ChromaticRatio). Both are fixed angles, so each wavelength
        sees them at sampling * lambda_c / lambda pixels. The facets are linear
        ramps, which are scale-invariant, so the facet slopes and the pupil
        positions stay achromatic.
        """
        ratio = self.ChromaticRatio()  # (Nwavelength,), exactly 1 for one wavelength
        sampling = (self.sampling * ratio).view(-1, 1)  # (Nwavelength, 1)
        if self.modulation == 0:
            F = self.PyramidMask(sampling=sampling)
        else:
            # Size the step count for the largest radius in lambda/D across the band
            # (at the shortest wavelength). The count is an integer, so no gradient.
            max_modulation = self.modulation * float(ratio.max().detach())
            nSteps = min(self.maxModulationSteps, max(round(6.28 * max_modulation / 4) * 4, 8))
            steps = torch.linspace(0,2*torch.pi * (1 - 1/nSteps),nSteps, device=self.device, dtype=torch.float32)
            radius = (self.modulation * self.sampling) * ratio.view(-1, 1)  # (Nwavelength, 1), pixels
            x = radius * torch.cos(steps)
            y = radius * torch.sin(steps)
            F = self.PyramidMask(x_offset=x, y_offset=y, sampling=sampling)
        self.pupil_centers = self.GetPupilCenter()
        self.SetMask(phaseMask=F)
    
    def GetPupilCenter(self):
        sign_tensor = -torch.tensor(
            [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]], device=self.device
        )
        frame_center = torch.ones(4, 2, device=self.device) * self.Npix / 2
        pupil_center = frame_center + sign_tensor * self.maskShifts * self.mainSlope * self.Npix / 4 / (torch.pi/2)
        pupil_center = torch.round(pupil_center).to(dtype=torch.int).cpu().detach().numpy()
        return pupil_center
    
    def PropagateField(self, uin, uin_padded):
        self.FFTPropagator(uin_padded)

    def BuildPrismMask(self, pupil_proportion, Nsamples = 5):
        self.BuildMask()
        displacement_size_in_pix = self.Nres * pupil_proportion / self.sampling
        displacement_array = torch.linspace(
            -displacement_size_in_pix/2, displacement_size_in_pix/2, Nsamples, 
            device = self.device, dtype = torch.float32
            ).view(Nsamples,1,1)

        standard_pupil_displacement_in_pix = self.mainSlope / (2 * torch.pi) * self.Npix
        samples_pupil_positions_array = displacement_array + standard_pupil_displacement_in_pix

        displacement_factor = samples_pupil_positions_array / self.mainSlope / self.Npix * (2 * torch.pi)

        mask = torch.clone(self.phaseMask).repeat((Nsamples,1,1))

        mask *= displacement_factor

        self.SetMask(phaseMask=mask)

    def PyramidMask(self, x_offset=0, y_offset=0, sampling=None):
        """Pyramid phase mask for tip offsets (x_offset, y_offset), in focal-plane pixels.

        The output shape is (*S, H, W), where S is the broadcast shape of the
        offsets and of `sampling`. Scalar offsets give a single (H, W) mask, and
        1-D offsets of length C give (C, H, W), one mask per offset pair.
        `sampling` (focal-plane pixels per lambda/D) sets the rooftop width in
        pixels and defaults to self.sampling. BuildMask passes (Nwavelength, 1)
        together with (Nwavelength, Nsteps) offsets, which gives
        (Nwavelength, Nsteps, H, W).
        """
        if sampling is None:
            sampling = self.sampling
        x_offset = torch.as_tensor(x_offset, device=self.device, dtype=torch.float32)
        y_offset = torch.as_tensor(y_offset, device=self.device, dtype=torch.float32)

        rooftop_in_pixels = torch.as_tensor(self.rooftop * sampling / np.sqrt(2))[..., None, None]

        x = self.x_mask + x_offset[..., None, None]  # (*S, H, W)
        y = self.y_mask + y_offset[..., None, None]  # (*S, H, W)

        P1 = (x + rooftop_in_pixels / 2) * self.maskShifts[0, 0] + (
            y + rooftop_in_pixels / 2
        ) * self.maskShifts[0, 1]
        P2 = -x * self.maskShifts[1, 0] + y * self.maskShifts[1, 1]
        P3 = (
            -(x - rooftop_in_pixels / 2) * self.maskShifts[2, 0]
            - (y - rooftop_in_pixels / 2) * self.maskShifts[2, 1]
        )
        P4 = x * self.maskShifts[3, 0] - y * self.maskShifts[3, 1]

        stacked = torch.stack(torch.broadcast_tensors(P1, P2, P3, P4))  # shape: (4, *S, H, W)

        F = torch.max(stacked * self.mainSlope, dim=0).values  # shape (*S, H, W)

        return F

    @property
    def modulation(self):
        return self._modulation
    @modulation.setter
    def modulation(self, value):
        with torch.no_grad():
            self._modulation = value
            if self.initialized:
                self.BuildMask()
                self.BuildReferenceIntensity()

    
   