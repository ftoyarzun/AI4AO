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
        self.prismPupilProportion = None  # set by BuildPrismMask

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

        If BuildPrismMask has been called, each wavelength's mask is then scaled
        by PrismFactor to model the dispersion of a prism.
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
        if self.prismPupilProportion is not None:
            F = F * self.PrismFactor().view(-1, 1, 1, 1)
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

    def BuildPrismMask(self, pupil_proportion):
        """Model a dispersing prism in front of the pyramid: each wavelength's
        four pupil images are displaced along the facet diagonals, so across the
        band the pupils are smeared over a range set by `pupil_proportion`.

        Requires a polychromatic WFS (len(self.wavelength) > 1), since the
        dispersion is sampled by the sensing wavelengths themselves. The setting
        is stored in self.prismPupilProportion and re-applied on every BuildMask
        (train()/eval(), forward in train mode, wavelength changes). Set it back
        to None and call BuildMask() to remove the prism. The mask and reference
        intensity are rebuilt here.
        """
        if self.wavelength.numel() < 2:
            raise ValueError(
                "BuildPrismMask needs a polychromatic WFS: assign wfs.wavelength a "
                "1-D tensor of several sensing wavelengths first."
            )
        self.prismPupilProportion = pupil_proportion
        self.RebuildMaskAndReference()

    def PrismFactor(self):
        """Per-wavelength scale of the pyramid mask, shape (Nwavelength,), for the
        prism set by BuildPrismMask.

        Scaling a wavelength's mask by k moves its pupil images from
        mainSlope / (2 pi) * Npix to k times that distance from the frame centre
        along each axis. The displacement is linear in wavelength and spans
        Nres * prismPupilProportion / sampling pixels across the band, from -1/2
        of that at the shortest wavelength to +1/2 at the longest, centred on
        lambda_c. For an evenly spaced band this matches the former
        linspace-over-samples version. A single wavelength has nothing to
        disperse, so its factor is exactly 1.
        """
        wl = self.wavelength
        if wl.numel() < 2:
            return torch.ones_like(wl)
        displacement_size_in_pix = self.Nres * self.prismPupilProportion / self.sampling
        lambda_c = (wl.min() + wl.max()) / 2
        displacement = (wl - lambda_c) / (wl.max() - wl.min()) * displacement_size_in_pix  # (Nwavelength,)

        standard_pupil_displacement_in_pix = self.mainSlope / (2 * torch.pi) * self.Npix
        return (standard_pupil_displacement_in_pix + displacement) / standard_pupil_displacement_in_pix

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

    
   