import torch # type: ignore[import]
import torch.nn as nn # type: ignore[import]
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
        self.Pupil_size_noise = wfsParams["Pupil_size_noise"]
        self.wfs = wfs

        self.Ncrop = self.Nres + self.Extract_pupils_pad
        self.Nout = self.Ncrop // self.bin_factor
        self.crop_center_offset = self.Ncrop // 2 - (self.Ncrop - 1) / 2

    def ProcessReference(self, reference_frame):

        frame = torch.clone(reference_frame)
        frame = frame.unsqueeze(0)

        frame = self.GetTrainingPupils(frame, add_position_noise=False, add_size_noise=False)
        frame = frame * self.bin_factor ** 2

        self.normalization = torch.std(frame, dim=(-2, -1), keepdim=True)
        self.reference = frame

    def ProcessFrame(self, input_frame, add_pupil_noise = True):

        frame = torch.clone(input_frame)

        frame = self.GetTrainingPupils(
            frame, add_position_noise=add_pupil_noise, add_size_noise=add_pupil_noise
        )
        frame = frame * self.bin_factor ** 2

        if self.Substract_reference:
            frame = frame - self.reference
            frame = frame / self.normalization
        else:
            frame = frame - frame.mean(dim=(-2, -1), keepdim=True)
            frame = frame / self.normalization

        return frame

    def GetTrainingPupils(self, images, add_position_noise = True, add_size_noise = True):
        # Vectorized affine_grid/grid_sample crop+resize: one call for every (batch, pupil) pair, no loop.
        B = images.shape[0]
        C = self.wfs.pupil_centers.shape[0]
        W_in = images.shape[-1]

        centers = torch.as_tensor(self.wfs.pupil_centers, device=self.device, dtype=torch.float32)
        centers = centers[None].expand(B, C, 2).clone()  # (B,C,2) as [y,x]

        if add_position_noise:
            pos_noise = (torch.rand(B, C, 2, device=self.device) * 2 - 1) * self.Centering_noise
            centers = centers + pos_noise

        if add_size_noise:
            jitter = torch.empty(B, C, device=self.device).uniform_(
                1 - self.Pupil_size_noise, 1 + self.Pupil_size_noise
            )
        else:
            jitter = torch.ones(B, C, device=self.device)

        centers = centers - self.crop_center_offset
        crop_size = self.Ncrop * jitter  # (B,C) crop window size in raw (unbinned) pixels

        scale = (crop_size / W_in).reshape(B * C)
        ty = ((2 * centers[..., 0] + 1 - W_in) / W_in).reshape(B * C)
        tx = ((2 * centers[..., 1] + 1 - W_in) / W_in).reshape(B * C)

        theta = torch.zeros(B * C, 2, 3, device=self.device, dtype=torch.float32)
        theta[:, 0, 0] = scale
        theta[:, 1, 1] = scale
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty

        images_expanded = images.unsqueeze(1).expand(B, C, *images.shape[-2:])
        images_expanded = images_expanded.reshape(B * C, 1, *images.shape[-2:])

        grid = F.affine_grid(theta, size=(B * C, 1, self.Nout, self.Nout), align_corners=False)
        patches = F.grid_sample(images_expanded, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

        return patches.reshape(B, C, self.Nout, self.Nout)
