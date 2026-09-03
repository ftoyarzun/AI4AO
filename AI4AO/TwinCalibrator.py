import numpy as np
import torch
import matplotlib.pyplot as plt
from IPython.display import display, clear_output


class TwinCalibrator:
    """
    Fits a simulated WFS/DM twin (already constructed) to an already-acquired
    bench interaction matrix / reference frame. Bench-data loading and any
    instrument-specific raw-format parsing (padding, tiling, valid-pixel
    scatter/crop) stays the caller's responsibility; this class only handles
    the fitting/diagnostic steps that are common across instruments.
    """

    def __init__(self, wfs, dm, device):
        self.wfs = wfs
        self.dm = dm
        self.device = device
        self.ref_pupil = None
        self.ref_phase = None

    def fit_pupil_to_reference(self, reference_frame, params_to_optimize, lr=3e-3, n_iter=200, live_plot=True, loss_fn=None):
        """
        `loss_fn(reference_frame, digital_image) -> scalar tensor` lets callers
        match an instrument's specific loss formula/scaling (e.g. some
        notebooks scale by 1e5 before an MSE loss instead of a raw sum of
        squares). Defaults to `sum(|reference_frame - digital_image|**2)`.
        """
        if loss_fn is None:
            loss_fn = lambda ref, dig: (torch.abs(ref - dig) ** 2).sum()

        wfs = self.wfs
        optimizer = torch.optim.AdamW(params_to_optimize, lr, fused=True)
        wfs.train()

        if live_plot:
            fig, ax = plt.subplots(figsize=(6, 6))
            img = ax.imshow(reference_frame.cpu().detach().numpy())
            fig.colorbar(img)
            plt.show()

        final_loss = None
        for u in range(n_iter):
            optimizer.zero_grad()

            wfs.BuildMask()
            wfs.BuildReferenceIntensity()
            digital_image = wfs.reference_intensity

            l = loss_fn(reference_frame, digital_image)
            l.backward()
            optimizer.step()
            final_loss = l.item()

            if live_plot and u % 10 == 0:
                clear_output(wait=True)
                img.set_data((reference_frame - wfs.reference_intensity).cpu().detach().numpy())
                img.set_clim(vmin=np.min(img.get_array()), vmax=np.max(img.get_array()))
                display(fig, clear=True)
                plt.pause(0.1)

        if live_plot:
            plt.close(fig)

        return final_loss

    def fit_rooftop(self, reference_frame, lr=1e-1, n_iter=100, live_plot=True):
        """
        Calibrates a `.rooftop` mask parameter (e.g. Papyrus) by matching the
        per-quadrant flux of a 2x2-tiled reference frame instead of the full
        pixel-wise image.
        """
        wfs = self.wfs
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW([wfs.rooftop], lr, fused=True)
        wfs.train()

        half = reference_frame.shape[-1] // 2
        bench_quadrants = reference_frame.view(2, half, 2, half).permute(0, 2, 1, 3).sum(dim=(-2, -1))

        if live_plot:
            fig, ax = plt.subplots(figsize=(6, 6))
            img = ax.imshow(reference_frame.cpu().detach().numpy())
            fig.colorbar(img)
            plt.show()

        final_loss = None
        for u in range(n_iter):
            optimizer.zero_grad()

            wfs.BuildMask()
            wfs.BuildReferenceIntensity()
            digital_image = wfs.reference_intensity
            digital_quadrants = digital_image.view(2, half, 2, half).permute(0, 2, 1, 3).sum(dim=(-2, -1))

            l = loss_fn(bench_quadrants, digital_quadrants)
            l.backward()
            optimizer.step()
            final_loss = l.item()

            if live_plot and u % 10 == 0:
                clear_output(wait=True)
                img.set_data((reference_frame - digital_image).cpu().detach().numpy())
                img.set_clim(vmin=np.min(img.get_array()), vmax=np.max(img.get_array()))
                display(fig, clear=True)
                plt.pause(0.1)

        if live_plot:
            plt.close(fig)

        return final_loss

    def rough_calibrate_dm(self, bench_iMat, M2C):
        self.wfs.BuildReferenceIntensity()
        self.dm.RoughCalibration(self.wfs, bench_iMat, M2C)

    def init_static_offsets(self):
        wfs = self.wfs
        self.ref_phase = torch.nn.Parameter(torch.zeros(*wfs.pupil.shape, device=self.device, dtype=torch.float32))
        self.ref_pupil = torch.nn.Parameter(torch.ones(*wfs.pupil.shape, device=self.device, dtype=torch.float32))
        return self.ref_pupil, self.ref_phase

    def sanity_check_plot(self, bench_iMat, M2C, mode_index, idx, batch_size=30):
        wfs, dm = self.wfs, self.dm
        target = bench_iMat[mode_index]

        modes = dm(M2C[:, mode_index].T)
        wfs.BuildInteractionMatrix(modes, pupil=self.ref_pupil, batch_size=batch_size, phaseOffset=self.ref_phase)
        digital_image = wfs.iMat

        plt.figure(figsize=(9, 4))
        plt.subplot(131)
        plt.imshow(target[idx].cpu().numpy())
        plt.subplot(132)
        plt.imshow(digital_image[idx].cpu().detach().numpy())
        plt.subplot(133)
        target_idx = target[idx].cpu().detach().numpy()
        plt.imshow((target - digital_image)[idx].cpu().detach().numpy(), vmin=target_idx.min(), vmax=target_idx.max())

    def fit_dm_and_offsets(self, bench_iMat, M2C, mode_index, n_iter=200, lr_dm=1e-2, lr_wfs=1e-3,
                            lr_offset_start=-10, lr_offset_end=-2, fit_static_offsets=True,
                            batch_size=30, live_plot=True, plot_mode_idx=4):
        """
        Jointly optimizes DM misregistration, WFS mask parameters, and
        (optionally) a static pupil-illumination/phase offset to match a
        bench interaction matrix, with a log-space learning-rate warmup on
        the offset parameters. Set `fit_static_offsets=False` to skip the
        offset fit entirely (e.g. Ekarus, where the bench data doesn't
        support it).
        """
        wfs, dm = self.wfs, self.dm
        wfs.train()
        dm.train()

        target = bench_iMat[mode_index]

        if fit_static_offsets:
            if self.ref_pupil is None or self.ref_phase is None:
                self.init_static_offsets()
            offset_params = [self.ref_phase, self.ref_pupil]
        else:
            self.ref_pupil = None
            self.ref_phase = 0
            offset_params = []

        if live_plot:
            fig, ax = plt.subplots(figsize=(6, 6))
            img = ax.imshow(target[plot_mode_idx].cpu().detach().numpy())
            fig.colorbar(img)

        original_positions = dm.anamorphosis_coordinates(dm.actuator_positions)
        original_positions = dm.rotate_coordinates(original_positions)

        warmup_iters = n_iter // 2
        param_groups = [
            {"params": dm.parameters(), "lr": lr_dm},
            {"params": wfs.parameters(), "lr": lr_wfs},
        ]
        offset_group_index = None
        if offset_params:
            offset_group_index = len(param_groups)
            param_groups.append({"params": offset_params, "lr": 10 ** lr_offset_start})

        optimizer = torch.optim.AdamW(param_groups, fused=True)

        final_loss = None
        for u in range(n_iter):
            if offset_group_index is not None:
                if u < warmup_iters:
                    alpha = u / warmup_iters
                    optimizer.param_groups[offset_group_index]["lr"] = 10 ** (
                        lr_offset_start + alpha * (lr_offset_end - lr_offset_start)
                    )
                if u == warmup_iters:
                    optimizer.param_groups[offset_group_index]["lr"] = 10 ** lr_offset_end

            optimizer.zero_grad()

            wfs.BuildMask()
            modes = dm(M2C[:, mode_index].T)
            wfs.BuildInteractionMatrix(modes, pupil=self.ref_pupil, batch_size=batch_size, phaseOffset=self.ref_phase)
            digital_image = wfs.iMat

            l = ((target - digital_image) ** 2).sum()
            l.backward()
            optimizer.step()
            final_loss = l.item()

            if live_plot and u % 10 == 0:
                clear_output(wait=True)
                img.set_data((target - digital_image)[plot_mode_idx].cpu().detach().numpy() * 1e10)
                img.set_clim(vmin=np.min(img.get_array()), vmax=np.max(img.get_array()))
                display(fig, clear=True)
                plt.pause(0.5)

                print("#" * 40)
                print(f"loss = {l.item():.5f}")
                print("#" * 40)

        if live_plot:
            plt.close(fig)

        print(dm.GetMisreg())

        transformed_positions = dm.anamorphosis_coordinates(dm.actuator_positions)
        transformed_positions = dm.rotate_coordinates(transformed_positions)

        return final_loss, original_positions, transformed_positions

    def plot_actuator_and_offsets(self, original_positions, transformed_positions):
        plt.figure(figsize=(16, 6))
        plt.subplot(131)
        plt.title("Old vs new actuator positions")
        plt.scatter(original_positions[:, 0].cpu().detach(), original_positions[:, 1].cpu().detach())
        plt.scatter(transformed_positions[:, 0].cpu().detach(), transformed_positions[:, 1].cpu().detach())
        plt.legend(['Old positions', 'New positions'])
        plt.axis('equal')

        plt.subplot(132)
        plt.title('Retrieved pupil illumination')
        if self.ref_pupil is not None:
            plt.imshow(self.ref_pupil.cpu().detach())
            plt.colorbar()

        plt.subplot(133)
        plt.title('Retrieved static phase')
        if self.ref_phase is not None:
            plt.imshow(self.ref_phase.cpu().detach())
            plt.colorbar()

    def rebuild_reconstruction_matrix(self, M2C, batch_size=30):
        self.dm.eval()
        modes = self.dm(M2C.T)
        with torch.no_grad():
            self.wfs.BuildReconstructionMatrix(modes, pupil=self.ref_pupil, batch_size=batch_size, phaseOffset=self.ref_phase)
        return modes

    def plot_fit_residual(self, bench_iMat, idx, vmin=None, vmax=None, show_diff=True, reshape_digital=False):
        """
        `reshape_digital=True` reshapes `wfs.iMat[idx]` to `bench_iMat[idx]`'s
        shape before plotting/differencing (needed when the twin's own iMat
        layout doesn't already match the bench's, e.g. tiled-pyramid data).
        """
        target = bench_iMat[idx]
        digital = self.wfs.iMat[idx]
        if reshape_digital:
            digital = digital.reshape(target.shape)

        n_panels = 3 if show_diff else 2

        plt.subplot(1, n_panels, 1)
        plt.imshow(target.cpu().detach().numpy())
        plt.subplot(1, n_panels, 2)
        plt.imshow(digital.cpu().detach().numpy())
        if show_diff:
            plt.subplot(1, n_panels, 3)
            plt.imshow((target - digital).cpu().detach().numpy(), vmin=vmin, vmax=vmax)

    def crosstalk_diagnostic(self, bench_iMat):
        with torch.no_grad():
            cov = bench_iMat.flatten(start_dim=-2) @ self.wfs.reconstructionMatrix

        plt.figure(figsize=(10, 6))
        plt.subplot(121)
        plt.imshow(cov.cpu().detach())
        plt.subplot(122)
        plt.plot(torch.diag(cov).cpu().detach())

        return cov

    def save(self, instrument_name, data_dir="../Data"):
        wfs_path = f"{data_dir}/{instrument_name}/{instrument_name}WFS.pth"
        dm_path = f"{data_dir}/{instrument_name}/{instrument_name}DM.pth"
        self.wfs.SaveCalibration(wfs_path)
        self.dm.SaveCalibration(dm_path)
        return wfs_path, dm_path

    def load(self, instrument_name, data_dir="../Data"):
        wfs_path = f"{data_dir}/{instrument_name}/{instrument_name}WFS.pth"
        dm_path = f"{data_dir}/{instrument_name}/{instrument_name}DM.pth"
        self.wfs.LoadCalibration(wfs_path)
        self.wfs.eval()
        self.dm.LoadCalibration(dm_path)
        self.dm.eval()
        return wfs_path, dm_path

    @staticmethod
    def tile_pyramid_frame(flat_frames, valid_pix_map, coords, full_shape, pupil_size, pupil_separation):
        """
        Scatters a batch of flat bench-pixel vectors (as read off the real
        4-pupil pyramid detector, one value per `valid_pix_map` pixel) into
        full detector-sized frames, crops a `pupil_size + 2*pupil_separation`
        patch around each of the 4 pupil `coords`, and stitches the 4 crops
        into a single 2x2-tiled image per input frame (matches the raw bench
        interaction-matrix layout used e.g. for Rama).

        Parameters
        ----------
        flat_frames : ndarray, shape (N, n_valid_pix)
        valid_pix_map : ndarray, shape full_shape, nonzero where flat_frames values go
        coords : sequence of 4 (x, y) pupil centers, in (top-left, bottom-left,
            bottom-right, top-right) order
        full_shape : tuple (H, W) of the raw detector frame
        pupil_size, pupil_separation : int

        Returns
        -------
        tiled : ndarray, shape (N, 2*(pupil_size+2*pupil_separation), 2*(pupil_size+2*pupil_separation))
        """
        crop_size = pupil_size + 2 * pupil_separation
        n_frames = flat_frames.shape[0]

        crops = np.zeros((4, crop_size, crop_size))
        full = np.zeros(full_shape)
        tiled = np.zeros((n_frames, 2 * crop_size, 2 * crop_size))

        for i in range(n_frames):
            full[valid_pix_map != 0] = flat_frames[i]
            for j, (x, y) in enumerate(coords):
                crops[j] = full[
                    x - pupil_separation - pupil_size // 2: x + pupil_size // 2 + pupil_separation,
                    y - pupil_separation - pupil_size // 2: y + pupil_size // 2 + pupil_separation,
                ]
            tiled[i] = np.block([
                [crops[0], crops[3]],
                [crops[1], crops[2]],
            ])

        return tiled

    @staticmethod
    def untile_pyramid_image(tiled, coords, full_shape, pupil_size, pupil_separation):
        """
        Inverse of `tile_pyramid_frame` for a single frame: reconstructs a
        full detector-sized image from a 2x2-tiled corner image, placing
        each quadrant back at its bench pupil `coords`. Used e.g. to convert
        a fitted synthetic interaction matrix back into the bench's flat
        valid-pixel format for direct comparison/export.

        Parameters
        ----------
        tiled : ndarray, shape (2*(pupil_size+2*pupil_separation), 2*(pupil_size+2*pupil_separation))
        coords : sequence of 4 (x, y) pupil centers, in (top-left, bottom-left,
            bottom-right, top-right) order
        full_shape : tuple (H, W) of the raw detector frame
        pupil_size, pupil_separation : int

        Returns
        -------
        full : ndarray, shape full_shape, zero outside the extracted regions
        """
        crop_size = pupil_size + 2 * pupil_separation
        full = np.zeros(full_shape, dtype=tiled.dtype)

        crops = [
            tiled[:crop_size, :crop_size],
            tiled[crop_size:, :crop_size],
            tiled[crop_size:, crop_size:],
            tiled[:crop_size, crop_size:],
        ]

        for crop, (x, y) in zip(crops, coords):
            full[
                x - pupil_separation - pupil_size // 2: x + pupil_size // 2 + pupil_separation,
                y - pupil_separation - pupil_size // 2: y + pupil_size // 2 + pupil_separation,
            ] = crop

        return full
