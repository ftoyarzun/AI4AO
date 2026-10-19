from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from .Utils import imshow_multiple, fused_optimizer_supported
from .paths import get_data_dir
from IPython.display import display, clear_output


class TwinCalibrator:
    """
    Fits a simulated WFS/DM twin (already constructed) to an already-acquired
    bench interaction matrix / reference frame. Bench-data loading and any
    instrument-specific raw-format parsing (padding, tiling, valid-pixel
    scatter/crop) stays the caller's responsibility; this class only handles
    the fitting/diagnostic steps that are common across instruments.

    Static offsets
    --------------
    Besides the WFS/DM parameters, the calibrator can fit a static pupil
    amplitude (`ref_pupil`) and a static OPD (`ref_opd`, meters), both of
    shape `(Nres, Nres)`. They don't exist until a fit asks for them
    (`fit_dm_and_offsets(fit_static_offsets=True)` or
    `fit_static_offsets_to_reference`). Until then `ref_pupil` is None and
    `ref_opd` is 0, which the WFS treats as "no offset".

    The amplitude is always positive: it is stored as a log-amplitude and
    `ref_pupil = exp(log_amplitude)`.

    The static OPD has one of three representations (`static_opd_mode`):

    - `"pixel"`: one free OPD value per pixel.
    - `"dm"`: the OPD produced by a static DM command,
      `dm.GetDMShape(command)`, through the DM's current influence
      functions. The command is either zonal (one coefficient per actuator,
      the default) or modal, `command = static_M2C @ coefficients`, when a
      `static_M2C` of shape `(totalAct, nModes)` is given. In this mode,
      `static_correction_command()` returns the command to send to the
      bench DM to cancel the static OPD.
    - `"dm+pixel"`: the DM OPD above plus a per-pixel residual for what the
      DM can't represent. Only the DM part can be corrected by the DM.

    Switching between representations (or between DM bases) keeps the
    current static OPD: it is least-squares projected onto the new one (see
    `project_static_opd_to_dm`).
    """

    STATIC_OPD_MODES = ("pixel", "dm", "dm+pixel")

    def __init__(self, wfs, dm, device):
        self.wfs = wfs
        self.dm = dm
        self.device = device

        self.reset_static_offsets()


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
        # weight_decay=0: decay would pull physical parameters (mask slopes/shifts,
        # DM amplitude, pupil offsets...) toward 0 for no physical reason
        optimizer = torch.optim.AdamW(params_to_optimize, lr, weight_decay=0, fused=fused_optimizer_supported(self.device))
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
        optimizer = torch.optim.AdamW([wfs.rooftop], lr, weight_decay=0, fused=fused_optimizer_supported(self.device))
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

    # ------------------------------------------------------------------
    # Static offsets: storage, representations and conversions
    # ------------------------------------------------------------------

    def reset_static_offsets(self):
        """Drops the static amplitude/OPD; the next fit that asks for them
        starts again from A = 1 and OPD = 0."""
        self._log_ref_pupil = None    # log of the static amplitude, (Nres, Nres)
        self._ref_opd_pixel = None    # per-pixel static OPD in microns, (Nres, Nres)
        self._static_coefs = None     # DM-basis coefficients, (nBasis,), see _static_coef_scale
        self._static_M2C = None       # (totalAct, nModes) modal basis, None = zonal
        self._static_coef_scale = None
        self.static_opd_mode = None

    @property
    def ref_pupil(self):
        """Static pupil amplitude (always positive), or None if not fitted."""
        if self._log_ref_pupil is None:
            return None
        return torch.exp(self._log_ref_pupil)

    @property
    def ref_opd(self):
        """Total static OPD in meters (DM part + per-pixel part), or 0 if not
        fitted. The DM part uses the DM's current influence functions."""
        return self._static_opd(self.dm.IF)

    def static_dm_command(self):
        """Zonal DM command (dm.forward's `coefs` convention, shape
        `(totalAct,)`) that reproduces the DM part of the static OPD."""
        if self._static_coefs is None:
            raise ValueError(
                "The static OPD has no DM part (static_opd_mode="
                f"{self.static_opd_mode!r}). Fit with static_opd_mode='dm' or "
                "'dm+pixel', or call project_static_opd_to_dm() first."
            )
        coefs = self._static_coefs * self._static_coef_scale
        if self._static_M2C is None:
            return coefs
        return self._static_M2C @ coefs

    def static_correction_command(self, stroke_limit=None):
        """
        Zonal command to send to the bench DM to cancel the static OPD, i.e.
        `-static_dm_command()`, detached, in the same units and actuator
        order as the commands used to measure the bench interaction matrix
        (the twin's fitted DM `sign` absorbs the bench's sign convention).
        Only the DM part is corrected; in `"dm+pixel"` mode the per-pixel
        residual stays.

        `stroke_limit` clips every actuator to `[-stroke_limit, stroke_limit]`
        and warns how many actuators were clipped.
        """
        with torch.no_grad():
            command = -self.static_dm_command().detach().clone()
            if stroke_limit is not None:
                n_clipped = int((command.abs() > stroke_limit).sum())
                if n_clipped:
                    print(f"static_correction_command: {n_clipped}/{command.numel()} "
                          f"actuators clipped to +/-{stroke_limit}")
                command = command.clamp(-stroke_limit, stroke_limit)
        return command

    def project_static_opd_to_dm(self, M2C=None, keep_residual=True):
        """
        Least-squares projects the current total static OPD (over the WFS
        pupil) onto the DM, using a zonal basis (`M2C=None`) or the modal
        basis `M2C` of shape `(totalAct, nModes)`. With `keep_residual=True`
        the part the DM can't represent is kept as a per-pixel residual
        (`"dm+pixel"` mode); with False it is dropped (`"dm"` mode).
        Typical use: fit in `"pixel"` mode, then project to get DM commands.

        Returns the RMS (meters, piston removed) of the static OPD that the
        DM basis does not capture.
        """
        if self.static_opd_mode is None:
            raise ValueError("No static OPD has been fitted yet.")
        mode = "dm+pixel" if keep_residual else "dm"
        return self._convert_static_opd(mode, M2C)

    def static_opd_parts(self):
        """`(dm_opd, pixel_opd)` in meters, each a `(Nres, Nres)` tensor or
        None when that part doesn't exist in the current mode."""
        dm_opd = pixel_opd = None
        if self._static_coefs is not None:
            dm_opd = self.dm.GetDMShape(self.static_dm_command()[None])[0]
        if self._ref_opd_pixel is not None:
            pixel_opd = self._ref_opd_pixel * 1e-6
        return dm_opd, pixel_opd

    def _static_opd(self, IF):
        opd = 0
        if self._ref_opd_pixel is not None:
            opd = self._ref_opd_pixel * 1e-6      # trained in microns
        if self._static_coefs is not None:
            opd = opd + torch.einsum('a,awh->wh', self.static_dm_command(), IF)
        return opd

    def _prepare_static_offsets(self, static_opd_mode, static_M2C):
        """Creates the static offsets if needed, and converts them when the
        requested representation differs from the current one.
        `static_opd_mode=None` keeps the current mode ("pixel" if none)."""
        if static_opd_mode is None:
            static_opd_mode = self.static_opd_mode or "pixel"
        if static_opd_mode not in self.STATIC_OPD_MODES:
            raise ValueError(f"static_opd_mode must be one of {self.STATIC_OPD_MODES}, got {static_opd_mode!r}")

        if self._log_ref_pupil is None:
            self._log_ref_pupil = torch.nn.Parameter(
                torch.zeros(*self.wfs.pupil.shape, device=self.device, dtype=torch.float32)
            )

        unchanged = static_opd_mode == self.static_opd_mode and (
            static_opd_mode == "pixel" or self._same_static_basis(static_M2C)
        )
        if not unchanged:
            self._convert_static_opd(static_opd_mode, static_M2C)

    def _same_static_basis(self, M2C):
        if M2C is None or self._static_M2C is None:
            return M2C is None and self._static_M2C is None
        M2C = torch.as_tensor(M2C, device=self.device, dtype=torch.float32)
        return M2C.shape == self._static_M2C.shape and torch.equal(M2C, self._static_M2C)

    def _convert_static_opd(self, mode, M2C):
        """Re-expresses the current total static OPD in `mode` (and DM basis
        `M2C`), creating fresh Parameters. Returns the piston-removed RMS
        (meters) of the part the DM basis doesn't capture (0 in "pixel" mode)."""
        pupil = self.wfs.pupil.bool()
        with torch.no_grad():
            IF = self.dm.IF.detach()
            current = self._static_opd(IF)
            if not torch.is_tensor(current):
                current = torch.zeros(*pupil.shape, device=self.device, dtype=torch.float32)
            remainder = current.detach().clone()

            coefs = scale = None
            if mode != "pixel":
                if M2C is not None:
                    M2C = torch.as_tensor(M2C, device=self.device, dtype=torch.float32)
                # OPD of each basis vector over the pupil: (nBasis, Npupil)
                basis_opd = IF[:, pupil] if M2C is None else M2C.T @ IF[:, pupil]
                # Scale the trained coefficients so one unit is ~1 um RMS of OPD,
                # the same order of magnitude as the per-pixel map (in microns).
                scale = 1e-6 / basis_opd.pow(2).mean(dim=-1).sqrt().mean().item()
                command_coefs = remainder[pupil] @ torch.linalg.pinv(basis_opd)
                remainder[pupil] -= command_coefs @ basis_opd
                coefs = command_coefs / scale

            uncaptured = remainder[pupil] - remainder[pupil].mean()
            uncaptured_rms = uncaptured.pow(2).mean().sqrt().item() if mode != "pixel" else 0.0

            self._static_M2C = M2C if mode != "pixel" else None
            self._static_coef_scale = scale
            self._static_coefs = torch.nn.Parameter(coefs) if coefs is not None else None
            self._ref_opd_pixel = torch.nn.Parameter(remainder / 1e-6) if mode != "dm" else None
            self.static_opd_mode = mode

        return uncaptured_rms

    def _static_param_groups(self, residual_lr_factor):
        """Optimizer groups for the static offsets. Each carries an
        `lr_factor`, so a schedule sets `lr = lr_factor * base_lr`. The
        per-pixel residual of "dm+pixel" mode gets `residual_lr_factor`, so
        the DM command captures as much of the static OPD as it can."""
        groups = [{"params": [self._log_ref_pupil], "lr_factor": 1.0}]
        if self._static_coefs is not None:
            groups[0]["params"].append(self._static_coefs)
        if self._ref_opd_pixel is not None:
            factor = residual_lr_factor if self.static_opd_mode == "dm+pixel" else 1.0
            groups.append({"params": [self._ref_opd_pixel], "lr_factor": factor})
        return groups

    @staticmethod
    def _set_static_lr(optimizer, base_lr):
        for group in optimizer.param_groups:
            if "lr_factor" in group:
                group["lr"] = group["lr_factor"] * base_lr

    def fit_static_offsets_to_reference(self, reference_frame, static_opd_mode=None, static_M2C=None,
                                        n_iter=200, lr=1e-2, residual_lr_factor=0.1,
                                        loss_fn=None, live_plot=True):
        """
        Fits only the static amplitude and OPD so that the WFS reference
        frame (no DM command applied) matches a single measured
        `reference_frame`. The WFS mask and DM influence functions are held
        fixed at their current values. Same representations as
        `fit_dm_and_offsets` (`static_opd_mode`, `static_M2C`,
        `residual_lr_factor`); `loss_fn` as in `fit_pupil_to_reference`.

        `reference_frame` must be normalized like `wfs.reference_intensity`.
        A single frame need not constrain every static aberration equally
        well, so check the retrieved maps rather than only the loss.

        Leaves `wfs.reference_intensity` set (graph-free) to the reference
        frame with the fitted offsets. Returns the final loss.
        """
        if loss_fn is None:
            loss_fn = lambda ref, dig: (torch.abs(ref - dig) ** 2).sum()

        wfs = self.wfs
        self._prepare_static_offsets(static_opd_mode, static_M2C)

        # Graph-free mask and influence functions: only the static maps are fit,
        # and reusing a mask/IF built with grad across iterations would try to
        # backprop through a freed graph.
        with torch.no_grad():
            wfs.BuildMask()
        IF = self.dm.IF.detach()

        optimizer = torch.optim.AdamW(self._static_param_groups(residual_lr_factor), lr,
                                      weight_decay=0, fused=fused_optimizer_supported(self.device))
        self._set_static_lr(optimizer, lr)

        if live_plot:
            fig, ax = plt.subplots(figsize=(6, 6))
            img = ax.imshow(reference_frame.cpu().detach().numpy())
            fig.colorbar(img)
            plt.show()

        final_loss = None
        for u in range(n_iter):
            optimizer.zero_grad()

            wfs.BuildReferenceIntensity(self._static_opd(IF), self.ref_pupil)
            digital_image = wfs.reference_intensity

            l = loss_fn(reference_frame, digital_image)
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

        with torch.no_grad():
            wfs.BuildReferenceIntensity(self._static_opd(IF), self.ref_pupil)

        return final_loss

    def sanity_check_plot(self, bench_iMat, M2C, mode_index, idx, batch_size=30):
        wfs, dm = self.wfs, self.dm
        target = bench_iMat[mode_index]

        modes = dm(M2C[:, mode_index].T)
        wfs.BuildInteractionMatrix(modes, pupil=self.ref_pupil, batch_size=batch_size, opdOffset=self.ref_opd)
        digital_image = wfs.iMat

        imshow_multiple([
            {"tensor": target, "title": "Bench iMat", "same_scale": True},
            {"tensor": digital_image, "title": "Starting guess iMat", "same_scale": True},
            {"tensor": (target - digital_image), "title": "Difference", "scale_reference": target},
        ],
        max_channel_number=idx,
        group_boxes=True
        )

    def fit_dm_and_offsets(self, bench_iMat, M2C, mode_index, n_iter=200, lr_dm=1e-2, lr_wfs=1e-3,
                            lr_offset_start=-10, lr_offset_end=-2, fit_static_offsets=True,
                            static_opd_mode=None, static_M2C=None, residual_lr_factor=0.1,
                            batch_size=30, live_plot=True, plot_mode_idx=4):
        """
        Jointly optimizes DM misregistration, WFS mask parameters, and
        (optionally) a static pupil amplitude/OPD offset to match a bench
        interaction matrix, with a log-space learning-rate warmup on the
        offset parameters (from `10**lr_offset_start` to `10**lr_offset_end`
        over the first half of the iterations).

        `fit_static_offsets=True` creates the static offsets on first use
        and fits them. With False they are not fitted: existing offsets
        (e.g. from `fit_static_offsets_to_reference`) are still applied as
        fixed, and if none exist none are used.

        `static_opd_mode` selects the static-OPD representation: "pixel",
        "dm" or "dm+pixel" (see the class docstring); None keeps the current
        one ("pixel" the first time). `static_M2C` (`(totalAct, nModes)`,
        default None = zonal) is the DM basis for the "dm" modes; it is
        independent of the `M2C` used for the interaction matrix.
        `residual_lr_factor` scales the learning rate of the per-pixel
        residual in "dm+pixel" mode.
        """
        wfs, dm = self.wfs, self.dm
        wfs.train()
        dm.train()

        target = bench_iMat[mode_index]

        static_groups = []
        if fit_static_offsets:
            self._prepare_static_offsets(static_opd_mode, static_M2C)
            static_groups = self._static_param_groups(residual_lr_factor)

        if live_plot:
            fig, ax = imshow_multiple([
                        {"tensor": target, "title": "Bench iMat", "same_scale": True},
                        {"tensor": target, "title": "Calibrated iMat", "same_scale": True},
                        {"tensor": (target - target), "title": "Difference", "scale_reference": target},
                    ],
                    max_channel_number=plot_mode_idx,
                    group_boxes=True
                    )

        original_positions = dm.anamorphosis_coordinates(dm.actuator_positions)
        original_positions = dm.rotate_coordinates(original_positions)

        warmup_iters = n_iter // 2
        # Only parameters left trainable are fit, so a caller can exclude one by
        # freezing it (e.g. `wfs.rooftop.requires_grad_(False)`) while the module is
        # in train mode; the train() calls above keep that choice. (Freezing it while
        # in eval mode is a no-op: train() restores the flags from before eval().)
        param_groups = [
            {"params": [p for p in dm.parameters() if p.requires_grad], "lr": lr_dm},
            {"params": [p for p in wfs.parameters() if p.requires_grad], "lr": lr_wfs},
        ]
        param_groups = [g for g in param_groups if g["params"]]
        for g in static_groups:
            g["lr"] = g["lr_factor"] * 10 ** lr_offset_start
        param_groups += static_groups

        optimizer = torch.optim.AdamW(param_groups, weight_decay=0, fused=fused_optimizer_supported(self.device))

        final_loss = None
        for u in range(n_iter):
            if static_groups:
                if u < warmup_iters:
                    exponent = lr_offset_start + (u / warmup_iters) * (lr_offset_end - lr_offset_start)
                else:
                    exponent = lr_offset_end
                self._set_static_lr(optimizer, 10 ** exponent)

            optimizer.zero_grad()

            wfs.BuildMask()
            modes = dm(M2C[:, mode_index].T)
            # ref_opd after dm(...): its DM part uses this iteration's influence functions
            wfs.BuildInteractionMatrix(modes, pupil=self.ref_pupil, batch_size=batch_size, opdOffset=self.ref_opd)
            digital_image = wfs.iMat

            l = ((target - digital_image) ** 2).sum()
            l.backward()
            optimizer.step()
            final_loss = l.item()

            if live_plot and u % 10 == 0:
                clear_output(wait=True)
                imshow_multiple([
                        {"tensor": target, "title": "Bench iMat", "same_scale": True},
                        {"tensor": digital_image, "title": "Calibrated iMat", "same_scale": True},
                        {"tensor": (target - digital_image), "title": "Difference", "scale_reference": target},
                    ],
                    fig=fig, axes=ax,
                    max_channel_number=plot_mode_idx,
                    group_boxes=True
                    )
                display(fig, clear=True)
                plt.pause(0.1)

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
        """Actuator positions, static amplitude and static OPD (in "dm+pixel"
        mode also its DM part and per-pixel residual, separately)."""
        with torch.no_grad():
            dm_opd, pixel_opd = self.static_opd_parts()
            static_opd = self.ref_opd
        panels = [("Retrieved pupil amplitude", self.ref_pupil),
                  (f"Retrieved static OPD [m] ({self.static_opd_mode})",
                   static_opd if torch.is_tensor(static_opd) else None)]
        if dm_opd is not None and pixel_opd is not None:
            panels += [("Static OPD: DM part [m]", dm_opd), ("Static OPD: pixel residual [m]", pixel_opd)]

        n = 1 + len(panels)
        plt.figure(figsize=(16 * n / 3, 6))
        plt.subplot(1, n, 1)
        plt.title("Old vs new actuator positions")
        plt.scatter(original_positions[:, 0].cpu().detach(), original_positions[:, 1].cpu().detach())
        plt.scatter(transformed_positions[:, 0].cpu().detach(), transformed_positions[:, 1].cpu().detach())
        plt.legend(['Old positions', 'New positions'])
        plt.axis('equal')

        for i, (title, tensor) in enumerate(panels):
            plt.subplot(1, n, i + 2)
            plt.title(title)
            if tensor is not None:
                plt.imshow(tensor.cpu().detach())
                plt.colorbar()

    def rebuild_reconstruction_matrix(self, M2C, batch_size=30):
        self.dm.eval()
        modes = self.dm(M2C.T)
        with torch.no_grad():
            self.wfs.BuildReconstructionMatrix(modes, pupil=self.ref_pupil, batch_size=batch_size, opdOffset=self.ref_opd)
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

    def _twin_paths(self, instrument_name, data_dir):
        """`(wfs_path, dm_path)` for an instrument. `data_dir` defaults to the
        AI4AO data directory (see AI4AO.paths); pass a path to override."""
        base = Path(data_dir) if data_dir is not None else get_data_dir()
        inst = base / instrument_name
        return inst / f"{instrument_name}WFS.pth", inst / f"{instrument_name}DM.pth"

    def save(self, instrument_name, data_dir=None):
        wfs_path, dm_path = self._twin_paths(instrument_name, data_dir)
        wfs_path.parent.mkdir(parents=True, exist_ok=True)
        self.wfs.SaveCalibration(str(wfs_path))
        self.dm.SaveCalibration(str(dm_path))
        return str(wfs_path), str(dm_path)

    def load(self, instrument_name, data_dir=None):
        wfs_path, dm_path = self._twin_paths(instrument_name, data_dir)
        wfs_path, dm_path = str(wfs_path), str(dm_path)
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
