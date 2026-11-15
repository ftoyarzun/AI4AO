"""Gradient-descent reconstructor: inverts the frozen twin frame by frame.

Design and prototype results: Ideas/24-gradient-descent-reconstructor.md.
"""
import torch  # type: ignore[import]


class GDReconstructor:
    """
    Reconstructs the OPD behind each WFS frame by gradient descent through the
    frozen twin: starting from zero, Adam fits an OPD so that ``wfs(OPD)``
    reproduces the frame (relative squared error, summed over the batch). No
    training is needed; each call runs ``n_iter`` forward/backward passes, so
    it is an analysis tool and a nonlinear reference, not a real-time
    controller.

    It has the reconstructor signature ``Trainer`` accepts for callables,
    ``modes = gd(wfs_frames, residual_opd=None)`` on raw WFS frames (no
    ``FramePreprocess``), like ``Trainer.linear_reconstructor()``. So it can be
    passed to ``Trainer(phaseReconstructor=gd).evaluate()`` or
    ``evaluate_many({"GD": gd, ...})``. ``residual_opd`` is ignored.

    ``opd_mode`` sets how the OPD is represented, as for ``TwinCalibrator``'s
    static OPD:

    - ``"dm"``: DM commands ``c`` in the basis ``M2C`` (``(totalAct, nModes)``,
      ``None`` = zonal), ``OPD = dm.GetDMShape(c @ M2C.T)``;
    - ``"pixel"``: a free per-pixel OPD map;
    - ``"dm+pixel"``: both; the per-pixel part learns ``residual_lr_factor``
      times slower, so the DM part captures as much as it can.

    A call returns DM coefficients ``(B, nModes)``, the DM's best fit to the
    whole fitted OPD, so the loop can be closed in every mode: the DM part
    plus the least-squares projection (over the WFS pupil) of the per-pixel
    part. The fitted maps of the last call are kept, detached, in meters:
    ``opd`` (B, Nres, Nres) and its parts ``dm_opd`` and ``pixel_opd`` (None
    when absent), and the final per-frame loss in ``loss``.

    ``"pixel"`` mode has no per-mode scale, so it is exact for small
    aberrations but gets stuck for large ones (e.g. ~50 nm left at 200 nm RMS
    on an unmodulated Pyramid, where ``"dm"`` reaches ~3 nm); ``"dm+pixel"``
    starts from the DM part and refines it.

    ``mode_scale`` (required in the ``"dm"`` modes) is the typical amplitude of
    each mode in turbulence, from ``mode_scale_from_dataset``. Adam runs on
    ``u`` with ``c = mode_scale * u``: Adam moves every coordinate by about
    ``lr`` per step, so without the scale, modes that are small in turbulence
    and poorly sensed are pushed as hard as the large low-order ones, which
    traps the fit in local minima (e.g. on an unmodulated Pyramid). With
    ``mode_scale`` in per-nm units, ``lr`` is roughly nm RMS per step. The
    per-pixel map is kept in nm, so in ``"pixel"`` mode ``lr`` is nm per pixel
    per step. The learning rate follows a cosine decay to 0 over ``n_iter``.

    The WFS and DM must be in eval mode (frozen); gradients only reach the
    fitted OPD. The model frame is noise-free and uses the WFS's own pupil.
    No random numbers are drawn, as ``evaluate_many`` requires.

    Examples
    --------
    >>> import torch
    >>> from mmengine import Config
    >>> from AI4AO import PyramidWFS, DeformableMirror, PhaseDataset, GDReconstructor
    >>> p = Config.fromfile("Tutorials/basics/wfs_params_exp.py")
    >>> wfs = PyramidWFS(dict(p.WFSParams, useNoise=False), "cpu").eval()
    >>> dm = DeformableMirror(p.WFSParams, p.DMParams, "cpu").eval()
    Total number of actuators: 101
    Total number of actuators: 101
    >>> M2C = dm.MakeZernikeM2C()[:, :20]
    >>> dataset = PhaseDataset(p.WFSParams, p.AtmosParams, p.LoopParams, p.DMParams, "cpu")
    >>> torch.manual_seed(0)  # doctest: +ELLIPSIS
    <torch._C.Generator object at ...>
    >>> sigma = GDReconstructor.mode_scale_from_dataset(dataset, dm, M2C, n_batches=2)
    >>> gd = GDReconstructor(wfs, dm, M2C, mode_scale=sigma, n_iter=50)
    >>> c_true = torch.zeros(1, 20)
    >>> c_true[0, 4] = 0.3
    >>> c = gd(wfs(dm.GetDMShape(c_true @ M2C.T)))       # (B, nModes)
    >>> bool((c - c_true).abs().max() < 0.03)
    True
    """

    OPD_MODES = ("dm", "pixel", "dm+pixel")

    def __init__(self, wfs, dm, M2C=None, opd_mode="dm", mode_scale=None,
                 n_iter=300, lr=30.0, residual_lr_factor=0.1):
        if opd_mode not in self.OPD_MODES:
            raise ValueError(f"opd_mode must be one of {self.OPD_MODES}, got {opd_mode!r}")
        self.wfs = wfs
        self.dm = dm
        device = dm.IF.device
        self.M2C = (torch.eye(int(dm.totalAct), device=device) if M2C is None
                    else torch.as_tensor(M2C, dtype=torch.float32, device=device))
        self.opd_mode = opd_mode
        self.uses_dm = opd_mode != "pixel"
        self.uses_pixel = opd_mode != "dm"
        if self.uses_dm:
            if mode_scale is None:
                raise ValueError(f"opd_mode={opd_mode!r} needs mode_scale: "
                                 "use GDReconstructor.mode_scale_from_dataset(...)")
            self.mode_scale = torch.as_tensor(mode_scale, dtype=torch.float32, device=device).reshape(1, -1)
            if self.mode_scale.shape[1] != self.M2C.shape[1]:
                raise ValueError(f"mode_scale has {self.mode_scale.shape[1]} entries for "
                                 f"{self.M2C.shape[1]} modes")
        self.n_iter = n_iter
        self.lr = lr
        self.residual_lr_factor = residual_lr_factor
        self._projection = None
        self.opd = self.dm_opd = self.pixel_opd = self.loss = None

    @staticmethod
    @torch.no_grad()
    def mode_scale_from_dataset(dataset, dm, M2C=None, n_batches=40):
        """
        Typical amplitude of each mode in turbulence, shape ``(nModes,)``: the
        RMS over ``n_batches`` batches of ``dataset`` of each screen's modal
        coefficients, each screen first normalized to 1 nm RMS over
        ``dm.pupil``. The projection is the least-squares one,
        ``pinv(dm.GetDMShape(M2C.T))``, as ``Trainer.z_inv``. Draws from the
        global RNG: seed before calling for a reproducible result.
        """
        device = dm.IF.device
        M2C = torch.eye(int(dm.totalAct), device=device) if M2C is None else M2C
        modes = dm.GetDMShape(M2C.T)                                    # (nModes, Nres, Nres)
        z_inv = torch.linalg.pinv(modes.flatten(start_dim=-2))
        pupil = dm.pupil.bool()
        coefs = []
        for _ in range(n_batches):
            opd = dataset[0]["opd"]
            c = opd.flatten(start_dim=-2) @ z_inv
            v = dm.GetDMShape(c @ M2C.T)[:, pupil]
            rms_nm = (v - v.mean(dim=-1, keepdim=True)).pow(2).mean(dim=-1).sqrt() * 1e9
            coefs.append(c / rms_nm[:, None])
        return torch.cat(coefs).pow(2).mean(dim=0).sqrt()

    def _project_to_dm(self, opd):
        """Least-squares DM coefficients of an OPD over the WFS pupil."""
        pupil = self.wfs.pupil.bool()
        if self._projection is None:
            basis = self.dm.GetDMShape(self.M2C.T)[:, pupil]            # (nModes, Npupil)
            self._projection = torch.linalg.pinv(basis)
        v = opd[:, pupil]
        return (v - v.mean(dim=-1, keepdim=True)) @ self._projection

    def __call__(self, wfs_frames, residual_opd=None):
        if self.wfs.training or self.dm.training:
            raise RuntimeError("GDReconstructor needs a frozen twin: call wfs.eval() and dm.eval()")
        frames = wfs_frames.detach()
        B = frames.shape[0]
        Nres = self.wfs.Nres
        device = frames.device

        groups, u, p = [], None, None
        if self.uses_dm:
            u = torch.zeros(B, self.M2C.shape[1], device=device, requires_grad=True)
            groups.append({"params": [u], "lr": self.lr})
        if self.uses_pixel:
            p = torch.zeros(B, Nres, Nres, device=device, requires_grad=True)   # nm
            factor = self.residual_lr_factor if self.uses_dm else 1.0
            groups.append({"params": [p], "lr": self.lr * factor})

        def opd_parts():
            dm_opd = self.dm.GetDMShape((self.mode_scale * u) @ self.M2C.T) if u is not None else None
            pixel_opd = p * 1e-9 if p is not None else None
            return dm_opd, pixel_opd

        norm = (frames ** 2).sum(dim=(-2, -1))
        use_noise = self.wfs.useNoise
        self.wfs.useNoise = False
        try:
            with torch.enable_grad():
                optimizer = torch.optim.Adam(groups)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.n_iter)
                for _ in range(self.n_iter):
                    dm_opd, pixel_opd = opd_parts()
                    opd = sum(x for x in (dm_opd, pixel_opd) if x is not None)
                    per_frame = ((self.wfs(opd) - frames) ** 2).sum(dim=(-2, -1)) / norm
                    optimizer.zero_grad()
                    per_frame.sum().backward()
                    optimizer.step()
                    scheduler.step()
            with torch.no_grad():
                dm_opd, pixel_opd = opd_parts()
                self.opd = sum(x for x in (dm_opd, pixel_opd) if x is not None)
                self.dm_opd, self.pixel_opd = dm_opd, pixel_opd
                self.loss = ((self.wfs(self.opd) - frames) ** 2).sum(dim=(-2, -1)) / norm
        finally:
            self.wfs.useNoise = use_noise

        with torch.no_grad():
            coefs = 0
            if self.uses_dm:
                coefs = (self.mode_scale * u).detach()
            if self.uses_pixel:
                coefs = coefs + self._project_to_dm(self.pixel_opd)
            return coefs
