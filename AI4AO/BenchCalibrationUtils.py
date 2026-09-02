import torch # type: ignore[import]
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from .Utils import imshow
from .TwinCalibrator import TwinCalibrator


class BenchCalibrator:
    """
    Live-hardware calibration helpers: grabbing frames from a bench camera
    and poking a real DM through shared memory. Wraps `get_frame_function`
    (a callable that returns one camera frame) and `dm_shm` (a DM
    shared-memory handle exposing `.get_data()`/`.set_data()`) so callers
    stop threading them through every call.
    """

    def __init__(self, get_frame_function, dm_shm, device=None):
        self.get_frame_function = get_frame_function
        self.dm_shm = dm_shm
        self.device = device

    def grab_frame(self, n_avg=1):
        frame = self.get_frame_function() * 0.0
        for _ in range(n_avg):
            frame += self.get_frame_function()
        return frame / n_avg

    @staticmethod
    def select_threshold(frame):
        W, H = frame.shape
        rgb_img = np.zeros((W, H, 3))

        threshold = {"value": np.max(frame) * 0.1}

        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)

        # Initial thresholded image
        binary = frame > threshold["value"]

        rgb_img[..., 0] = binary
        rgb_img[..., 1] = frame / frame.mean()
        rgb_img[..., 2] = frame / frame.mean()

        img = ax.imshow(rgb_img)
        img.set_clim(vmin=0, vmax=1)
        ax.set_title(f"Threshold = {threshold['value']:.3f}")
        ax.axis("off")

        # Slider
        slider_ax = plt.axes([0.2, 0.08, 0.6, 0.04])

        slider = Slider(
            slider_ax,
            "Threshold",
            valmin=float(frame.min()),
            valmax=float(frame.max()),
            valinit=threshold["value"],
        )

        def update(val):
            threshold["value"] = val

            binary = frame > val
            rgb_img[..., 0] = binary
            img.set_data(rgb_img)

            ax.set_title(f"Threshold = {val:.3f}")

            fig.canvas.draw_idle()

        slider.on_changed(update)

        # Accept button
        button_ax = plt.axes([0.4, 0.01, 0.2, 0.05])
        button = Button(button_ax, "Accept")

        def accept(event):
            plt.close(fig)

        button.on_clicked(accept)

        plt.show()

        return frame > threshold["value"]

    @staticmethod
    def confirm(frame):
        result = {"accepted": None}

        fig, ax = imshow(frame, figsize=(6, 6))
        plt.subplots_adjust(bottom=0.2)

        # Buttons
        ax_accept = plt.axes([0.25, 0.05, 0.2, 0.075])
        ax_reject = plt.axes([0.55, 0.05, 0.2, 0.075])

        btn_accept = Button(ax_accept, "Accept")
        btn_reject = Button(ax_reject, "Reject")

        def accept(event):
            result["accepted"] = True
            plt.close(fig)

        def reject(event):
            result["accepted"] = False
            plt.close(fig)

        btn_accept.on_clicked(accept)
        btn_reject.on_clicked(reject)

        plt.show()

        return result["accepted"]

    def calibrate_pupil_positions(self, wfs, target_frame, lr=3e-3, n_iter=200, live_plot=True):
        """
        Fits `wfs.mainSlope`/`wfs.maskShifts` to a live bench `target_frame`
        by delegating to `TwinCalibrator.fit_pupil_to_reference` on a
        scratch calibrator (no DM needed for this fit).
        """
        modulation = wfs.modulation
        wfs.modulation = 5

        target_frame = torch.from_numpy(target_frame).to(device=wfs.device, dtype=torch.float32)
        target_frame = target_frame / target_frame.sum()

        calibrator = TwinCalibrator(wfs, dm=None, device=wfs.device)
        final_loss = calibrator.fit_pupil_to_reference(
            target_frame,
            [wfs.mainSlope, wfs.maskShifts],
            lr=lr,
            n_iter=n_iter,
            live_plot=live_plot,
        )

        wfs.modulation = modulation

        return final_loss

    def make_pupil(self, amp=0.05, n_iter=5000):
        frame = self.get_frame_function() * 0

        for i in range(n_iter):
            coefs = np.random.randn(*self.dm_shm.get_data().shape) * amp
            coefs = coefs.astype(np.float32)

            self.dm_shm.set_data(coefs)
            frame += self.get_frame_function()

        frame /= frame.sum()

        self.dm_shm.set_data(coefs * 0)

        return frame
