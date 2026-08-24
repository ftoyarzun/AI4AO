import torch # type: ignore[import]
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from .Utils import imshow
from IPython.display import display, clear_output


def select_threshold(frame):
    W,H = frame.shape
    rgb_img = np.zeros((W,H,3))
    
    threshold = {"value": np.max(frame)*0.1}

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    # Initial thresholded image
    binary = frame > threshold["value"]

    rgb_img[...,0] = binary
    rgb_img[...,1] = frame/frame.mean()
    rgb_img[...,2] = frame/frame.mean()

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
        rgb_img[...,0] = binary
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


def show_frame_for_approval(frame):
    result = {"accepted": None}

    fig, ax = imshow(frame, figsize=(6,6))
    plt.subplots_adjust(bottom=0.2)

    # ax.imshow(frame, cmap="gray")
    # ax.set_title("Accept this frame?")
    # ax.axis("off")

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


def calibrate_pupil_positions(wfs, target_frame):


    
    modulation = wfs.modulation
    wfs.modulation = 5

    final_train_loss = 0
    target_frame = torch.from_numpy(target_frame).to(device = wfs.device, dtype = torch.float32)
    target_frame /= target_frame.sum()
    TrainRunNb = 200
    loss = torch.nn.MSELoss()
    # plt.ion()

    fig, ax = plt.subplots(figsize=(6, 6))
    img = ax.imshow(target_frame.cpu().detach().numpy())
    fig.colorbar(img)

    plt.show(block=False)

    optimizer = torch.optim.AdamW(
        [wfs.mainSlope, wfs.maskShifts],
        3e-3,
        fused=True
    )

    wfs.train()

    for u in range(TrainRunNb):

        optimizer.zero_grad()

        wfs.BuildMask()
        wfs.BuildReferenceIntensity()

        digital_image = wfs.reference_intensity

        l = (torch.abs(target_frame - digital_image) ** 2).sum()

        l.backward()
        optimizer.step()

        if u % 10 == 0:

            img.set_data(
                (target_frame - wfs.reference_intensity)
                .detach()
                .cpu()
                .numpy()
            )

            # Don't use np.min/max on the AxesImage object
            data = img.get_array()
            img.set_clim(
                vmin=np.min(data),
                vmax=np.max(data)
            )

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            plt.pause(0.01)

    plt.close(fig)
    wfs.modulation = modulation


def MakePupil(get_frame_function, dm_shm, amp = 0.05, n_iter = 5000):

    frame = get_frame_function() * 0

    for i in range(n_iter):
        coefs = np.random.randn(*dm_shm.get_data().shape) * amp
        coefs = coefs.astype(np.float32)

        dm_shm.set_data(coefs)
        frame += get_frame_function()

    frame /= frame.sum()

    dm_shm.set_data(coefs * 0)

    return frame

class BenchCalibrator:
    def __init__(self):

        pass

    

    def MakePupil(self, get_frame_function, dm_shm, wfs, amp = 0.05, n_iter = 5000):

        frame = get_frame_function() * 0

        for i in range(n_iter):
            coefs = np.random.randn(*dm_shm.get_data().shape) * amp
            coefs = coefs.astype(np.float32)

            dm_shm.set_data(coefs)
            frame += get_frame_function()

        dm_shm.set_data(coefs * 0)

        pupils = self.GetPupils(frame)
        npad = self.Extract_pupils_pad // 2
        pupil = pupils.sum(axis = 0)[npad:-npad,npad:-npad]
        pupil /= np.mean(pupil[wfs.pupil.cpu().numpy()])

        return pupil