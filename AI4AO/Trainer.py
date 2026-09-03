import os
from dataclasses import dataclass

import torch  # type: ignore[import]
import numpy as np
import torch.nn as nn # type: ignore[import]
import matplotlib.pyplot as plt
from tqdm import tqdm


@dataclass
class EvaluationResult:
    """Trajectories from a closed-loop evaluation rollout, each shaped (n_steps, ...)."""
    phase: torch.Tensor
    pupil: torch.Tensor
    phase_reconstructed: torch.Tensor
    residual_phase: torch.Tensor
    wfs_frames: torch.Tensor
    psfs: torch.Tensor
    z_output: torch.Tensor


class Trainer:
    def __init__(self, wfs, framePreprocessor, dm, M2C, phaseReconstructor, dataset, loss, optimizer):
        self.wfs = wfs
        self.framePreprocessor = framePreprocessor
        self.dm = dm
        self.M2C = M2C
        self.phaseReconstructor = phaseReconstructor
        self.dataset = dataset
        self.loss = loss
        self.optimizer = optimizer
        self.device = wfs.device

        self.z_inv = torch.linalg.pinv(dm(M2C.T).flatten(start_dim = -2))

    def train(self, training_steps = 1000, closed_loop_iterations = 1):

        if self.dm.training:
            self.dm.eval()
        if self.wfs.training:
            self.wfs.eval()

        loss_tracker = torch.zeros(training_steps // closed_loop_iterations, device=self.device)
        loss_tracker_ideal = torch.zeros(training_steps // closed_loop_iterations, device=self.device)

        self.phaseReconstructor.train()

        M2C_T = self.M2C.T

        progressBar = tqdm(range(training_steps // closed_loop_iterations))

        for u in progressBar:
            with torch.no_grad():
                batch = self.dataset[0]
                phaseGT = batch["phase"]
                pupilGT = batch["pupil"]
                gain = batch["loop_gain"]
                leak = batch["loop_leak"]
                photons = batch["nphotons"]
                ron = batch["ron"]

                modes = torch.matmul(phaseGT.flatten(start_dim = -2), self.z_inv)

                self.wfs.SetPhotonsAndRON(photons, ron)

                # Closed-loop correction
                z_estimated = torch.zeros_like(modes)  # Start with zero correction 
                z_buffer = torch.zeros_like(modes)  
                z_output = torch.zeros_like(modes)    
                phase_reconstructed = torch.zeros_like(phaseGT)
                phase_reconstructed_iter = torch.zeros_like(phaseGT)
                phase_reconstructed_ideal = torch.zeros_like(phaseGT)
                phase_reconstructed_iter_ideal = torch.zeros_like(phaseGT)

                total_loss = 0
                ideal_loss = 0

            for i in range(closed_loop_iterations):
                with torch.no_grad():
                    # Get new WFS images after applying the correction
                    if i > 0:
                        batch = self.dataset[i]
                        phaseGT = batch["phase"]
                        pupilGT = batch["pupil"]

                    # modes = torch.matmul(phaseGT.flatten(start_dim = -2), z_inv)
                    # phaseGT = dm(modes @ M2C.T)

                    residual_phase = phaseGT - phase_reconstructed 

                    modes = torch.matmul(phaseGT.flatten(start_dim = -2), self.z_inv)
                    Ze = torch.matmul(residual_phase.flatten(start_dim = -2), self.z_inv)


                    # Predict coefficients and update estimate
                    z_estimated = z_estimated * leak + gain * z_buffer  # Apply correction with gain
                    z_buffer = torch.clone(z_output)

                
                    wfs_frames = self.wfs(residual_phase, pupilGT)
                    preprocessed_frames = self.framePreprocessor.ProcessFrame(wfs_frames)
                z_output = self.phaseReconstructor(preprocessed_frames)
                
                # Convert modes coefficients to full-resolution wavefront
                phase_reconstructed = self.dm(z_estimated @ M2C_T)
                phase_reconstructed_iter = self.dm(z_output @ M2C_T)
                
                # phase_reconstructed_ideal = dm(modes @ M2C.T)
                phase_reconstructed_iter_ideal = self.dm(Ze @ M2C_T)
                
                
                # Compute loss for this iteration
                corrected_residual_phase = residual_phase - phase_reconstructed_iter
                total_loss += self.loss(Ze, z_output, residual_phase, corrected_residual_phase, wfs_frames) / closed_loop_iterations

                # Compute ideal loss for comparison
                with torch.no_grad():
                    ideal_corrected_residual_phase = residual_phase - phase_reconstructed_iter_ideal
                    ideal_loss += self.loss(Ze, Ze, residual_phase, ideal_corrected_residual_phase, wfs_frames) / closed_loop_iterations

                
            # **Backpropagation**
            self.optimizer.zero_grad(set_to_none = True)
            
            total_loss.backward()
                
            self.optimizer.step()
            
            # **Track loss and parameters**
            
            loss_tracker[u] = total_loss.detach()
            loss_tracker_ideal[u] = ideal_loss.detach()
            
            if u % (300 // closed_loop_iterations) == 1:
                lower_lim = max(0, u - 100 // closed_loop_iterations)
                progressBar.set_postfix({'Loss': float(loss_tracker[lower_lim:u].mean()), 'Loss_ideal': float(loss_tracker_ideal[lower_lim:u].mean())})

        return loss_tracker, loss_tracker_ideal

    @torch.no_grad()
    def evaluate(self, n_steps = 100, dataset = None, gain = None, leak = None, psf_sampling = 4, psf_fov = 20):
        """Run a closed-loop rollout with no backprop and no pupil noise, for inspection/plotting."""
        dataset = dataset if dataset is not None else self.dataset

        if self.dm.training:
            self.dm.eval()
        if self.wfs.training:
            self.wfs.eval()

        M2C_T = self.M2C.T
        Nmodes = self.z_inv.shape[-1]

        self.phaseReconstructor.eval()

        batch = dataset[0]
        phaseGT = batch["phase"]
        pupilGT = batch["pupil"]
        gain = gain if gain is not None else batch["loop_gain"]
        leak = leak if leak is not None else batch["loop_leak"]
        photons = batch["nphotons"]
        ron = batch["ron"]

        self.wfs.SetPhotonsAndRON(photons, ron)

        z_estimated = torch.zeros(phaseGT.shape[0], Nmodes, device=self.device)
        z_buffer = torch.zeros_like(z_estimated)
        z_output = torch.zeros_like(z_estimated)
        phase_reconstructed = torch.zeros_like(phaseGT)

        phases, pupils, reconstructed, residuals, frames, psfs, outputs = [], [], [], [], [], [], []

        for i in range(n_steps):
            if i > 0:
                batch = dataset[i]
                phaseGT = batch["phase"]
                pupilGT = batch["pupil"]

            residual_phase = phaseGT - phase_reconstructed

            
            

            wfs_frames = self.wfs(residual_phase, pupilGT)
            psf = self.wfs.GetPSF(residual_phase, pupilGT, psf_sampling, psf_fov)
            preprocessed_frames = self.framePreprocessor.ProcessFrame(wfs_frames, False)
            z_output = self.phaseReconstructor(preprocessed_frames)

            z_buffer = torch.clone(z_output)


            if i > n_steps * 0.3:
                z_estimated = z_estimated * leak + gain * z_buffer

            phase_reconstructed = self.dm(z_estimated @ M2C_T)

            phases.append(phaseGT)
            pupils.append(pupilGT)
            reconstructed.append(phase_reconstructed)
            residuals.append(residual_phase)
            frames.append(wfs_frames)
            psfs.append(psf)
            outputs.append(z_output)

        return EvaluationResult(
            phase = torch.stack(phases),
            pupil = torch.stack(pupils),
            phase_reconstructed = torch.stack(reconstructed),
            residual_phase = torch.stack(residuals),
            wfs_frames = torch.stack(frames),
            psfs = torch.stack(psfs),
            z_output = torch.stack(outputs),
        )

    def plot_losses(self, loss_tracker, loss_tracker_ideal = None, smoothing_window = 100, ylim = None, log_x = False, show = True):
        losses = loss_tracker.detach().cpu().numpy()
        smoothed = np.convolve(losses, np.ones(smoothing_window) / smoothing_window, "valid")

        fig, ax = plt.subplots()
        plot_fn = ax.semilogx if log_x else ax.plot
        plot_fn(smoothed, label = "Training loss")

        if loss_tracker_ideal is not None:
            ideal = loss_tracker_ideal.detach().cpu().numpy()
            smoothed_ideal = np.convolve(ideal, np.ones(smoothing_window) / smoothing_window, "valid")
            plot_fn(smoothed_ideal, label = "Ideal loss")

        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"Loss = $\ln(\mathrm{var}(\phi - \phi_{est}))$")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend()

        if show:
            plt.show()

        return fig, ax

    def save_checkpoint(self, path, **extra_state):
        """Save the reconstructor + optimizer state. Extra keyword args (e.g. step, loss_tracker,
        TrainParams) are stored alongside for reference and returned as-is by load_checkpoint."""
        checkpoint = {
            "phase_reconstructor_state_dict": self.phaseReconstructor.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        checkpoint.update(extra_state)
        torch.save(checkpoint, path)

    def load_checkpoint(self, path, load_optimizer = True):
        """Load the reconstructor (+ optimizer) state from path. """
        if not os.path.exists(path):
            print(f"No checkpoint found at {path}, starting from scratch")
            return None

        checkpoint = torch.load(path, map_location = self.device)
        self.phaseReconstructor.load_state_dict(checkpoint["phase_reconstructor_state_dict"])

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                        