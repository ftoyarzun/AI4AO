#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 16:24:10 2025

@author: ptrouve
"""
import torch

from PhaseDataset import PhaseDataset, PermanentPhaseDataset
from SimuEnd2EndWFS import End2EndWFS, CheckpointManager
from LossFunctions import (
    Custom_Loss_Function,
    Physics_loss,
    ResidualPhaseLoss,
    Relative_Loss_Function,
    WFSSignalLoss,
    LogResidualVarianceLoss,
    TestLoss,
    RMSELoss,
    EstimatedSignalVarianceLoss,
)
from MaskGeneration import MaskVisualizator
from mmengine import Config
import numpy as np
import matplotlib.pyplot as plt
import time
from TorchPropagator import Zernike
import copy


from tqdm import tqdm


def test(End2EndWFS, dataset, loss, TestRunNb, device="cuda"):

    wfs = End2EndWFS.WFS

    wfs_Pyramid_mod = copy.copy(wfs)
    wfs_Pyramid_mod.param = [1.57, 1.57]
    wfs_Pyramid_mod.BuildPyramidMask()
    wfs_Pyramid_mod.modulation = 5

    wfs_Pyramid = copy.copy(wfs)
    wfs_Pyramid.param = [1.57, 1.57]
    wfs_Pyramid.BuildPyramidMask()

    Nzernike = dataset.Nzernike

    z_FullRes = dataset.z_FullRes.permute(2, 0, 1).to(device=device)

    wfs_Pyramid_mod.BuildReconstructionMatrix(z_FullRes)
    wfs_Pyramid.BuildReconstructionMatrix(z_FullRes)

    wfs_Pyramid_mod.BuildReferenceIntensity()
    wfs_Pyramid.BuildReferenceIntensity()

    phaseGT, zernike, photons, ron, r0s, _, _ = dataset.GetMovingWavefront(False)
    End2EndWFS.WFS.SetPhotonsAndRON(photons, ron)

    plt.close("all")
    output = End2EndWFS(phaseGT[0, :, :])
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))

    img = axes[0].imshow(End2EndWFS.Image[0, :, :].cpu().detach().numpy())
    (line1,) = axes[1].plot(zernike[0, :].cpu().data.numpy())
    (line2,) = axes[1].plot(output[0, :].cpu().data.numpy())
    (line3,) = axes[1].plot(output[0, :].cpu().data.numpy())
    (line4,) = axes[1].plot(output[0, :].cpu().data.numpy())

    (line5,) = axes[2].plot(output[0, :].cpu().data.numpy())
    (line6,) = axes[2].plot(output[0, :].cpu().data.numpy())
    (line7,) = axes[2].plot(output[0, :].cpu().data.numpy())

    axes[1].set_ylim(-1, 1)
    axes[2].set_ylim(-1, 1)

    final_test_loss = 0

    End2EndWFS.eval()
    # End2EndWFS.train()

    with torch.no_grad():

        End2EndWFS.WFS.SetPhotonsAndRON(photons, ron)
        output = End2EndWFS(phaseGT)

        wfs_Pyramid_mod.SetPhotonsAndRON(photons, ron)
        test_frame_wfs_Zernike = wfs_Pyramid_mod.Propagator(phaseGT)
        estimated_phase_wfs_Zernike = wfs_Pyramid_mod.GetReconstructedPhase(
            test_frame_wfs_Zernike
        )

        wfs_Pyramid.SetPhotonsAndRON(photons, ron)
        test_frame_wfs_Pyramid = wfs_Pyramid.Propagator(phaseGT)
        estimated_phase_wfs_Pyramid = wfs_Pyramid.GetReconstructedPhase(
            test_frame_wfs_Pyramid
        )

        loss_NN = loss(output, zernike, r0s)
        loss_Zernike = loss(estimated_phase_wfs_Zernike, zernike, r0s)
        loss_Pyramid = loss(estimated_phase_wfs_Pyramid, zernike, r0s)

        final_test_loss = loss_NN + final_test_loss

        # print(" Test Run n°  {}, test loss : {:.4f}  \n".format(u, loss_NN), end="")
        print(
            f"NN loss = {loss_NN.item():.4f}, Zernike WFS loss = {loss_Zernike.item():.4f}, Pyramid WFS loss = {loss_Pyramid.item():.4f}"
        )

        for u in range(10):

            # plot of an example within the batch

            plt.subplot(1, 3, 1)
            plt.title("WFS image")
            img.set_data(End2EndWFS.Image[u, :, :].cpu().detach().numpy())
            img.set_clim(vmin=np.min(img.get_array()), vmax=np.max(img.get_array()))
            # plt.colorbar()
            # plt.draw()

            plt.subplot(1, 3, 2)
            plt.title("Phase estimation")
            line1.set_ydata(zernike[u, :].cpu().data.numpy())
            line2.set_ydata(output[u, :].cpu().data.numpy())
            line3.set_ydata(estimated_phase_wfs_Zernike[..., u, :].cpu().data.numpy())
            line4.set_ydata(estimated_phase_wfs_Pyramid[..., u, :].cpu().data.numpy())
            plt.xlabel("Zernike mode index")
            plt.ylabel("Zernike mode Amplitude")
            plt.title(
                f"NN loss = {loss_NN.item():.4f}, Zernike WFS loss = {loss_Zernike.item():.4f}, Pyramid WFS loss = {loss_Pyramid.item():.4f}"
            )
            plt.legend(
                [
                    "Ground truth",
                    "CNN",
                    "Linear reconctructor Zernike WFS",
                    "Linear reconstructor Pyramid WFS",
                ]
            )

            plt.subplot(1, 3, 3)
            plt.title("Phase residuals")
            line5.set_ydata(
                zernike[u, :].cpu().data.numpy() - output[u, :].cpu().data.numpy()
            )
            line6.set_ydata(
                zernike[u, :].cpu().data.numpy()
                - estimated_phase_wfs_Zernike[..., u, :].cpu().data.numpy()
            )
            line7.set_ydata(
                zernike[u, :].cpu().data.numpy()
                - estimated_phase_wfs_Pyramid[..., u, :].cpu().data.numpy()
            )
            plt.xlabel("Zernike mode index")
            plt.ylabel("Zernike mode Amplitude")
            plt.legend(
                [
                    "CNN",
                    "Linear reconctructor Zernike WFS",
                    "Linear reconstructor Pyramid WFS",
                ]
            )

            plt.show()

            plt.pause(0.5)

    return final_test_loss / TestRunNb


def train_closed_loop(
    End2EndWFS,
    dataset,
    loss,
    TrainRunNb,
    optimizer_o,
    optimizer_n,
    gain=0.3,
    num_iterations=5,
    device="cuda",
):

    gain = torch.rand((dataset.Nphases, 1), device=device) * 0.5 + 0.2
    leak = torch.rand((dataset.Nphases, 1), device=device) * 0.05 + 0.95

    if visual:
        maskVisualizator.SetCanvas()

    z_FullRes = (
        dataset.z_FullRes.to(device, dtype=torch.float32)
        .view(-1, dataset.Nzernike)
        .transpose(0, 1)
    )

    End2EndWFS.train()

    progressBar = tqdm(range(TrainRunNb // num_iterations))

    dataset.ResetMovingWavefront()
    phaseGT, zernike, photons, ron, r0s, _, _ = dataset.GetMovingWavefront(
        generateClosedLoop=(num_iterations == 1)
    )
    End2EndWFS.WFS.SetPhotonsAndRON(photons, ron)

    for u in progressBar:

        dataset.ResetMovingWavefront()
        phaseGT, zernike, photons, ron, r0s, _, _ = dataset.GetMovingWavefront(
            generateClosedLoop=(num_iterations == 1)
        )
        End2EndWFS.WFS.SetPhotonsAndRON(photons, ron)

        # Closed-loop correction
        z_estimated = torch.zeros_like(zernike)  # Start with zero correction
        z_reconstructed = torch.zeros_like(phaseGT)
        z_reconstructed_iter = torch.zeros_like(phaseGT)
        z_prev = torch.zeros_like(phaseGT)

        total_loss = 0

        for i in range(num_iterations):
            # Get new WFS images after applying the correction
            if i > 1:
                phaseGT, zernike, _, _, _, _, _ = dataset.GetMovingWavefront(
                    generateClosedLoop=(num_iterations == 1)
                )
            residual_phase = phaseGT - z_reconstructed
            Ze = torch.matmul(residual_phase.flatten(1, 2), dataset.invZ)
            # Predict Zernike coefficients and update estimate

            z_output = End2EndWFS(residual_phase)

            z_estimated = (
                z_estimated * leak + gain * z_output
            )  # Apply correction with gain

            # Convert Zernike coefficients to full-resolution wavefront
            z_reconstructed = torch.matmul(z_estimated, z_FullRes).view_as(phaseGT)
            z_prev = z_reconstructed_iter.detach()
            z_reconstructed_iter = torch.matmul(z_output, z_FullRes).view_as(phaseGT)

            # Compute loss for this iteration

            convergence_ratio = 1 - (1 - gain.unsqueeze(1)) ** (i + 1)

            # total_loss += loss_relative(z_reconstructed_iter, residual_phase) * 1 / num_iterations
            # total_loss += loss_relative(z_reconstructed, phaseGT * convergence_ratio) * 1 / num_iterations

            # total_loss += loss_variance_penalty(z_reconstructed_iter - z_prev) * 1

            # total_loss += loss_variance(phaseGT * convergence_ratio - z_reconstructed) / num_iterations
            total_loss += (
                loss_variance(residual_phase - z_reconstructed_iter) / num_iterations
            )

        # **Backpropagation**

        optimizer_n.zero_grad(set_to_none=True)
        optimizer_o.zero_grad(set_to_none=True)

        total_loss.backward()

        if End2EndWFS.OptimizeMask:
            optimizer_o.step()

        optimizer_n.step()

        # **Track loss and parameters**

        loss_tracker[u] = total_loss.detach()
        # param_tracker[u] = End2EndWFS.maskManager.param.detach()

        if u % (100 // num_iterations) == 1:
            pass
            progressBar.set_postfix({"Loss": float(total_loss)})
            if visual:
                # maskVisualizator.update_plots(zernike, z_estimated)
                maskVisualizator.update_plots(Ze, z_output)
                maskVisualizator.show()

    return loss_tracker.cpu().numpy(), param_tracker.cpu().numpy()


def closeLoop():

    gain = 0.3
    dataset.Nphases = 1
    dataset.ResetMovingWavefront()
    phaseGT, zernike, photons, ron, r0s, _, _ = dataset.GetMovingWavefront(
        generateClosedLoop=True
    )
    Trained_End2EndWFS.WFS.SetPhotonsAndRON(photons, ron)

    fig, ax = plt.subplots(1, 4)
    img1 = ax[0].imshow(phaseGT[0].cpu().detach())
    img2 = ax[1].imshow(phaseGT[0].cpu().detach())
    img3 = ax[2].imshow(phaseGT[0].cpu().detach())
    img4 = ax[3].imshow(Trained_End2EndWFS.Image[0].cpu().detach())

    Trained_End2EndWFS.eval()

    with torch.no_grad():

        a = time.perf_counter()

        z_output = Trained_End2EndWFS(phaseGT)

        # Closed-loop correction
        z_estimated = torch.zeros_like(z_output)  # Start with zero correction
        z_reconstructed = torch.zeros_like(phaseGT)

        for i in range(500):
            a = time.perf_counter()
            # Get new WFS images after applying the correction
            if i > 0:
                phaseGT, zernike, _, _, _, _, _ = dataset.GetMovingWavefront()
            # residual_phase = phaseGT - z_reconstructed
            residual_phase = phaseGT - z_reconstructed

            z_output = Trained_End2EndWFS(residual_phase)

            z_estimated = (
                z_estimated * 0.99 + gain * z_output
            )  # Apply correction with gain

            z_reconstructed = torch.matmul(z_estimated, z_FullRes).view_as(phaseGT)

            img1.set_data(phaseGT[0].cpu().detach().numpy())
            img1.set_clim(vmin=np.min(img1.get_array()), vmax=np.max(img1.get_array()))
            img2.set_data(z_reconstructed[0].cpu().detach().numpy())
            img2.set_clim(vmin=np.min(img1.get_array()), vmax=np.max(img1.get_array()))
            img3.set_data(residual_phase[0].cpu().detach().numpy())
            img3.set_clim(vmin=np.min(img1.get_array()), vmax=np.max(img1.get_array()))
            img4.set_data(Trained_End2EndWFS.Image[0].cpu().detach().numpy())
            img4.set_clim(vmin=np.min(img4.get_array()), vmax=np.max(img4.get_array()))

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)

            b = time.perf_counter()

            loop_freq = 1 / (b - a)
            input_variance = Trained_End2EndWFS.GetPhaseVariance(phaseGT[0]).item()
            output_variance = Trained_End2EndWFS.GetPhaseVariance(
                residual_phase[0]
            ).item()

            print(
                f"Loop frequency = {loop_freq:.1f} Hz. Input variance = {input_variance:.2f} Residual variance = {output_variance:.2f} ",
                end="\r",
                flush=True,
            )


if __name__ == "__main__":

    device = "cuda"  # set to "cpu" if Cuda is not available

    paramfile = "params_exp.py"  # file of experimental parameters

    gif_path = "free_mask_animation_2ndStage3.gif"

    # Config extraction
    AtmosParams = Config.fromfile(paramfile)["AtmosParams"]
    WFSParams = Config.fromfile(paramfile)["WFSParams"]
    LoopParams = Config.fromfile(paramfile)["LoopParams"]
    TrainParams = Config.fromfile(paramfile)["TrainParams"]

    # Dataset creation
    dataset = PhaseDataset(WFSParams, AtmosParams, LoopParams, device)
    z_FullRes = (
        dataset.z_FullRes.to(device, dtype=torch.float32)
        .view(-1, dataset.Nzernike)
        .transpose(0, 1)
    )

    # Initialisation of the system
    Trained_End2EndWFS = End2EndWFS(WFSParams, AtmosParams, device)

    # Setting the loss function
    loss_test = Custom_Loss_Function(degree=2, NZernike=WFSParams["Nzernike"])
    loss_relative = Relative_Loss_Function(degree=2)
    loss_WFSSignal = WFSSignalLoss()
    loss_residual = ResidualPhaseLoss(z_FullRes, dataset.pupil)
    loss_physics = Physics_loss(z_FullRes, dataset[0][0])
    loss = [loss_WFSSignal, loss_test, loss_residual]
    loss_mse = torch.nn.MSELoss()
    loss_variance = LogResidualVarianceLoss(dataset.pupil)
    loss_variance_penalty = EstimatedSignalVarianceLoss()

    loss_jonatan = RMSELoss()
    # loss_jonatan = #TestLoss(z_FullRes, dataset.pupil)

    # Optimization parameters (learning rate lr and nb of runs)
    lrn = TrainParams["lrn"]
    lro = TrainParams["lro"]

    # Number of training and testing run

    TrainRunNb = TrainParams["TrainRunNb"]
    TestRunNb = TrainParams["TestRunNb"]

    # Setting the optimizer (here Adam)
    optimizer_o = torch.optim.AdamW(
        Trained_End2EndWFS.maskManager.parameters(), lro, fused=True
    )
    optimizer_n = torch.optim.AdamW(
        Trained_End2EndWFS.PhaseEstimator.parameters(),
        lrn,
        fused=True,
        weight_decay=1e-2,
    )
    # optimizer_n = torch.optim.AdamW([
    # {
    #     "params": Trained_End2EndWFS.PhaseEstimator.papy1.parameters(),
    #     "lr": lrn * 1.0,
    #     "weight_decay": 1e-2   # normal decay
    # },
    # {
    #     "params": Trained_End2EndWFS.PhaseEstimator.papy2.parameters(),
    #     "lr": lrn * 1.0,
    #     "weight_decay": 1e-3   # smaller decay
    # },
    # {
    #     "params": Trained_End2EndWFS.PhaseEstimator.papy3.parameters(),
    #     "lr": lrn * 3.0,
    #     "weight_decay": 1e-4    # no decay for high order
    # }
    # ])

    checkpoint_path = "Test.pth"
    # checkpoint_path = "Papyrus_KL150_PapyrusBlock2.pth"
    checkpointManager = CheckpointManager(
        Trained_End2EndWFS,
        WFSParams,
        TrainParams,
        checkpoint_path,
        optimizer_o,
        optimizer_n,
    )

    # checkpointManager.load()
    # checkpointManager.load_parametric_mask('Papyrus.pth', should_load_optimizer = False)
    checkpointManager.load_network(should_load_optimizer=False)
    # checkpointManager.load_free_phaseMask('Pyramid_mask.pth')
    # checkpointManager.load_free_transmisionMask('PhaseTransmisionTest2.pth')
    # checkpointManager.load_model()

    Trained_End2EndWFS.maskManager.update_masks()
    Trained_End2EndWFS.WFS.BuildReferenceIntensity()
    Trained_End2EndWFS.OptimizeMask = TrainParams["OptimizeMask"]

    # Training part for parameters optimization
    num_iterations = 1
    loss_tracker = torch.zeros(TrainRunNb // num_iterations, device=device)
    param_tracker = torch.zeros(
        TrainRunNb // num_iterations, len(WFSParams["InitParam"]), device=device
    )

    maskVisualizator = MaskVisualizator(Trained_End2EndWFS, loss_tracker)
    visual = True

    # for param in Trained_End2EndWFS.PhaseEstimator.papy1.parameters():
    #     param.requires_grad = False
    # for param in Trained_End2EndWFS.PhaseEstimator.papy2.parameters():
    #     param.requires_grad = False
    # for param in Trained_End2EndWFS.PhaseEstimator.papy2.parameters():
    #     param.requires_grad = False

    total_params = sum(
        p.numel()
        for p in Trained_End2EndWFS.PhaseEstimator.parameters()
        if p.requires_grad
    )
    print(f"Total trainable parameters: {total_params:,}")

    # with torch.profiler.profile(
    # activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    # record_shapes=True,
    # profile_memory=True,
    # with_stack=True
    # ) as prof:

    train_loss, train_parameters = train_closed_loop(
        Trained_End2EndWFS,
        dataset,
        loss,
        TrainRunNb,
        optimizer_o,
        optimizer_n,
        gain=1,
        num_iterations=num_iterations,
        device=device,
    )

    # prof.export_chrome_trace("trace_new.json")

    # # Save network and parameters
    checkpointManager.save()

    # # Testing part
    # # permanentPhaseDataset = PermanentPhaseDataset()

    # # var_loss = ResidualPhaseLoss(dataset.z_FullRes, dataset.pupil)
    # var_loss = Custom_Loss_Function(degree = 2, NZernike = WFSParams["Nzernike"])

    # test_loss = test(Trained_End2EndWFS,dataset,loss_test,TestRunNb,device)
