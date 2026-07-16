#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 16:45:31 2025

@author: ptrouve
"""

from AI4AO.TorchPropagator import WFS
from AI4AO.PhaseEstimators import *
from AI4AO.MaskGeneration import MaskManager
from AI4AO.FramePreprocess import FramePreprocess
import torch.nn as nn # type: ignore[import]
import torch # type: ignore[import]
import os
from AI4AO.Constants import mask_types_list, param_needed_mask_list


class End2EndWFS(nn.Module):
    def __init__(self, wfsParams, atmosParams, device):
        super().__init__()
        self.device = device
        self.Nmodes = wfsParams["Nmodes"]
        self.N = wfsParams["Nres"] * wfsParams["sampling"]
        self.Nres = wfsParams["Nres"]
        self.WFS = WFS(wfsParams, device)
        self.ReconstructionType = wfsParams["Reconstruction"]
        self.OptimizeMask = False


        # Phase Estimator Selection
        self.PhaseEstimator = self._build_phase_estimator(
            wfsParams, atmosParams, device
        )

        # Initialize Mask Manager
        self.maskManager = MaskManager(wfsParams, device, self.WFS)
        self.maskManager.update_masks()

        self.framePreprocessor = FramePreprocess(
            wfsParams, atmosParams, device, self.maskManager
        )

    def _build_phase_estimator(self, wfsParams, atmosParams, device):
        if wfsParams["Reconstruction"] == "Linear":
            return LinearEstimator(self.WFS)
        elif wfsParams["Reconstruction"] == "SimpleNet":
            return SimpleNet(wfsParams, atmosParams, device).to(self.device)
        elif wfsParams["Reconstruction"] == "DataFusion":
            return DataFusion(wfsParams, atmosParams, device).to(self.device)
        elif wfsParams["Reconstruction"] == "Papyrus":
            return Papyrus1stStage(wfsParams, atmosParams, device).to(self.device)
        elif wfsParams["Reconstruction"] == "Papyrus2ndstage":
            return Papyrus2ndStage(wfsParams, atmosParams, device).to(self.device)
        elif wfsParams["Reconstruction"] == "Rama":
            return Rama(wfsParams, atmosParams, device).to(self.device)
        elif wfsParams["Reconstruction"] == "PapyrusPhase":
            return PapyrusPhase(self.WFS.pupil, device).to(self.device)
        elif wfsParams["Reconstruction"] == "VGGNet":
            return VGGNet(wfsParams).to(self.device)
        elif wfsParams["Reconstruction"] == "UNet":
            return UNetWithMLP(2, mlp_output_size=self.Nmodes).to(self.device)
        elif wfsParams["Reconstruction"] == "Transformer":
            return ViT_PyTorch(
                img_size=92,
                patch_size=8,
                in_channels=2,
                embed_dim=128,
                depth=8,
                num_heads=4,
                mlp_ratio=4.0,
                out_dim=self.Nmodes,
                dropout=0.1,
            ).to(self.device)
        else:
            raise ValueError(f"Unknown phase estimator: {wfsParams['Reconstruction']}")

    def forward(self, phase, pupil):
        if self.OptimizeMask:
            self.UpdateMask()
            self.Image = self.WFS.Propagator(phase, pupil)
            input_to_network = torch.clone(self.Image)
            input_to_network = self.framePreprocessor.ProcessFrame(input_to_network)

        else:
            with torch.no_grad():
                self.Image = self.WFS.Propagator(phase, pupil)
                input_to_network = torch.clone(self.Image)
                if self.ReconstructionType is not "Linear":
                    input_to_network = self.framePreprocessor.ProcessFrame(input_to_network)
                
        # self.focalPlaneImage = self.WFS.psf_with_noise

        EstimatedPhase = self.PhaseEstimator(input_to_network)
        return EstimatedPhase

    def UpdateMask(self):
        self.maskManager.update_masks()
        with torch.no_grad():
            self.WFS.BuildReferenceIntensity()
            self.framePreprocessor.ProcessReference(self.WFS.reference_intensity)

class AOLoop:
    def __init__(
        self,
        ParamsDict,
        End2EndWFS,
        z_FullRes,
        gain,
        leak,
        phaseTemplate,
        outputTemplate,
        photons,
        ron,
        start_after_iteration=0,
        modulation=0,
    ):

        self.End2EndWFS = End2EndWFS
        self.z_FullRes = z_FullRes

        if ParamsDict["Reconstruction"] == "Linear":
            # self.End2EndWFS.WFS.param = [torch.pi/2, torch.pi/2]
            # self.End2EndWFS.WFS.modulation = modulation
            # self.End2EndWFS.WFS.BuildPyramidMask()
            Nres = self.End2EndWFS.WFS.Nres
            self.End2EndWFS.WFS.BuildReconstructionMatrix(
                self.z_FullRes.view(-1, Nres, Nres)
            )
            self.End2EndWFS.WFS.BuildReferenceIntensity()

        self.gain = gain
        self.leak = leak
        self.z_estimated = torch.zeros_like(
            outputTemplate
        )  # Start with zero correction
        self.z_reconstructed = torch.zeros_like(phaseTemplate)
        self.residual_phase = torch.zeros_like(
            phaseTemplate
        )  # Start with the original phase
        self.pupil = self.End2EndWFS.WFS.pupil
        self.residual_variance = torch.var(
            phaseTemplate[:, self.pupil.bool()], dim=-1
        ).unsqueeze(-1)
        self.start_after_iteration = start_after_iteration

        out = self.End2EndWFS(phaseTemplate, self.End2EndWFS.WFS.pupil)
        self.images = self.End2EndWFS.Image
        self.End2EndWFS.WFS.SetPhotonsAndRON(photons, ron)
        self.iteration = 0
        self.filter = (
            (torch.arange(0, out.shape[-1]) < 100)
            .unsqueeze(0)
            .to(device=End2EndWFS.device)
        )

        self.long_exposure_psf = torch.zeros_like(self.End2EndWFS.WFS.psf_no_noise)

    def step(self, phase):
        self.iteration += 1
        # self.residual_phase = phase - self.z_reconstructed
        # self.residual_variance = torch.cat(
        #     (
        #         self.residual_variance,
        #         torch.var(self.residual_phase[:, self.pupil.bool()], dim=-1).unsqueeze(
        #             -1
        #         ),
        #     ),
        #     dim=1,
        # )
        # z_output = self.End2EndWFS(self.residual_phase, self.End2EndWFS.WFS.pupil)

        z_output = self.End2EndWFS(phase, self.End2EndWFS.WFS.pupil)
        self.images = self.End2EndWFS.Image

        if self.iteration > self.start_after_iteration:
            self.z_estimated = self.leak * self.z_estimated + self.gain * z_output
            self.z_reconstructed = torch.matmul(
                self.z_estimated, self.z_FullRes
            ).view_as(self.z_reconstructed)

        self.residual_phase = phase - self.z_reconstructed
        self.residual_variance = torch.cat(
            (
                self.residual_variance,
                torch.var(self.residual_phase[:, self.pupil.bool()], dim=-1).unsqueeze(
                    -1
                ),
            ),
            dim=1,
        )

        if self.iteration > self.start_after_iteration + 20:
            self.long_exposure_psf += self.End2EndWFS.WFS.psf_no_noise

    def ResetAOLoop(self):
        self.iteration = 0
        self.z_reconstructed = self.z_reconstructed * 0
        self.z_estimated = self.z_estimated * 0


class CheckpointManager:
    def __init__(
        self,
        model,
        WFSParams,
        TrainParams,
        checkpoint_path,
        optimizer_o=None,
        optimizer_n=None,
    ):
        self.model = model
        self.maskManager = model.maskManager
        self.optimizer_o = optimizer_o
        self.optimizer_n = optimizer_n
        self.WFSParams = WFSParams
        self.TrainParams = TrainParams
        self.checkpoint_path = checkpoint_path

    def load(self, should_load_optimizer=True):
        """Load checkpoint from the given path"""

        if not os.path.exists(self.checkpoint_path):
            print("Starting from scratch")
            return

        self.load_network(self.checkpoint_path, should_load_optimizer)

        if self.WFSParams["MaskType"] in [
            "FreePhase",
            "FreePhaseTransmision",
            "ModalMask",
        ]:
            self.load_free_phaseMask(self.checkpoint_path, should_load_optimizer)

        if self.WFSParams["MaskType"] in ["FreeTransmision", "FreePhaseTransmision"]:
            self.load_free_transmisionMask(self.checkpoint_path, should_load_optimizer)

        if self.WFSParams["MaskType"] in param_needed_mask_list:
            self.load_parametric_mask(self.checkpoint_path, should_load_optimizer)

        if self.WFSParams["MaskType"] not in mask_types_list:
            raise ValueError(f"Unsupported mask type: {self.WFSParams['MaskType']}")

    def load_network(self, network_path=None, should_load_optimizer=True):

        if network_path is None:
            network_path = self.checkpoint_path

        if not os.path.exists(network_path):
            print(f"The path {network_path} does not exist")
            return

        checkpoint = torch.load(network_path)
        self.model.PhaseEstimator.load_state_dict(
            checkpoint["PhaseEstimator_state_dict"]
        )
        if should_load_optimizer:
            self.optimizer_n.load_state_dict(checkpoint["optimizer_n_state_dict"])
            for param_group in self.optimizer_n.param_groups:
                param_group["lr"] = self.TrainParams["lrn"]

    def load_free_phaseMask(self, mask_path=None, should_load_optimizer=True):
        if mask_path is None:
            mask_path = self.checkpoint_path

        if not os.path.exists(mask_path):
            print(f"The path {mask_path} does not exist")
            return

        checkpoint = torch.load(mask_path)
        self.maskManager.phaseMaskGenerator.load_state_dict(
            checkpoint["Phase_Mask_state_dict"]
        )
        if should_load_optimizer:
            self.optimizer_o.load_state_dict(checkpoint["optimizer_o_state_dict"])
            for param_group in self.optimizer_o.param_groups:
                param_group["lr"] = self.TrainParams["lro"]

    def load_free_transmisionMask(self, mask_path=None, should_load_optimizer=True):
        if mask_path is None:
            mask_path = self.checkpoint_path

        if not os.path.exists(mask_path):
            print(f"The path {mask_path} does not exist")
            return

        checkpoint = torch.load(mask_path)
        self.maskManager.transmisionMaskGenerator.load_state_dict(
            checkpoint["Transmision_Mask_state_dict"]
        )
        if should_load_optimizer:
            self.optimizer_o.load_state_dict(checkpoint["optimizer_o_state_dict"])
            for param_group in self.optimizer_o.param_groups:
                param_group["lr"] = self.TrainParams["lro"]

    def load_parametric_mask(self, mask_path=None, should_load_optimizer=True):
        if mask_path is None:
            mask_path = self.checkpoint_path

        if not os.path.exists(mask_path):
            print(f"The path {mask_path} does not exist")
            return

        checkpoint = torch.load(mask_path)
        self.maskManager.load_state_dict(checkpoint["Mask_state_dict"])
        if should_load_optimizer:
            self.optimizer_o.load_state_dict(checkpoint["optimizer_o_state_dict"])
            for param_group in self.optimizer_o.param_groups:
                param_group["lr"] = self.TrainParams["lro"]

    def load_model(self, model_path=None, should_load_optimizer=True):
        if model_path is None:
            model_path = self.checkpoint_path

        if not os.path.exists(model_path):
            print(f"The path {model_path} does not exist")
            return

        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if not should_load_optimizer:
            return
        self.optimizer_o.load_state_dict(checkpoint["optimizer_o_state_dict"])
        self.optimizer_n.load_state_dict(checkpoint["optimizer_n_state_dict"])
        for param_group in self.optimizer_n.param_groups:
            param_group["lr"] = self.TrainParams["lrn"]
        for param_group in self.optimizer_o.param_groups:
            param_group["lr"] = self.TrainParams["lro"]

    def save(self, save_path=None):

        if save_path is None:
            save_path = self.checkpoint_path

        """Save checkpoint to the given path"""
        dict_to_save = {}
        dict_to_save["PhaseEstimator_state_dict"] = (
            self.model.PhaseEstimator.state_dict()
        )
        dict_to_save["optimizer_o_state_dict"] = self.optimizer_o.state_dict()
        dict_to_save["optimizer_n_state_dict"] = self.optimizer_n.state_dict()

        if self.WFSParams["MaskType"] in ["FreePhase", "FreePhaseTransmision"]:
            dict_to_save["Phase_Mask_state_dict"] = (
                self.maskManager.phaseMaskGenerator.state_dict()
            )

        if self.WFSParams["MaskType"] in ["FreeTransmision", "FreePhaseTransmision"]:
            dict_to_save["Transmision_Mask_state_dict"] = (
                self.maskManager.transmisionMaskGenerator.state_dict()
            )

        if self.WFSParams["MaskType"] == "ModalMask":
            dict_to_save["Phase_Mask_state_dict"] = (
                self.maskManager.phaseMaskGenerator.state_dict()
            )

        if self.WFSParams["MaskType"] in param_needed_mask_list:
            dict_to_save["Mask_state_dict"] = self.maskManager.state_dict()

        if self.WFSParams["MaskType"] not in mask_types_list:
            raise ValueError(f"Unsupported mask type: {self.WFSParams['MaskType']}")

        torch.save(dict_to_save, save_path)

