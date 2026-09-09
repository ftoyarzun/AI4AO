import torch  # type: ignore[import]
import torch.nn as nn  # type: ignore[import]

from .Utils import MakePupil
from .PhaseDataset import Zernike
from .paths import ensure_parent

class DeformableMirror(nn.Module):
    def __init__(self, WFSDict, DMDict, device, offset_to_fit_number_of_actuators = 0.2, misreg = None, per_actuator_calibration = False):
        super().__init__()

        self.initialized = False

        self.device = device
        self.Nres =         WFSDict["Nres"]
        self.D =            WFSDict["D"]
        self.wavelength =   WFSDict["Wavelength"]
        self.wavenumber =   2 * torch.pi / self.wavelength

        self.offset_to_fit_number_of_actuators = offset_to_fit_number_of_actuators

        self.Nact =    DMDict["Nactuator"]
        self.nModes =  DMDict["Nmodes"]
        self.flip_lr = DMDict["FlipLeftRight"]
        self.flip_tb = DMDict["FlipTopBottom"]
        self.flip_matrix = torch.tensor([[-1 if self.flip_lr else 1, -1 if self.flip_tb else 1]], device = self.device).unsqueeze(dim = -1).unsqueeze(dim = -1)

        # Gates only how moffatParameter/sign/mechCoupling are USED in
        # MakeZonalModes (independently vs. tied via a differentiable mean),
        # not what is stored -- their setters always keep a full length-
        # totalAct vector regardless of this flag.
        self.per_actuator_calibration = per_actuator_calibration

        self._rotationAngle = nn.Parameter(torch.empty(1, device=self.device))
        self._grid_shift = nn.Parameter(torch.empty((1,2,1,1), device=self.device))
        self._radialScaling = nn.Parameter(torch.empty(1, device=self.device))
        self._tangentialScaling = nn.Parameter(torch.empty(1, device=self.device))
        self._anamorphosisAngle = nn.Parameter(torch.empty(1, device=self.device))

        self.pupil = MakePupil(self.Nres, self.device)

        # totalAct must be known before the per-actuator parameters below are
        # created, so the actuator grid is built here rather than at the end.
        self.MakeActGrid()

        self._moffatParameter = nn.Parameter(torch.empty(self.totalAct, device=self.device))
        self._sign = nn.Parameter(torch.empty(self.totalAct, device=self.device))
        self._mechCoupling = nn.Parameter(torch.empty(self.totalAct, device=self.device))

        self.moffatParameter = torch.tensor([DMDict["moffatParam"]], device=self.device, dtype=torch.float32)
        self.sign = torch.tensor([DMDict["signedAmplitude"]], device=self.device, dtype=torch.float32)
        self.mechCoupling = torch.tensor([DMDict["MechCoupling"]], device=self.device, dtype=torch.float32)

        if misreg is None:
            self.rotationAngle = torch.tensor([0], device=self.device, dtype=torch.float32)
            self.grid_shift = torch.tensor([[0.,0.]], device=self.device, dtype=torch.float32).unsqueeze(dim = -1).unsqueeze(dim = -1)
            self.radialScaling = torch.tensor([0.], device=self.device, dtype=torch.float32)
            self.tangentialScaling = torch.tensor([0], device=self.device, dtype=torch.float32)
            self.anamorphosisAngle = torch.tensor([0.], device=self.device, dtype=torch.float32)
        else:
            self.ApplyMisreg(misreg)

        self.MakeZonalModes()

        self.initialized = True
        

    def forward(self, coefs):
        if self.training:
            self.MakeZonalModes()
        return self.GetDMShape(coefs)

    def GetDMShape(self, coefs):
        return torch.einsum('rc,cwh->rwh', coefs, self.IF)

    def ApplyMisreg(self, misreg):
        self.rotationAngle = torch.tensor([misreg['rotationAngle']], device=self.device, dtype=torch.float32)
        self.grid_shift = torch.tensor([[misreg['shiftX'] * self.Nres / self.D,misreg['shiftY'] * self.Nres / self.D]], device=self.device, dtype=torch.float32).unsqueeze(dim = -1).unsqueeze(dim = -1)
        self.radialScaling = torch.tensor([misreg['radialScaling'] / 100], device=self.device, dtype=torch.float32)
        self.tangentialScaling = torch.tensor([misreg['tangentialScaling'] / 100], device=self.device, dtype=torch.float32)
        self.anamorphosisAngle = torch.tensor([misreg['anamorphosisAngle']], device=self.device, dtype=torch.float32)

    def MakeActGrid(self):
        x = torch.arange(0, self.Nact, device = self.device) - self.Nact/2 + 0.5
        x,y = torch.meshgrid(x,x, indexing = 'xy')
        self.grid = (x**2 + y**2) < ((self.Nact/2 + self.offset_to_fit_number_of_actuators)**2)
        self.totalAct = self.grid.sum()
        self.actuator_positions = torch.stack((x,y))[:, self.grid].permute(1,0)
        self.actuator_positions = self.actuator_positions * self.Nres / (self.Nact - 1)

        x = torch.arange(0, self.Nres, device = self.device) - self.Nres/2 + 0.5
        x,y = torch.meshgrid(x,x, indexing = 'xy')
        self.positions = torch.stack((x,y)).repeat(self.totalAct,1,1,1)

        print(f"Total number of actuators: {self.totalAct}")

    
    def rotate_coordinates(self, actuator_positions):
        theta = self.rotationAngle * torch.pi / 180

        c = torch.cos(theta)
        s = torch.sin(theta)

        R = torch.stack([
            torch.stack([ c, -s]),
            torch.stack([ s,  c])
        ]).squeeze()

        return actuator_positions @ R.T

    def anamorphosis_coordinates(self, actuator_positions):
        theta = self.anamorphosisAngle * torch.pi / 180
        mRad = 1 + self.radialScaling
        mNorm = 1 + self.tangentialScaling

        c = torch.cos(theta)
        s = torch.sin(theta)

        M = torch.stack([
            torch.stack([mRad*c*c + mNorm*s*s,
                        (mNorm - mRad)*s*c]),
            torch.stack([(mNorm - mRad)*s*c,
                        mRad*s*s + mNorm*c*c])
        ]).squeeze()

        return actuator_positions @ M.T

    def _CollapseIfGlobal(self, raw_param, transform):
        # per_actuator_calibration=True: use each actuator's own raw value.
        # per_actuator_calibration=False: use a differentiable mean of the raw
        # (reparameterized) values, expanded back out to every actuator, so an
        # optimizer step keeps every actuator tied to the same value.
        if self.per_actuator_calibration:
            return transform(raw_param)
        return transform(raw_param.mean()).expand(self.totalAct)

    def MakeZonalModes(self):

        transformed_positons = self.anamorphosis_coordinates(self.actuator_positions)
        transformed_positons = self.rotate_coordinates(transformed_positons)

        actuator_grids = transformed_positons[...,None,None] * self.flip_matrix + self.grid_shift - self.positions

        X = actuator_grids[:,0]
        Y = actuator_grids[:,1]

        effective_sign = self._CollapseIfGlobal(self._sign, lambda p: p * 1e-6).view(-1, 1, 1)
        effective_moffat = self._CollapseIfGlobal(self._moffatParameter, torch.exp).view(-1, 1, 1)
        effective_mech = self._CollapseIfGlobal(self._mechCoupling, torch.sigmoid).view(-1, 1, 1)

        cx = (1+self.radialScaling)*(self.Nres / self.Nact)/torch.sqrt(2*torch.log(1./effective_mech))
        cy = (1+self.tangentialScaling)*(self.Nres / self.Nact)/torch.sqrt(2*torch.log(1./effective_mech))

        # Radial direction of the anamorphosis
        theta = self.anamorphosisAngle*torch.pi/180

        # Compute the 2D Gaussian coefficients
        a = torch.cos(theta)**2/(2*cx**2) + torch.sin(theta)**2/(2*cy**2)
        b = -torch.sin(2*theta)/(4*cx**2) + torch.sin(2*theta)/(4*cy**2)
        c = torch.sin(theta)**2/(2*cx**2) + torch.cos(theta)**2/(2*cy**2)

        r2 = (a*X**2 + 2*b*X*Y + c*Y**2)

        self.IF = effective_sign * 1 / (1 + r2/effective_moffat)**effective_moffat

        self.IF *= self.pupil
        self.IF[:, self.pupil] -= self.IF[:, self.pupil].mean(dim=(-1), keepdim=True)


    def GetMisreg(self):

        xy_values = self.grid_shift.detach().cpu().squeeze().tolist()

        misreg = {}

        misreg['rotationAngle'] = self.rotationAngle.detach().cpu().item()
        # shift X in m
        misreg['shiftX'] = xy_values[0] / self.Nres * self.D
        # shift Y in m
        misreg['shiftY'] = xy_values[1] / self.Nres * self.D
        # amamorphosis angle in degrees
        misreg['anamorphosisAngle'] = self.anamorphosisAngle.detach().cpu().item()
        # normal scaling in % of diameter
        misreg['tangentialScaling'] = self.tangentialScaling.detach().cpu().item() * 100
        # radial scaling in % of diameter
        misreg['radialScaling'] = self.radialScaling.detach().cpu().item() * 100

        DMDict = {}

        # Representative summary only -- these three are now per-actuator
        # vectors; full fidelity is preserved separately via state_dict().
        DMDict["moffatParam"] = self.moffatParameter.detach().mean().cpu().item()
        DMDict["signedAmplitude"] = self.sign.detach().mean().cpu().item()
        DMDict["MechCoupling"] = self.mechCoupling.detach().mean().cpu().item()
        DMDict["FlipLeftRight"] = self.flip_lr
        DMDict["FlipTopBottom"] = self.flip_tb
        DMDict["offset_to_fit_number_of_actuators"] = self.offset_to_fit_number_of_actuators

        return misreg, DMDict
    
    def RoughCalibration(self, wfs, bench_iMat, M2C):
        index = torch.arange(0, 5, device = self.device)
        target = bench_iMat[index]
        rotations = torch.arange(0,4, device = self.device) * 360/8
        old_sign = self.sign
        signs = [-old_sign, old_sign]
        flips_lr = [False, True]
        flips_tb = [False, True]

        best_loss = torch.inf
        loss = torch.nn.MSELoss()

        wfs.BuildReferenceIntensity()
        
        best_params = []
        with torch.no_grad():
            for rotation in rotations:
                for sign in signs:
                    for flip_lr in flips_lr:
                        for flip_tb in flips_tb:

                            self.flip_matrix = torch.tensor([[-1 if flip_lr else 1, -1 if flip_tb else 1]], device = self.device).unsqueeze(dim = -1).unsqueeze(dim = -1)
                            self.sign = sign
                            self.rotationAngle = rotation

                            self.MakeZonalModes()
                            modes = self.GetDMShape(M2C[:,index].T)

                            wfs.BuildInteractionMatrix(modes, single_pass = True)

                            digital_image = wfs.iMat.view(-1, wfs.Npix, wfs.Npix)
            
                            l = loss(target, digital_image)

                            if l < best_loss:
                                best_loss = l
                                best_params = [rotation, sign, flip_lr, flip_tb]
        
            print("Best configuration found to be: ")
            print(f"Rotation angle: {best_params[0]}")
            print(f"Sign: " + ("Possitive" if bool((best_params[1] > 0).all()) else "Negative"))
            print(f"Flip left-right: " + ("True" if best_params[2] else "False"))
            print(f"Flip top-bottom: " + ("True" if best_params[3] else "False"))
            print()
            print('Updating values in the DM')
            self.rotationAngle = best_params[0]
            self.sign = best_params[1]
            self.flip_lr = best_params[2]
            self.flip_tb = best_params[3]

    def MakeZernikeM2C(self, nModes=None):
        if nModes is None:
            nModes = self.nModes
        with torch.no_grad():
            z, _ = Zernike(self.pupil, nModes)
            z /= self.wavenumber
            inv_IF = torch.linalg.pinv(self.IF[:, self.pupil])

            M2C = z @ inv_IF
            return M2C.T


    def LoadCalibration(self, file_path):
        # map_location so a checkpoint saved on CUDA still loads on a CPU-only machine
        checkpoint = torch.load(file_path, map_location=self.device)

        model = checkpoint["model"]
        DMDict = checkpoint["config"]
        misreg = checkpoint["misreg"]

        # Geometry must be restored (which resizes _sign/_moffatParameter/
        # _mechCoupling to match) BEFORE load_state_dict, since their shape
        # now depends on totalAct.
        self.flip_lr = DMDict["FlipLeftRight"]
        self.flip_tb = DMDict["FlipTopBottom"]
        self.offset_to_fit_number_of_actuators = DMDict["offset_to_fit_number_of_actuators"]

        self.load_state_dict(model)

        with torch.no_grad():
            self.ApplyMisreg(misreg)
            self.MakeZonalModes()



    def SaveCalibration(self, file_path):

        ensure_parent(file_path)
        misreg, DMDict = self.GetMisreg()

        torch.save({
        "model": self.state_dict(),
        "config": DMDict,
        "misreg": misreg
            }, file_path)


    def train(self, mode=True):
        # Let PyTorch handle the normal train/eval behavior
        super().train(mode)

        if not mode:
            self.requires_grad_(False)
            with torch.no_grad():
                self.MakeActGrid()
                self.MakeZonalModes()
        else:
            self.requires_grad_(True)
            self.MakeActGrid()
            self.MakeZonalModes()
        return self

    #### These properties are set such that when optimizing these values they all share the same order of magnitude.
    # ---------- Rotation ----------
    @property
    def rotationAngle(self):
        return self._rotationAngle * 180.0
    @rotationAngle.setter
    def rotationAngle(self, value):
        with torch.no_grad():
            self._rotationAngle.copy_(torch.as_tensor(value, device=self.device) / 180.0)

    # ---------- Shift ----------
    @property
    def grid_shift(self):
        return self._grid_shift * 5.0      # train around [-1,1], physical ±5 px
    @grid_shift.setter
    def grid_shift(self, value):
        with torch.no_grad():
            self._grid_shift.copy_(torch.as_tensor(value, device=self.device) / 5.0)

    # ---------- Amplitude ----------
    # Always stored as a length-totalAct vector: a scalar assignment is just
    # the trivial case of broadcasting one value into every actuator's slot.
    @property
    def sign(self):
        return self._sign * 1e-6            # train around O(1), output in m
    @sign.setter
    def sign(self, value):
        value = torch.broadcast_to(torch.as_tensor(value, device=self.device), (self.totalAct,)).clone()
        with torch.no_grad():
            self._sign.copy_(value / 1e-6)

    # ---------- Radial Scaling ----------
    @property
    def radialScaling(self):
        return self._radialScaling / 10
    @radialScaling.setter
    def radialScaling(self, value):
        with torch.no_grad():
            self._radialScaling.copy_(torch.as_tensor(value, device=self.device) * 10)

    # ---------- Tangential Scaling ----------
    @property
    def tangentialScaling(self):
        return self._tangentialScaling / 10
    @tangentialScaling.setter
    def tangentialScaling(self, value):
        with torch.no_grad():
            self._tangentialScaling.copy_(torch.as_tensor(value, device=self.device) * 10)

    # ---------- Anamorphosis Angle ----------
    @property
    def anamorphosisAngle(self):
        return self._anamorphosisAngle * 180.0
    @anamorphosisAngle.setter
    def anamorphosisAngle(self, value):
        with torch.no_grad():
            self._anamorphosisAngle.copy_(torch.as_tensor(value, device=self.device) / 180.0)


    # ---------- Moffat Parameter ----------
    # Always stored as a length-totalAct vector -- see `sign` above.
    @property
    def moffatParameter(self):
        return torch.exp(self._moffatParameter)
    @moffatParameter.setter
    def moffatParameter(self, value):
        value = torch.broadcast_to(torch.as_tensor(value, device=self.device), (self.totalAct,)).clone()
        if torch.any(value <= 0):
            raise ValueError("moffatParameter must be strictly positive.")
        value = torch.log(value)
        with torch.no_grad():
            self._moffatParameter.copy_(value)

    # ---------- Mechanical Coupling ----------
    # Always stored as a length-totalAct vector -- see `sign` above.
    @property
    def mechCoupling(self):
        return torch.sigmoid(self._mechCoupling)
    @mechCoupling.setter
    def mechCoupling(self, value):
        value = torch.broadcast_to(torch.as_tensor(value, device=self.device), (self.totalAct,)).clone()
        if torch.any(value <= 0) or torch.any(value >= 1):
            raise ValueError("mechCoupling must be between 0 and 1.")
        value = torch.log(value / (1 - value))
        with torch.no_grad():
            self._mechCoupling.copy_(value)

    # ---------- Per-actuator calibration flag ----------
    # Gates only how moffatParameter/sign/mechCoupling are USED in
    # MakeZonalModes (independently vs. tied via a differentiable mean of the
    # raw values) -- NOT what is stored. The three setters above always keep
    # a full length-totalAct vector regardless of this flag, so a caller can
    # (and sometimes will) see a non-uniform vector even while this is False;
    # in that state the DM's physical response still reflects only the mean.
    @property
    def per_actuator_calibration(self):
        return self._per_actuator_calibration
    @per_actuator_calibration.setter
    def per_actuator_calibration(self, value):
        self._per_actuator_calibration = value
        if self.initialized:
            self.MakeZonalModes()

    @property
    def flip_lr(self):
        return self._flip_lr
    @flip_lr.setter
    def flip_lr(self, value):
        self._flip_lr = value
        if self.initialized:
            self.flip_matrix = torch.tensor([[-1 if self._flip_lr else 1, -1 if self._flip_tb else 1]], device = self.device).unsqueeze(dim = -1).unsqueeze(dim = -1)
            self.MakeZonalModes()

    @property
    def flip_tb(self):
        return self._flip_tb
    @flip_tb.setter
    def flip_tb(self, value):
        self._flip_tb = value
        if self.initialized:
            self.flip_matrix = torch.tensor([[-1 if self._flip_lr else 1, -1 if self._flip_tb else 1]], device = self.device).unsqueeze(dim = -1).unsqueeze(dim = -1)
            self.MakeZonalModes()

    @property
    def offset_to_fit_number_of_actuators(self):
        return self._offset_to_fit_number_of_actuators
    @offset_to_fit_number_of_actuators.setter
    def offset_to_fit_number_of_actuators(self,value):
        # Only safe to change before an optimizer has been built over this
        # DM's parameters (e.g. at construction time, or in a following
        # notebook cell while still fitting an uncalibrated DM's actuator
        # count) -- never on an already-calibrated DM. Changing totalAct
        # replaces _sign/_moffatParameter/_mechCoupling with new nn.Parameter
        # objects, silently orphaning any existing optimizer state for them.
        self._offset_to_fit_number_of_actuators = value
        if self.initialized:
            sign_mean = self._sign.detach().mean()
            moffat_mean = self._moffatParameter.detach().mean()
            mech_mean = self._mechCoupling.detach().mean()
            self.MakeActGrid()
            self._sign = nn.Parameter(sign_mean.expand(self.totalAct).clone())
            self._moffatParameter = nn.Parameter(moffat_mean.expand(self.totalAct).clone())
            self._mechCoupling = nn.Parameter(mech_mean.expand(self.totalAct).clone())
            self.MakeZonalModes()
        