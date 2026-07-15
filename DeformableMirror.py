import torch
import torch.nn as nn

class DeformableMirror(nn.Module):
    def __init__(self, WFSDict, DMDict, device, offset_to_fit_number_of_actuators = 0.2, misreg = None):
        super().__init__()

        self.device = device
        self.Nres =         WFSDict["Nres"]
        self.sampling =     WFSDict["sampling"]
        self.D =            WFSDict["D"]
        self.Nact =         WFSDict["Nactuator"]
        self.wavelength =   WFSDict["Wavelength"]
        self.wavenumber = 2 * torch.pi / self.wavelength
        
        self.totalAct = 241
        self.offset_to_fit_number_of_actuators = offset_to_fit_number_of_actuators
        
        self.flip_lr = DMDict["FlipLeftRight"]
        self.flip_tb = DMDict["FlipTopBottom"]
        self.flip_matrix = torch.tensor([[-1 if self.flip_tb else 1, -1 if self.flip_lr else 1]], device = self.device).unsqueeze(dim = -1).unsqueeze(dim = -1)
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
            self.rotationAngle = torch.tensor([misreg['rotationAngle']], device=self.device, dtype=torch.float32)
            self.grid_shift = torch.tensor([[misreg['shiftX'] * self.Nres / self.D,misreg['shiftY'] * self.Nres / self.D]], device=self.device, dtype=torch.float32).unsqueeze(dim = -1).unsqueeze(dim = -1)
            self.radialScaling = torch.tensor([misreg['radialScaling'] / 100], device=self.device, dtype=torch.float32)
            self.tangentialScaling = torch.tensor([misreg['tangentialScaling'] / 100], device=self.device, dtype=torch.float32)
            self.anamorphosisAngle = torch.tensor([misreg['anamorphosisAngle']], device=self.device, dtype=torch.float32)


        self.MakeActGrid()
        self.modesComputation()

    def MakeDMTrainable(self):
        self.moffatParameter = nn.Parameter(self.moffatParameter)
        self.sign = nn.Parameter(self.sign)
        self.mechCoupling = nn.Parameter(self.mechCoupling)
        
        self.rotationAngle = nn.Parameter(self.rotationAngle)
        self.grid_shift = nn.Parameter(self.grid_shift)
        self.radialScaling = nn.Parameter(self.radialScaling)
        self.tangentialScaling = nn.Parameter(self.tangentialScaling)
        self.anamorphosisAngle = nn.Parameter(self.anamorphosisAngle)


    def GetModalModes(self, M2C):
        return torch.einsum('cr,cwh->rwh', M2C, self.modes)

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

    def modesComputation(self):

        transformed_positons = self.anamorphosis_coordinates(self.actuator_positions)
        transformed_positons = self.rotate_coordinates(transformed_positons)

        actuator_grids = transformed_positons[...,None,None] * self.flip_matrix + self.grid_shift - self.positions

        X = actuator_grids[:,0]
        Y = actuator_grids[:,1]

        cx = (1+self.radialScaling)*(self.Nres / self.Nact)/torch.sqrt(2*torch.log(1./self.mechCoupling))
        cy = (1+self.tangentialScaling)*(self.Nres / self.Nact)/torch.sqrt(2*torch.log(1./self.mechCoupling))

        # Radial direction of the anamorphosis
        theta = self.anamorphosisAngle*torch.pi/180

        # Compute the 2D Gaussian coefficients
        a = torch.cos(theta)**2/(2*cx**2) + torch.sin(theta)**2/(2*cy**2)
        b = -torch.sin(2*theta)/(4*cx**2) + torch.sin(2*theta)/(4*cy**2)
        c = torch.sin(theta)**2/(2*cx**2) + torch.cos(theta)**2/(2*cy**2)

        r2 = (a*X**2 + 2*b*X*Y + c*Y**2)

        self.modes = self.sign * 1e-6 * 1 / (1 + r2/self.moffatParameter)**self.moffatParameter * self.wavenumber


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

        DMDict["moffatParam"] = self.moffatParameter.detach().cpu().item()
        DMDict["signedAmplitude"] = self.sign.detach().cpu().item()
        DMDict["MechCoupling"] = self.mechCoupling.detach().cpu().item()
        DMDict["FlipLeftRight"] = self.flip_lr
        DMDict["FlipTopBottom"] = self.flip_tb

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

        best_params = []
        with torch.no_grad():
            for rotation in rotations:
                for sign in signs:
                    for flip_lr in flips_lr:
                        for flip_tb in flips_tb:

                            self.flip_matrix = torch.tensor([[-1 if flip_tb else 1, -1 if flip_lr else 1]], device = self.device).unsqueeze(dim = -1).unsqueeze(dim = -1)
                            self.sign = sign
                            self.rotationAngle = rotation

                            self.modesComputation()
                            modes = self.GetModalModes(M2C[:,index])

                            wfs.BuildInteractionMatrix(modes, single_pass = True)

                            digital_image = wfs.iMat.view(-1, wfs.Npix, wfs.Npix)
            
                            l = loss(target, digital_image)

                            if l < best_loss:
                                best_loss = l
                                best_params = [rotation, sign, flip_lr, flip_tb]
        
            print("Best configuration found to be: ")
            print(f"Rotation angle: {best_params[0]}")
            print(f"Sign: " + ("Possitive" if best_params[1] > 0 else "Negative"))
            print(f"Flip left-right: " + ("True" if best_params[2] else "False"))
            print(f"Flip top-bottom: " + ("True" if best_params[3] else "False"))
            print()
            print('Updating values in the DM')
            self.rotationAngle = best_params[0]
            self.sign = best_params[1]
            self.flip_lr = best_params[2]
            self.flip_tb = best_params[3]
            self.flip_matrix = torch.tensor([[-1 if self.flip_tb else 1, -1 if self.flip_lr else 1]], device = self.device).unsqueeze(dim = -1).unsqueeze(dim = -1)