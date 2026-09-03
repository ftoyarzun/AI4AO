# AI4AO

Artificial Inteligence for Adaptive Optics (AI4AO) is a project under development to propose a python-based tool to perform end-to-end AO simulations and machine learning for phase reconstruction and control.
This code is inspired from OOPAO: https://github.com/cheritier/OOPAO developped by C.T. Heritier 
The project was initially intended for personal use. It is now open to any interested user. 

## FUNCTIONALITIES

	_ Phase Dataset: 		Extremely fast multi-layer phase screen generation, with and without scintillation.
	_ Wavefront sensor: 	Fully differentiable and parallelized end-to-end WFS simulations
	_ Mask Generation:      Test classic (Pyramid, Zernike) masks or create/optimize you own design
	_ Deformable mirror:    Fully differentiable deformable mirror model to compute misregistration


![WFS animation](Images/wfs_animation.gif)

## MODULES REQUIRED
The code is written for Python 3 and requires the following modules:
```
mmengine
torch
numpy
matplotlib
scipy
tqdm 
```

## INSTALLATION 

### (Recommended) Creating a virtual environment

It is always recommended that you use a virtual environment. First create it:

```
python -m venv venv
```

And finally activate it:

```
# Unix
source ./venv/bin/activate

# or

# Windows PowerShell
.\venv\Scripts\activate
```

After the environment is set up and activated, make sure to update pip to the latest version
```
python -m pip install --upgrade pip setuptools wheel typing-extensions
```

Then, follow the instructions from Pytorch's website to install the corresponding version for you. Using CUDA is highly recomended.
```
https://pytorch.org/get-started/locally/
```

Once the Pytorch installation finishes, this package can then be easily installed. Anytime you wish to use this
package, you should activate the respective environment.

### Using `pip`

First clone the repository:

```
https://github.com/ftoyarzun/AI4AO
```

And then install the package using `pip`:

```
python -m pip install -e AI4AO
