
import torch # type: ignore[import]
import torch.nn as nn # type: ignore[import]
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math



def imshow(
    tensor,
    max_batch_number=None,
    max_channel_number=None,
    same_scale=False,
    group_boxes=True,
    cmap="viridis",
    figsize=None,
    colorbar=False,
    **kwargs
):
    """
    Display tensors of shape [WH], [CWH], or [BCWH].

    Parameters
    ----------
    tensor : torch.Tensor or np.ndarray
        Input image tensor.

    show_selection : int, optional
        Number of images/groups to show.

    same_scale : bool
        Use common vmin/vmax.

    group_boxes : bool
        Draw rectangles around B groups for BCWH input.

    cmap : str
        Matplotlib colormap.

    kwargs :
        Passed to plt.imshow.
    """

    # Convert to numpy
    if torch.is_tensor(tensor):
        tensor = tensor.detach().cpu().numpy()

    # Convert dimensions
    if tensor.ndim == 2:
        tensor = tensor[None, None, ...]     # B,C,W,H
    elif tensor.ndim == 3:
        tensor = tensor[None, ...]            # B,C,W,H
    elif tensor.ndim != 4:
        raise ValueError(
            f"Tensor must be [WH], [CWH], or [BCWH], got {tensor.shape}"
        )

    B, C, W, H = tensor.shape

    # Selection
    if max_batch_number is not None:
        tensor = tensor[:min(B,max_batch_number)]
        
    if max_channel_number is not None:
        tensor = tensor[:,:min(C,max_channel_number)]

    B, C, W, H = tensor.shape

    # Scaling
    if same_scale:
        vmin = tensor.min()
        vmax = tensor.max()
    else:
        vmin = kwargs.pop("vmin", None)
        vmax = kwargs.pop("vmax", None)


    def divisor(n):
        divs = [1]
        for i in range(2,n+1):
            if n%i == 0:
                divs.append(i)
        return divs[len(divs)//2]
    # -------------------------
    # Single group [C,W,H]
    # -------------------------
    if B == 1:

        ncols = divisor(C)
        nrows = math.ceil(C / ncols)

        if figsize is None:
            figsize = (2*ncols, 2*nrows)

        fig = plt.figure(figsize=figsize)

        gs = fig.add_gridspec(
            nrows,
            ncols,
            wspace=0.05,
            hspace=0.05
        )

        axes = []

        for c in range(C):

            ax = fig.add_subplot(gs[c//ncols, c%ncols])

            ax.imshow(
                tensor[0,c],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
                **kwargs
            )

            ax.axis("off")
            axes.append(ax)

            if colorbar:
                fig.colorbar(ax.images[0], ax=ax)

        # Remove unused axes
        for i in range(C, nrows*ncols):
            ax = fig.add_subplot(gs[i//ncols, i%ncols])
            ax.axis("off")

        plt.tight_layout()

        return fig, axes


    # -------------------------
    # Multiple groups [BCWH]
    # -------------------------

    bcols = ncols = divisor(B)
    brows = math.ceil(B / bcols)

    ncols = ncols = divisor(C)
    nrows = math.ceil(C / ncols)


    if figsize is None:
        figsize = (
            bcols * ncols * 2,
            brows * nrows * 2
        )


    fig = plt.figure(figsize=figsize)


    outer = fig.add_gridspec(
        brows,
        bcols,
        wspace=0.25,
        hspace=0.25
    )

    axes = []

    for b in range(B):

        inner = outer[b//bcols, b%bcols].subgridspec(
            nrows,
            ncols,
            wspace=0.05,
            hspace=0.05
        )

        group_axes = []

        for c in range(C):

            ax = fig.add_subplot(
                inner[c//ncols, c%ncols]
            )

            ax.imshow(
                tensor[b,c],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
                **kwargs
            )

            ax.axis("off")

            group_axes.append(ax)
            axes.append(ax)

            if colorbar:
                fig.colorbar(ax.images[0], ax=ax)


        # Remove empty slots
        for i in range(C, nrows*ncols):
            ax = fig.add_subplot(
                inner[i//ncols, i%ncols]
            )
            ax.axis("off")


        # Draw group rectangle
        if group_boxes:

            fig.canvas.draw()

            positions = [
                ax.get_position()
                for ax in group_axes
            ]

            x0 = min(p.x0 for p in positions)
            y0 = min(p.y0 for p in positions)
            x1 = max(p.x1 for p in positions)
            y1 = max(p.y1 for p in positions)

            rect = Rectangle(
                (x0,y0),
                x1-x0,
                y1-y0,
                transform=fig.transFigure,
                fill=False,
                linewidth=2
            )

            fig.patches.append(rect)

    plt.tight_layout()

    return fig, axes


def get_activations(model, x):
    """
    Returns a dictionary containing the output of every Conv2d layer.
    """
    activations = {}
    hooks = []

    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach().cpu()
        return hook

    # Register a hook on every Conv2d layer
    conv_idx = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(
                module.register_forward_hook(hook_fn(f"conv{conv_idx}"))
            )
            conv_idx += 1

    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(x)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return activations

def get_conv_kernel_gradients(model, conv_idx):
    """
    Returns the gradient of the weights of the conv_idx-th Conv2d layer.

    Shape:
        [out_channels, in_channels/groups, kernel_h, kernel_w]
    """
    conv_layers = [m for m in model.modules() if (isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear))]

    if conv_idx >= len(conv_layers):
        raise ValueError(f"Model has only {len(conv_layers)} Conv2d layers.")

    layer = conv_layers[conv_idx]

    if layer.weight.grad is None:
        raise RuntimeError("No gradients available. Did you call loss.backward()?")

    return layer.weight.grad.detach().cpu()


def get_conv_kernels(model, conv_idx):
    """
    Returns the weight tensor of the conv_idx-th Conv2d layer.

    Shape:
        [out_channels, in_channels, kernel_h, kernel_w]
    """
    conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]

    if conv_idx >= len(conv_layers):
        raise ValueError(f"Model has only {len(conv_layers)} Conv2d layers.")

    return conv_layers[conv_idx].weight.detach().cpu()


def print_graph(fn, indent=0, visited=None):
    if visited is None:
        visited = set()

    if fn is None or fn in visited:
        return

    visited.add(fn)

    print(" " * indent + type(fn).__name__)

    for next_fn, _ in fn.next_functions:
        print_graph(next_fn, indent + 2, visited)

 # print_graph(total_loss.grad_fn)



if __name__ == "__main__":
    a = """
    from torch.profiler import record_function

    import torch
    from torch.profiler import (
        profile,
        schedule,
        ProfilerActivity,
        tensorboard_trace_handler,
        record_function,
    )

    for p in dm.parameters():
        p.requires_grad_(False)

    for p in wfs.parameters():
        p.requires_grad_(False)



    # ------------------------------
    # Profiler
    # ------------------------------
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        schedule=schedule(
            wait=100,      # Skip first 2 iterations
            warmup=100,    # Warm up profiler
            active=50,    # Profile next 5 iterations
            repeat=1,
        ),
        on_trace_ready=tensorboard_trace_handler("./profiler_logs"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:

        for i in range(500):

            with record_function("Dataset"):
                with record_function("Generate sample"):
                    with torch.no_grad():
                        batch = dataset[0]

                with record_function("Extract tensors"):
                    with torch.no_grad():
                        phaseGT = batch["phase"]
                        pupilGT = batch["pupil"]
                        gain = batch["loop_gain"]
                        leak = batch["loop_leak"]

                with record_function("Project phase"):
                    with torch.no_grad():
                        modes = phaseGT.flatten(start_dim=-2) @ z_inv
                        Ze = residual_phase.flatten(start_dim=-2) @ z_inv

                with record_function("Initialize tensors"):
                    with torch.no_grad():
                        z_estimated = torch.zeros_like(modes)
                        z_buffer = torch.zeros_like(modes)
                        z_output = torch.zeros_like(modes)
                        phase_reconstructed = torch.zeros_like(phaseGT)

                with record_function("Set WFS"):
                    with torch.no_grad():
                        wfs.SetPhotonsAndRON(photons, ron)


                with record_function("Initial stuff"):
                    with torch.no_grad():
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

                        residual_phase = phaseGT - phase_reconstructed
                        
                        # modes = torch.matmul(phaseGT.flatten(start_dim = -2), z_inv)
                        # phaseGT = dm(modes @ M2C.T)
                
                        residual_phase = phaseGT - phase_reconstructed 
                
                        modes = torch.matmul(phaseGT.flatten(start_dim = -2), z_inv)
                        Ze = torch.matmul(residual_phase.flatten(start_dim = -2), z_inv)
                
                
                        # Predict coefficients and update estimate
                        z_estimated = z_estimated * leak + gain * z_buffer  # Apply correction with gain
                        z_buffer = torch.clone(z_output)

            with record_function("WFS forward"):
                with torch.no_grad():
                    wfs_frames = wfs(residual_phase, pupilGT)
                    preprocessed_frames = framePreprocessor.ProcessFrame(wfs_frames)

            with record_function("CNN"):
                z_output = phaseReconstructor(preprocessed_frames)

            with record_function("DM"):
                phase_reconstructed = dm(z_estimated @ M2C.T)
                phase_reconstructed_iter = dm(z_output @ M2C.T)
                
                # phase_reconstructed_ideal = dm(modes @ M2C.T)
                phase_reconstructed_iter_ideal = dm(Ze @ M2C.T)

            with record_function("Loss"):
                total_loss += loss_variance((residual_phase - phase_reconstructed_iter)) / num_iterations

            with record_function("Backward"):
                total_loss.backward()

            with record_function("Optimizer"):
                optimizer_n.step()

            # Tell the profiler that one iteration finished
            prof.step()

    # ------------------------------
    # Print summary
    # ------------------------------
    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=50,
        )
    )
    """