
import torch # type: ignore[import]
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

