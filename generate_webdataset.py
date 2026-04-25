# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 17:07 2026

@author: Matias Marambio-Jimenez
"""

"""Generate phase/frame pairs and write WebDataset shards.

Each sample is written with a key so files inside shards are named
`<key>.frame.npy` and `<key>.phase.npy`.

Usage example:
  python generate_webdataset.py --out shards/data-%03d.tar --n 1000 --shard-size 256
"""
import io
import argparse
import numpy as np
import torch
import webdataset as wds
import json
import os
from tqdm import tqdm
from PhaseDataset import PhaseDataset
import TorchPropagator as TorchPropagator
import wfs_params as params
from ShackHartmann import ShackHartmann


def npy_bytes_from_array(arr: np.ndarray):
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="shards/data-%04d.tar")
    parser.add_argument("--n", type=int, default=10000, help="number of samples")
    parser.add_argument("--shard-size", type=int, default=500, help="samples per shard")
    parser.add_argument("--device", type=str, default=None, help="torch device to use (default: auto-detect)")
    parser.add_argument("--params", type=str, default=None, help="optional params file (unused, kept for compatibility)")
    parser.add_argument("--include-metadata", action="store_true", help="include scalar metadata (r0, Nphotons, RON) as meta.json")
    parser.add_argument("--include-ze", action="store_true", help="include Zernike coefficients as ze.npy")
    args = parser.parse_args()

    # compute padding width based on total number of samples
    pad = len(str(args.n - 1)) if args.n > 1 else 1
    pad_shard = len(str((args.n - 1) // args.shard_size)) if args.n > 1 else 1

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare parameter sets and ensure we generate one phase per call
    WFSParams = params.WFSParams.copy()
    AtmosParams = params.AtmosParams.copy()
    LoopParams = params.LoopParams.copy()

    dataset = PhaseDataset(WFSParams, AtmosParams, LoopParams, device)

    wfs = ShackHartmann(WFSParams, device)

    # Ensure output directory exists (ShardWriter will create files)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    writer = wds.ShardWriter(args.out, maxcount=args.shard_size)

    try:
        with tqdm(total=args.n, desc="Generating WebDataset", unit="sample") as pbar:
            for i in range(args.n):
                # generate one sample
                phase, _, nPhotons, RON, _ = dataset[i]

                # phaseMap has shape (1, H, W) - extract first)
                wfs.Nphotons = nPhotons
                wfs.RON = RON

                # Produce frame (returns tensor with leading batch dim)
                with torch.no_grad():
                    frame = wfs.Propagator(phase).squeeze()

                frame_np = frame.squeeze().cpu().numpy()
                phase_np = phase.cpu().numpy()

                key = f"{i:0{pad}d}"
                shard_idx = i // args.shard_size

                sample = {
                    "__key__": key,
                    "frame.npy": npy_bytes_from_array(frame_np),
                    "phase.npy": npy_bytes_from_array(phase_np),
                }

                if args.include_metadata:
                    pass

                if args.include_ze:
                    pass

                writer.write(sample)
                pbar.set_postfix(shard=f"{shard_idx:0{pad_shard}d}")
                pbar.update(1)


    finally:
        writer.close()


if __name__ == "__main__":
    main()
