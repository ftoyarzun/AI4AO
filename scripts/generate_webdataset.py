# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 17:07 2026

@author: Matias Marambio-Jimenez
"""

"""Generate phase/frame pairs and write WebDataset shards.

Each sample is written with a key so files inside shards are named
`<key>.frame.npy` and `<key>.phase.npy`.

Usage example:
  python scripts/generate_webdataset.py --out shards/data-%03d.tar --n 1000 --shard-size 256
"""
import io
import argparse
import numpy as np
import torch
import webdataset as wds
import json

from PhaseDataset import PhaseDataset
import TorchPropagator as TorchPropagator
import params_exp as params


def npy_bytes_from_array(arr: np.ndarray):
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="shards/data-%03d.tar")
    parser.add_argument("--n", type=int, default=1000, help="number of samples")
    parser.add_argument("--shard-size", type=int, default=256, help="samples per shard")
    parser.add_argument("--device", type=str, default=None, help="torch device to use (default: auto-detect)")
    parser.add_argument("--params", type=str, default=None, help="optional params file (unused, kept for compatibility)")
    parser.add_argument("--include-metadata", action="store_true", help="include scalar metadata (r0, Nphotons, RON) as meta.json")
    parser.add_argument("--include-ze", action="store_true", help="include Zernike coefficients as ze.npy")
    args = parser.parse_args()

    # compute padding width based on total number of samples
    pad = len(str(args.n - 1)) if args.n > 1 else 1

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare parameter sets and ensure we generate one phase per call
    WFSParams = params.WFSParams.copy()
    AtmosParams = params.AtmosParams.copy()
    LoopParams = params.LoopParams.copy()

    AtmosParams["Nphases"] = 1

    dataset = PhaseDataset(WFSParams, AtmosParams, LoopParams, device)

    wfs = TorchPropagator.WFS(WFSParams, device)

    # Ensure output directory exists (ShardWriter will create files)
    writer = wds.ShardWriter(args.out, maxcount=args.shard_size)

    try:
        for i in range(args.n):
            # generate one sample
            phaseMap, Ze, Nphotons, RON, r0 = dataset.__getitem__(0)

            # phaseMap has shape (1, H, W) - extract first
            phase = phaseMap[0].to(device)

            # configure WFS photon/RON
            try:
                nphot = float(Nphotons.flatten()[0].cpu().item())
            except Exception:
                nphot = float(Nphotons.cpu().numpy().ravel()[0])

            try:
                ron = float(RON.flatten()[0].cpu().item())
            except Exception:
                ron = float(RON.cpu().numpy().ravel()[0])

            wfs.SetPhotonsAndRON(nphot, ron)

            # Produce frame (returns tensor with leading batch dim)
            with torch.no_grad():
                frame = wfs.Propagator(phase.unsqueeze(0))

            frame_np = frame[0].cpu().numpy()
            phase_np = phase.cpu().numpy()

            key = f"{i:0{pad}d}"

            sample = {
                "__key__": key,
                "frame.npy": npy_bytes_from_array(frame_np),
                "phase.npy": npy_bytes_from_array(phase_np),
            }

            if args.include_metadata:
                try:
                    r0_val = float(r0.flatten()[0].cpu().item())
                except Exception:
                    r0_val = float(r0.cpu().numpy().ravel()[0])

                meta = {"r0": r0_val, "Nphotons": nphot, "RON": ron}
                sample["meta.json"] = json.dumps(meta).encode("utf-8")

            if args.include_ze:
                try:
                    ze_np = Ze[0].cpu().numpy()
                except Exception:
                    ze_np = Ze.cpu().numpy().ravel()
                sample["ze.npy"] = npy_bytes_from_array(ze_np)

            writer.write(sample)

            if (i + 1) % 100 == 0:
                print(f"Written {i+1}/{args.n} samples")

    finally:
        writer.close()


if __name__ == "__main__":
    main()
