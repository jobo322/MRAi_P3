"""Cross-section extraction (NumPy-only mirror of StoP3.split_2D).

This module duplicates ONLY the NumPy logic of src/MRAi_P3/StoP3.py::split_2D /
Para_split_2D so that training-data preparation does not require TensorFlow
(StoP3 imports TF at module level for model loading, which is unneeded here).

Reference: src/MRAi_P3/StoP3.py lines 10-26. Any change there must be mirrored here.
Output contract: X (H*W,128,1) L2-normalised per vector, Norm (H*W,), wrap padding.
"""
import numpy as np


def para_split_2d(S_pad, S_shape, S_size, Cross_Size, S_data):
    for ss in range(S_size):
        i = ss // S_shape[1]
        j = ss % S_shape[1]
        S_data[ss, :Cross_Size[0], 0] = S_pad[i:i + Cross_Size[0], j + int(Cross_Size[1] / 2)]
        S_data[ss, -Cross_Size[1]:, 0] = S_pad[i + int(Cross_Size[0] / 2), j:j + Cross_Size[1]]
    return S_data


def split_2d(S, Cross_Size):
    S_shape = S.shape
    S_size = S.size
    S_data = np.zeros((S_size, np.sum(Cross_Size), 1))
    pad_width = ((int(Cross_Size[0] / 2), int(Cross_Size[0] / 2 - 1 / 2)),
                 (int(Cross_Size[1] / 2), int(Cross_Size[1] / 2 - 1 / 2)))
    S = np.pad(S, pad_width, "wrap")
    S_data = para_split_2d(S, S_shape, S_size, Cross_Size, S_data)
    Norm = np.linalg.norm(S_data, axis=(1, 2))
    S_data = S_data / Norm[:, None, None]
    return S_data, Norm
