from typing import List, Tuple

import numpy as np


def get_tile_slices(shape: Tuple[int, int], tile_size=(2 ** 13, 2 ** 13), col_first=True, pad=0) -> List[Tuple[slice, slice]]:
    r, c = shape
    tile_size_r, tile_size_c = tile_size
    num_col = int(np.ceil(c / tile_size_c))
    num_row = int(np.ceil(r / tile_size_r))
    slices = []
    if col_first:
        for col_i in range(num_col):
            for row_i in range(num_row):
                slices.append(get_tile_slice(col_i, row_i, shape, tile_size, pad))
    else:
        for row_i in range(num_row):
            for col_i in range(num_col):
                slices.append(get_tile_slice(col_i, row_i, shape, pad, tile_size))
    return slices


def get_tile_slice(col_i: int, row_i: int, shape: Tuple[int, int], pad: int, tile_size=(2 ** 13, 2 ** 13)) -> Tuple[slice, slice]:
    r, c = shape

    tile_size_r, tile_size_c = tile_size

    col_end = min(c, (col_i + 1) * tile_size_c + pad)
    col_start = max(0, col_i * tile_size_c - pad)
    cs = slice(col_start, col_end)

    row_end = min(r, (row_i + 1) * tile_size_r + pad)
    row_start = max(0, row_i * tile_size_r - pad)
    rs = slice(row_start, row_end)
    return rs, cs


def get_final_slices(roi_slice: Tuple[slice, slice],
                     tile_slices: List[Tuple[slice, slice]]) -> List[Tuple[slice, slice]]:
    return [get_final_slice(roi_slice, tile_slice) for tile_slice in tile_slices]


def get_final_slice(roi_slice: Tuple[slice, slice], tile_slice: Tuple[slice, slice]) -> \
        Tuple[slice, slice]:
    roi_rs, roi_cs = roi_slice
    rs, cs = tile_slice

    final_rs = slice(roi_rs.start + rs.start, roi_rs.start + rs.stop)
    final_cs = slice(roi_cs.start + cs.start, roi_cs.start + cs.stop)

    return final_rs, final_cs
