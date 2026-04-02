"""
Python wrapper for the splat library - 3D Gaussian Splatting I/O and operations.

This module provides a high-level Pythonic interface to the splat library.

Usage:
    import splat_transform as splat

    # Read a PLY file
    dt = splat.read_ply("scene.ply")
    print(f"Loaded {dt.num_rows} splats with {dt.num_columns} columns")
    
    # Access column data as numpy arrays
    x = dt["x"]  # or dt.get_column_data_by_name("x")
    
    # Transform it
    import numpy as np
    splat.transform(dt, np.array([0, 0, 0], dtype=np.float32),
                    np.array([1, 0, 0, 0], dtype=np.float32), 1.0)
    
    # Write it back
    splat.write_splat(dt, "output.splat")
"""

from splat_transform_cpp import (
    Column,
    DataTable,
    RotateSH,
    WriteGlbOptions,
    WriteVoxelOptions,
    combine,
    get_splat_version,
    read_ksplat,
    read_lcc,
    read_ply,
    read_sog,
    read_splat,
    read_spz,
    read_voxel,
    sigmoid,
    simplify_gaussians,
    sort_morton_order,
    transform,
    write_csv,
    write_glb,
    write_ply,
    write_sog,
    write_splat,
    write_voxel,
)

__version__ = get_splat_version()

__all__ = [
    "Column",
    "DataTable",
    "RotateSH",
    "WriteGlbOptions",
    "WriteVoxelOptions",
    "combine",
    "read_ksplat",
    "read_lcc",
    "read_ply",
    "read_sog",
    "read_splat",
    "read_spz",
    "read_voxel",
    "sigmoid",
    "simplify_gaussians",
    "sort_morton_order",
    "transform",
    "write_csv",
    "write_glb",
    "write_ply",
    "write_sog",
    "write_splat",
    "write_voxel",
]
