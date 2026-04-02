/***********************************************************************************
 *
 * splat - A C++ library for reading and writing 3D Gaussian Splatting (splat) files.
 *
 * This library provides functionality to convert, manipulate, and process
 * 3D Gaussian splatting data formats used in real-time neural rendering.
 *
 * This file is part of splat.
 *
 * splat is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * splat is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 * For more information, visit the project's homepage or contact the author.
 *
 ***********************************************************************************/

#pragma once

/**
 * @file voxel.h
 * @brief Unified header for voxel module
 *
 * This module provides voxel-based operations for Gaussian splatting data,
 * including voxelization, spatial queries, surface extraction, and navigation simplification.
 */

#include <splat/voxel/collision-glb.h>
#include <splat/voxel/gaussian-aabb.h>
#include <splat/voxel/gaussian-bvh.h>
#include <splat/voxel/gpu-voxelization.h>
#include <splat/voxel/marching-cubes.h>
#include <splat/voxel/nav-simplify.h>
#include <splat/voxel/sparse-octree.h>
#include <splat/voxel/voxel-filter.h>

namespace splat {

/// Voxelization and spatial operations module

}  // namespace splat
