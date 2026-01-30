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

#include <string>

namespace splat {

class DataTable;

/**
 * @brief Writes 3D Gaussian splatting data to a .splat file format.
 *
 * This function converts internal Gaussian splatting data representation to the
 * binary .splat file format, which is optimized for efficient storage and loading.
 * The .splat format stores each Gaussian splat as a 32-byte record containing:
 * - Position (3 × float32): x, y, z coordinates
 * - Scale (3 × float32): scale factors along each axis (stored as linear values)
 * - Color and opacity (4 × uint8): RGB color and alpha opacity
 * - Rotation (4 × uint8): Quaternion rotation encoded as normalized bytes
 *
 * The conversion process includes:
 * - Converting log-scale values to linear scale using exp()
 * - Converting spherical harmonic coefficients to RGB color using SH_C0 constant
 * - Applying sigmoid function to opacity values
 * - Normalizing and encoding quaternion rotation to byte range [0, 255]
 *
 * @param datatable Pointer to the DataTable containing Gaussian splat data.
 *                 Must contain the following columns:
 *                 - Position: "x", "y", "z" (float)
 *                 - Scale: "scale_0", "scale_1", "scale_2" (float, log-scale)
 *                 - Color: "f_dc_0", "f_dc_1", "f_dc_2" (float, spherical harmonics)
 *                 - Opacity: "opacity" (float, log-space)
 *                 - Rotation: "rot_0", "rot_1", "rot_2", "rot_3" (float, quaternion)
 *
 * @param filepath Output file path where the .splat data will be written.
 *                 The file will be created or overwritten if it exists.
 *
 * @throws std::runtime_error If the file cannot be opened for writing.
 * @throws std::runtime_error If required columns are missing from the DataTable.
 *
 * @note The function writes data in binary format with little-endian byte order.
 *       Each splat is written as a 32-byte record, making the total file size
 *       predictable: num_splats × 32 bytes.
 *
 * @note For performance, the function flushes the output stream every 1000 splats
 *       to balance memory usage and I/O efficiency.
 *
 * @see readSplat() for reading .splat files
 * @see DataTable for data structure requirements
 */
void writeSplat(const DataTable* datatable, const std::string& filepath);

}  // namespace splat
