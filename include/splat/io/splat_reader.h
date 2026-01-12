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

#include <splat/models/data-table.h>

#include <string>

namespace splat {

/**
 * @brief Reads and parses a .splat file containing Gaussian splatting data
 *
 * This function loads a binary .splat file containing uncompressed Gaussian splatting data.
 * Each splat is represented as a fixed 32-byte structure containing position, scale, color,
 * opacity, and rotation data. The file format is a simple concatenation of these 32-byte records.
 *
 * The .splat format stores data in the following layout per splat (all little-endian):
 * - Bytes 0-11:   Position (x, y, z) as 3 × float32
 * - Bytes 12-23:  Scale (sx, sy, sz) as 3 × float32
 * - Bytes 24-27:  Color and opacity as 4 × uint8 (RGBA format)
 * - Bytes 28-31:  Rotation quaternion as 4 × uint8
 *
 * The function performs necessary conversions during loading:
 * - Position coordinates are used directly as float32 values
 * - Scale factors are converted from linear to logarithmic space (logf)
 * - Color values are converted from uint8 [0,255] range to spherical harmonic coefficients
 * - Opacity values are converted from uint8 to float with inverse sigmoid transformation
 * - Rotation quaternions are unpacked from uint8 representation and normalized
 *
 * @param filename Path to the .splat file to read
 *
 * @return std::unique_ptr<DataTable> containing the parsed Gaussian splatting data.
 *         The DataTable contains exactly 14 columns in the following order:
 *         1.  x (float): X-coordinate position
 *         2.  y (float): Y-coordinate position
 *         3.  z (float): Z-coordinate position
 *         4.  scale_0 (float): Logarithm of first scale component (log scale)
 *         5.  scale_1 (float): Logarithm of second scale component (log scale)
 *         6.  scale_2 (float): Logarithm of third scale component (log scale)
 *         7.  f_dc_0 (float): Spherical harmonic DC term for red channel
 *         8.  f_dc_1 (float): Spherical harmonic DC term for green channel
 *         9.  f_dc_2 (float): Spherical harmonic DC term for blue channel
 *         10. opacity (float): Opacity after inverse sigmoid transformation
 *         11. rot_0 (float): First component of normalized rotation quaternion
 *         12. rot_1 (float): Second component of normalized rotation quaternion
 *         13. rot_2 (float): Third component of normalized rotation quaternion
 *         14. rot_3 (float): Fourth component of normalized rotation quaternion
 *
 * @throws std::runtime_error if:
 *         - The file cannot be opened or does not exist
 *         - File size is not a multiple of 32 bytes (invalid format)
 *         - File is empty (0 bytes)
 *         - Failed to read expected amount of data during chunked reading
 *
 * @note The function reads data in chunks of 1024 splats (32KB) for memory efficiency
 * @note Scale factors are stored as linear values in the file but converted to log scale
 *       internally for numerical stability in Gaussian computations
 * @note Color conversion uses the spherical harmonic C0 constant (0.28209479177387814)
 *       to transform uint8 values to proper SH coefficients
 * @note Invalid rotation quaternions (zero length) are replaced with identity quaternion [0,0,0,1]
 * @note This format does not support higher-order spherical harmonics (SH bands > 0)
 *
 * @see BYTES_PER_SPLAT for the fixed record size
 * @see readFloatLE for little-endian float reading
 * @see readUInt8 for byte reading
 */
std::unique_ptr<DataTable> readSplat(const std::string& filename);

}  // namespace splat
