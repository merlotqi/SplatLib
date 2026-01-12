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

namespace splat {

/**
 * @brief Reads and parses a Gaussian Splatting (.sog) file into a DataTable
 * 
 * This function loads and decodes a Gaussian Splatting scene stored in .sog format.
 * The .sog format is a compressed representation of Gaussian splatting data that
 * stores position, scale, color, rotation, and optionally higher-order spherical
 * harmonics in an efficient texture-based encoding.
 * 
 * The function supports two input modes:
 * 1. Reading from a single .sog ZIP archive file containing all components
 * 2. Reading from individual component files in a directory structure
 * 
 * @param file Path to the main .sog file or directory containing component files
 * @param sourceName Source name/path used to locate component files. If sourceName
 *                   ends with ".sog", the function treats it as a ZIP archive.
 *                   Otherwise, it treats sourceName as a directory containing
 *                   individual component files.
 * 
 * @return std::unique_ptr<DataTable> containing the decoded Gaussian splatting data.
 *         The DataTable contains the following columns (at minimum):
 *         - x, y, z: 3D positions (float)
 *         - scale_0, scale_1, scale_2: 3D scale factors (float)
 *         - f_dc_0, f_dc_1, f_dc_2: Spherical harmonic band 0 coefficients (RGB colors, float)
 *         - opacity: Opacity values (float)
 *         - rot_0, rot_1, rot_2, rot_3: Rotation quaternions (float)
 *         Additional columns for higher-order spherical harmonics (f_rest_*) are added
 *         if present in the source data.
 * 
 * @throws std::runtime_error if:
 *         - The .sog file or component files cannot be opened/read
 *         - Required metadata (meta.json) is missing or invalid
 *         - Texture dimensions are insufficient for the declared splat count
 *         - File format inconsistencies are detected
 * 
 * @note The function performs parallel processing using OpenMP for performance
 * @note Position coordinates are transformed using invLogTransform to restore
 *       the original coordinate space
 * @note Scale factors are decoded from a codebook-based quantization
 * @note Rotation quaternions are unpacked from a compressed 8-bit per channel representation
 * @note Colors and opacity are decoded from quantized SH0 coefficients with sigmoid inverse
 *       transformation for opacity
 * @note Higher-order spherical harmonics (bands > 0) are optionally decoded if present
 *       in the metadata
 * 
 * @see Meta::parseFromJson
 * @see decodeMeans
 * @see unpackQuat
 * @see invLogTransform
 * @see sigmoidInv
 */
std::unique_ptr<DataTable> readSog(const std::string& file, const std::string& sourceName);

}  // namespace splat
