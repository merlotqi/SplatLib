/**
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
 */

#pragma once

#include <splat/splat_version.h>

#include <splat/maths/maths.h>
#include <splat/maths/rotate-sh.h>

#include <splat/models/data-table.h>
#include <splat/models/lcc.h>
#include <splat/models/morton-order.h>
#include <splat/models/ply.h>
#include <splat/models/sog.h>

#include <splat/readers/decompress_ply.h>
#include <splat/readers/ksplat_reader.h>
#include <splat/readers/lcc_reader.h>
#include <splat/readers/ply_reader.h>
#include <splat/readers/sog_reader.h>
#include <splat/readers/splat_reader.h>
#include <splat/readers/spz_reader.h>

#include <splat/spatial/btree.h>
#include <splat/spatial/kdtree.h>
#include <splat/spatial/kmeans.h>

#include <splat/utils/crc.h>
#include <splat/utils/logger.h>
#include <splat/utils/webp-codec.h>
#include <splat/utils/zip-reader.h>
#include <splat/utils/zip-writer.h>

#include <splat/writers/compressed_chunk.h>
#include <splat/writers/compressed_ply_writer.h>
#include <splat/writers/csv_writer.h>
#include <splat/writers/lod_writer.h>
#include <splat/writers/ply_writer.h>
#include <splat/writers/sog_writer.h>
