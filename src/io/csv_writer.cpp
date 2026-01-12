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

#include <absl/strings/str_join.h>
#include <splat/io/csv_writer.h>
#include <splat/models/data-table.h>

#include <fstream>

namespace splat {

void writeCSV(const std::string& path, DataTable* dataTable) {
  assert(dataTable);
  const size_t len = dataTable->getNumRows();

  // write header
  std::ofstream file;
  file.open(path);
  file << absl::StrJoin(dataTable->getColumnNames(), ",") << "\n";

  for (size_t i = 0; i < len; ++i) {
    std::string row;
    for (size_t c = 0; c < dataTable->getNumColumns(); c++) {
      if (c) {
        row += ",";
      }
      row += dataTable->getColumn(c).getValue<std::string>(i);
    }
    file << row << "\n";
  }
}

}  // namespace splat
