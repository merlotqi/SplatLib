#include <absl/strings/str_join.h>
#include <splat/io/csv_writer.h>
#include <splat/models/data-table.h>

#include <fstream>

namespace splat {

void writeCSV(const std::filesystem::path& path, DataTable* dataTable) {
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
    if ((i + 1) % 1000 == 0) {
      file.flush();
    }
  }
  file.flush();
  file.close();
}

}  // namespace splat
