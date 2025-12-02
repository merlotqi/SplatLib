#pragma once

#include <filesystem>
#include <fstream>

#include "../utils/data_table.h"

namespace fs = std::filesystem;

namespace writer {
namespace csv {

std::string stringListJoin(const std::vector<std::string>& strings, const std::string& delimiter) {
  std::string result;
  for (size_t i = 0; i < strings.size(); ++i) {
    result += strings[i];
    if (i < strings.size() - 1) {
      result += delimiter;
    }
  }
  return result;
}

void writeCSV(const fs::path& path, const DataTable& data_table) {
  const size_t len = data_table.row_size();

  std::ofstream file;
  file.open(path);
  file << stringListJoin(data_table.get_column_names(), ",") << std::endl;

  for (size_t i = 0; i < len; ++i) {
    std::string row;
    for (size_t c = 0; c < data_table.columns.size(); c++) {
      if (c) {
        row += ",";
      }
      row += data_table.value_at<std::string>(i, c);
    }
    file << row << std::endl;
  }
}

}  // namespace csv
}  // namespace writer
