#include "splat_c.h"

#include <splat/io/csv_writer.h>
#include <splat/io/glb_writer.h>
#include <splat/io/ksplat_reader.h>
#include <splat/io/ply_reader.h>
#include <splat/io/ply_writer.h>
#include <splat/io/sog_reader.h>
#include <splat/io/sog_writer.h>
#include <splat/io/splat_reader.h>
#include <splat/io/splat_writer.h>
#include <splat/io/spz_reader.h>
#include <splat/models/data-table.h>
#include <splat/models/ply.h>
#include <splat/op/combine.h>
#include <splat/op/decimate.h>
#include <splat/op/filter_visibility.h>
#include <splat/op/morton_order.h>
#include <splat/op/transform.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct splat_c_table {
  std::unique_ptr<splat::DataTable> table;
};

struct splat_c_table_array {
  std::vector<std::unique_ptr<splat_c_table>> tables;
};

namespace {

thread_local std::string g_last_error;

splat_c_status setError(splat_c_status status, std::string message) {
  g_last_error = std::move(message);
  return status;
}

void clearError() { g_last_error.clear(); }

template <typename T>
void requireOutput(T* output, const char* name) {
  if (output == nullptr) {
    throw std::invalid_argument(std::string(name) + " must not be null");
  }
}

std::string toLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return value;
}

bool endsWith(const std::string& value, const std::string& suffix) {
  return value.size() >= suffix.size() && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

const splat_c_table* getTableAt(const splat_c_table_array* tables, size_t index) {
  if (tables == nullptr) {
    throw std::invalid_argument("tables must not be null");
  }
  if (index >= tables->tables.size()) {
    throw std::out_of_range("table index out of range");
  }
  return tables->tables[index].get();
}

const splat::DataTable& requireTable(const splat_c_table* table) {
  if (table == nullptr) {
    throw std::invalid_argument("table must not be null");
  }
  if (table->table == nullptr) {
    throw std::runtime_error("table handle is empty");
  }
  return *table->table;
}

splat::DataTable& requireMutableTable(splat_c_table* table) {
  if (table == nullptr) {
    throw std::invalid_argument("table must not be null");
  }
  if (table->table == nullptr) {
    throw std::runtime_error("table handle is empty");
  }
  return *table->table;
}

const splat::Column& requireColumn(const splat_c_table* table, size_t column_index) {
  return requireTable(table).getColumn(column_index);
}

splat::Column& requireMutableColumn(splat_c_table* table, size_t column_index) {
  return requireMutableTable(table).getColumn(column_index);
}

const char* requireName(const char* name, const char* param_name) {
  if (name == nullptr || name[0] == '\0') {
    throw std::invalid_argument(std::string(param_name) + " must not be null or empty");
  }
  return name;
}

std::unique_ptr<splat_c_table> makeOwnedTable(std::unique_ptr<splat::DataTable> table) {
  if (table == nullptr) {
    throw std::runtime_error("table must not be null");
  }
  auto handle = std::make_unique<splat_c_table>();
  handle->table = std::move(table);
  return handle;
}

void appendTable(splat_c_table_array& tables, std::unique_ptr<splat::DataTable> table) {
  tables.tables.push_back(makeOwnedTable(std::move(table)));
}

void requireFinite(float value, const char* name) {
  if (!std::isfinite(value)) {
    throw std::invalid_argument(std::string(name) + " must be finite");
  }
}

void requireFinite(double value, const char* name) {
  if (!std::isfinite(value)) {
    throw std::invalid_argument(std::string(name) + " must be finite");
  }
}

size_t requireExactRowCount(const splat::DataTable& table, size_t count, const char* count_name) {
  const size_t rows = table.getNumRows();
  if (count != rows) {
    throw std::invalid_argument(std::string(count_name) + " must equal the table row count");
  }
  return rows;
}

template <typename T>
std::vector<T> makeVectorFromBytes(size_t length, const void* data) {
  std::vector<T> values(length);
  if (data != nullptr && length != 0) {
    std::memcpy(values.data(), data, length * sizeof(T));
  }
  return values;
}

splat::TypedArray makeTypedArray(splat_c_column_type type, size_t length, const void* data) {
  switch (type) {
    case SPLAT_C_COLUMN_TYPE_INT8:
      return makeVectorFromBytes<int8_t>(length, data);
    case SPLAT_C_COLUMN_TYPE_UINT8:
      return makeVectorFromBytes<uint8_t>(length, data);
    case SPLAT_C_COLUMN_TYPE_INT16:
      return makeVectorFromBytes<int16_t>(length, data);
    case SPLAT_C_COLUMN_TYPE_UINT16:
      return makeVectorFromBytes<uint16_t>(length, data);
    case SPLAT_C_COLUMN_TYPE_INT32:
      return makeVectorFromBytes<int32_t>(length, data);
    case SPLAT_C_COLUMN_TYPE_UINT32:
      return makeVectorFromBytes<uint32_t>(length, data);
    case SPLAT_C_COLUMN_TYPE_FLOAT32:
      return makeVectorFromBytes<float>(length, data);
    case SPLAT_C_COLUMN_TYPE_FLOAT64:
      return makeVectorFromBytes<double>(length, data);
  }

  throw std::invalid_argument("unknown column type");
}

void copyIntoColumn(splat::Column& column, const void* data, size_t size_bytes) {
  if (size_bytes != column.totalByteSize()) {
    throw std::invalid_argument("size_bytes must equal the target column byte size");
  }
  if (size_bytes != 0 && data == nullptr) {
    throw std::invalid_argument("data must not be null when size_bytes is non-zero");
  }
  if (size_bytes != 0) {
    std::memcpy(column.rawPointer(), data, size_bytes);
  }
}

template <typename Fn>
splat_c_status guard(Fn&& fn) {
  clearError();
  try {
    fn();
    return SPLAT_C_STATUS_OK;
  } catch (const std::invalid_argument& ex) {
    return setError(SPLAT_C_STATUS_INVALID_ARGUMENT, ex.what());
  } catch (const std::out_of_range& ex) {
    return setError(SPLAT_C_STATUS_OUT_OF_RANGE, ex.what());
  } catch (const std::exception& ex) {
    return setError(SPLAT_C_STATUS_RUNTIME_ERROR, ex.what());
  } catch (...) {
    return setError(SPLAT_C_STATUS_RUNTIME_ERROR, "unknown error");
  }
}

}  // namespace

const char* splat_c_version(void) { return SPLAT_C_VERSION; }

const char* splat_c_last_error_message(void) { return g_last_error.c_str(); }

splat_c_status splat_c_write(const splat_c_table* table, const char* filename, const splat_c_write_options* options) {
  return guard([&] {
    const auto& data_table = requireTable(table);
    const std::filesystem::path path(requireName(filename, "filename"));
    const std::string lower_path = toLower(path.string());
    const std::string lower_name = toLower(path.filename().string());

    const int sog_iterations = (options == nullptr || options->sog_iterations <= 0) ? 10 : options->sog_iterations;
    const int sog_bundle = (options == nullptr) ? -1 : options->sog_bundle;

    if (endsWith(lower_path, ".csv")) {
      auto tmp = data_table.clone();
      splat::writeCSV(path, tmp.get());
      return;
    }

    if (endsWith(lower_path, ".splat")) {
      splat::writeSplat(&data_table, path);
      return;
    }

    if (endsWith(lower_path, ".sog") || lower_name == "meta.json") {
      const bool bundle = (sog_bundle < 0) ? endsWith(lower_path, ".sog") : (sog_bundle != 0);
      splat::writeSog(path, &data_table, bundle, sog_iterations);
      return;
    }

    if (endsWith(lower_path, ".ply")) {
      splat::PlyData ply;
      ply.elements.push_back({"vertex", data_table.clone()});
      splat::writePly(path, ply);
      return;
    }

    if (endsWith(lower_path, ".glb")) {
      splat::writeGlb(path, &data_table);
      return;
    }

    throw std::invalid_argument("unsupported output file type");
  });
}

splat_c_status splat_c_read(const char* filename, splat_c_table_array** out_tables) {
  clearError();
  if (filename == nullptr || filename[0] == '\0') {
    return setError(SPLAT_C_STATUS_INVALID_ARGUMENT, "filename must not be null or empty");
  }
  if (out_tables == nullptr) {
    return setError(SPLAT_C_STATUS_INVALID_ARGUMENT, "out_tables must not be null");
  }
  *out_tables = nullptr;

  try {
    auto tables = std::make_unique<splat_c_table_array>();
    const std::filesystem::path path(filename);
    const std::string lower_path = toLower(path.string());
    const std::string lower_name = toLower(path.filename().string());

    if (endsWith(lower_path, ".ksplat")) {
      appendTable(*tables, splat::readKsplat(path));
    } else if (endsWith(lower_path, ".splat")) {
      appendTable(*tables, splat::readSplat(path));
    } else if (endsWith(lower_path, ".sog") || lower_name == "meta.json") {
      appendTable(*tables, splat::readSog(path, path));
    } else if (endsWith(lower_path, ".ply")) {
      appendTable(*tables, splat::readPly(path));
    } else if (endsWith(lower_path, ".spz")) {
      appendTable(*tables, splat::readSpz(path));
    } else if (endsWith(lower_path, ".lcc")) {
      return setError(SPLAT_C_STATUS_UNSUPPORTED_FORMAT, "reading .lcc files is not implemented yet");
    } else {
      return setError(SPLAT_C_STATUS_UNSUPPORTED_FORMAT, "unsupported input file type");
    }

    *out_tables = tables.release();
    return SPLAT_C_STATUS_OK;
  } catch (const std::invalid_argument& ex) {
    return setError(SPLAT_C_STATUS_INVALID_ARGUMENT, ex.what());
  } catch (const std::out_of_range& ex) {
    return setError(SPLAT_C_STATUS_OUT_OF_RANGE, ex.what());
  } catch (const std::exception& ex) {
    return setError(SPLAT_C_STATUS_RUNTIME_ERROR, ex.what());
  } catch (...) {
    return setError(SPLAT_C_STATUS_RUNTIME_ERROR, "unknown error");
  }
}

void splat_c_table_array_destroy(splat_c_table_array* tables) { delete tables; }

splat_c_status splat_c_table_array_size(const splat_c_table_array* tables, size_t* out_size) {
  return guard([&] {
    requireOutput(out_size, "out_size");
    if (tables == nullptr) {
      throw std::invalid_argument("tables must not be null");
    }
    *out_size = tables->tables.size();
  });
}

splat_c_status splat_c_table_array_get(const splat_c_table_array* tables, size_t index,
                                       const splat_c_table** out_table) {
  return guard([&] {
    requireOutput(out_table, "out_table");
    *out_table = getTableAt(tables, index);
  });
}

void splat_c_table_destroy(splat_c_table* table) { delete table; }

splat_c_status splat_c_table_create(splat_c_table** out_table) {
  return guard([&] {
    requireOutput(out_table, "out_table");
    *out_table = makeOwnedTable(std::make_unique<splat::DataTable>()).release();
  });
}

splat_c_status splat_c_table_clone(const splat_c_table* table, splat_c_table** out_table) {
  return guard([&] {
    requireOutput(out_table, "out_table");
    *out_table = nullptr;
    *out_table = makeOwnedTable(requireTable(table).clone()).release();
  });
}

splat_c_status splat_c_table_clone_columns(const splat_c_table* table, const char* const* column_names,
                                           size_t column_count, splat_c_table** out_table) {
  return guard([&] {
    requireOutput(out_table, "out_table");
    *out_table = nullptr;
    if (column_count != 0 && column_names == nullptr) {
      throw std::invalid_argument("column_names must not be null when column_count is non-zero");
    }

    std::vector<std::string> names;
    names.reserve(column_count);
    for (size_t i = 0; i < column_count; ++i) {
      names.emplace_back(requireName(column_names[i], "column_names entry"));
    }

    *out_table = makeOwnedTable(requireTable(table).clone(names)).release();
  });
}

splat_c_status splat_c_table_num_rows(const splat_c_table* table, size_t* out_rows) {
  return guard([&] {
    requireOutput(out_rows, "out_rows");
    *out_rows = requireTable(table).getNumRows();
  });
}

splat_c_status splat_c_table_num_columns(const splat_c_table* table, size_t* out_columns) {
  return guard([&] {
    requireOutput(out_columns, "out_columns");
    *out_columns = requireTable(table).getNumColumns();
  });
}

splat_c_status splat_c_table_column_index(const splat_c_table* table, const char* name, int32_t* out_index) {
  return guard([&] {
    requireOutput(out_index, "out_index");
    *out_index = requireTable(table).getColumnIndex(requireName(name, "name"));
  });
}

splat_c_status splat_c_table_has_column(const splat_c_table* table, const char* name, int32_t* out_has_column) {
  return guard([&] {
    requireOutput(out_has_column, "out_has_column");
    *out_has_column = requireTable(table).hasColumn(requireName(name, "name")) ? 1 : 0;
  });
}

splat_c_status splat_c_table_column_name(const splat_c_table* table, size_t column_index, const char** out_name) {
  return guard([&] {
    requireOutput(out_name, "out_name");
    *out_name = requireColumn(table, column_index).name.c_str();
  });
}

splat_c_status splat_c_table_column_type(const splat_c_table* table, size_t column_index,
                                         splat_c_column_type* out_type) {
  return guard([&] {
    requireOutput(out_type, "out_type");
    *out_type = static_cast<splat_c_column_type>(requireColumn(table, column_index).getType());
  });
}

splat_c_status splat_c_table_column_length(const splat_c_table* table, size_t column_index, size_t* out_length) {
  return guard([&] {
    requireOutput(out_length, "out_length");
    *out_length = requireColumn(table, column_index).length();
  });
}

splat_c_status splat_c_table_column_element_size(const splat_c_table* table, size_t column_index,
                                                 size_t* out_element_size) {
  return guard([&] {
    requireOutput(out_element_size, "out_element_size");
    *out_element_size = requireColumn(table, column_index).bytePreElement();
  });
}

splat_c_status splat_c_table_column_data(const splat_c_table* table, size_t column_index, const void** out_data,
                                         size_t* out_size_bytes) {
  return guard([&] {
    requireOutput(out_data, "out_data");
    const auto& column = requireColumn(table, column_index);
    *out_data = column.rawPointer();
    if (out_size_bytes != nullptr) {
      *out_size_bytes = column.totalByteSize();
    }
  });
}

splat_c_status splat_c_table_add_column(splat_c_table* table, const char* name, splat_c_column_type type, size_t length,
                                        const void* data) {
  return guard([&] {
    auto& data_table = requireMutableTable(table);
    const char* column_name = requireName(name, "name");
    if (data_table.hasColumn(column_name)) {
      throw std::invalid_argument("column name already exists");
    }
    data_table.addColumn(splat::Column{column_name, makeTypedArray(type, length, data)});
  });
}

splat_c_status splat_c_table_remove_column(splat_c_table* table, const char* name, int32_t* out_removed) {
  return guard([&] {
    requireOutput(out_removed, "out_removed");
    *out_removed = requireMutableTable(table).removeColumn(requireName(name, "name")) ? 1 : 0;
  });
}

splat_c_status splat_c_table_column_copy_data(splat_c_table* table, size_t column_index, const void* data,
                                              size_t size_bytes) {
  return guard([&] { copyIntoColumn(requireMutableColumn(table, column_index), data, size_bytes); });
}

splat_c_status splat_c_table_column_value_as_double(const splat_c_table* table, size_t column_index, size_t row_index,
                                                    double* out_value) {
  return guard([&] {
    requireOutput(out_value, "out_value");
    *out_value = requireColumn(table, column_index).getValue<double>(row_index);
  });
}

splat_c_status splat_c_table_column_set_value_from_double(splat_c_table* table, size_t column_index, size_t row_index,
                                                          double value) {
  return guard([&] {
    auto& column = requireMutableColumn(table, column_index);
    if (!std::isfinite(value) && column.getType() != splat::ColumnType::FLOAT32 &&
        column.getType() != splat::ColumnType::FLOAT64) {
      throw std::invalid_argument("value must be finite for integer columns");
    }
    column.setValue<double>(row_index, value);
  });
}

splat_c_status splat_c_table_transform(const splat_c_table* table, float translation_x, float translation_y,
                                       float translation_z, float rotation_w, float rotation_x, float rotation_y,
                                       float rotation_z, float uniform_scale, splat_c_table** out_table) {
  return guard([&] {
    requireOutput(out_table, "out_table");
    *out_table = nullptr;

    requireFinite(translation_x, "translation_x");
    requireFinite(translation_y, "translation_y");
    requireFinite(translation_z, "translation_z");
    requireFinite(rotation_w, "rotation_w");
    requireFinite(rotation_x, "rotation_x");
    requireFinite(rotation_y, "rotation_y");
    requireFinite(rotation_z, "rotation_z");
    requireFinite(uniform_scale, "uniform_scale");
    if (uniform_scale <= 0.0f) {
      throw std::invalid_argument("uniform_scale must be greater than zero");
    }

    Eigen::Quaternionf rotation(rotation_w, rotation_x, rotation_y, rotation_z);
    if (rotation.norm() == 0.0f) {
      throw std::invalid_argument("rotation quaternion must not be zero");
    }
    rotation.normalize();

    auto transformed = requireTable(table).clone();
    splat::transform(transformed.get(), Eigen::Vector3f(translation_x, translation_y, translation_z), rotation,
                     uniform_scale);

    *out_table = makeOwnedTable(std::move(transformed)).release();
  });
}

splat_c_status splat_c_tables_combine(const splat_c_table* const* tables, size_t table_count,
                                      splat_c_table** out_table) {
  return guard([&] {
    requireOutput(out_table, "out_table");
    *out_table = nullptr;
    if (tables == nullptr) {
      throw std::invalid_argument("tables must not be null");
    }
    if (table_count == 0) {
      throw std::invalid_argument("table_count must be greater than zero");
    }

    std::vector<std::unique_ptr<splat::DataTable>> clones;
    clones.reserve(table_count);
    for (size_t i = 0; i < table_count; ++i) {
      if (tables[i] == nullptr) {
        throw std::invalid_argument("tables contains a null entry");
      }
      clones.push_back(requireTable(tables[i]).clone());
    }

    *out_table = makeOwnedTable(splat::combine(clones)).release();
  });
}

splat_c_status splat_c_table_compute_morton_order(const splat_c_table* table, uint32_t* out_indices,
                                                  size_t index_count) {
  return guard([&] {
    requireOutput(out_indices, "out_indices");
    const auto& data_table = requireTable(table);
    const size_t rows = requireExactRowCount(data_table, index_count, "index_count");
    std::iota(out_indices, out_indices + rows, uint32_t{0});
    splat::sortMortonOrder(&data_table, absl::Span<uint32_t>(out_indices, rows));
  });
}

splat_c_status splat_c_table_compute_visibility_order(const splat_c_table* table, uint32_t* out_indices,
                                                      size_t index_count) {
  return guard([&] {
    requireOutput(out_indices, "out_indices");
    const auto& data_table = requireTable(table);
    const size_t rows = requireExactRowCount(data_table, index_count, "index_count");

    std::vector<unsigned int> indices(rows);
    std::iota(indices.begin(), indices.end(), 0u);
    splat::sortByVisibility(&data_table, indices);

    for (size_t i = 0; i < rows; ++i) {
      out_indices[i] = static_cast<uint32_t>(indices[i]);
    }
  });
}

splat_c_status splat_c_table_reorder(const splat_c_table* table, const uint32_t* indices, size_t index_count,
                                     splat_c_table** out_table) {
  return guard([&] {
    requireOutput(out_table, "out_table");
    *out_table = nullptr;
    if (indices == nullptr && index_count != 0) {
      throw std::invalid_argument("indices must not be null when index_count is non-zero");
    }

    std::vector<uint32_t> order;
    if (index_count != 0) {
      order.assign(indices, indices + index_count);
    }
    *out_table = makeOwnedTable(requireTable(table).permuteRows(order)).release();
  });
}

splat_c_status splat_c_table_simplify(const splat_c_table* table, int32_t target_count, splat_c_table** out_table) {
  return guard([&] {
    requireOutput(out_table, "out_table");
    *out_table = nullptr;
    *out_table = makeOwnedTable(splat::simplifyGaussians(requireTable(table), target_count)).release();
  });
}
