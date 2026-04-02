/**
 * @file binding.cpp
 * @brief Python bindings for the splat library using pybind11.
 *
 * Exposes DataTable, Column, I/O functions, and operations to Python.
 */

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <splat/splat.h>

#include <Eigen/Dense>
#include <memory>
#include <sstream>
#include <stdexcept>


namespace py = pybind11;
using namespace splat;

// Helper: Convert numpy array to Column
static Column numpy_to_column(const std::string& name, const py::array& arr) {
  py::buffer_info buf = arr.request();

  if (buf.ndim != 1) {
    throw std::runtime_error("Array must be 1-dimensional");
  }

  if (buf.format == py::format_descriptor<float>::format()) {
    std::vector<float> data(static_cast<float*>(buf.ptr), static_cast<float*>(buf.ptr) + buf.size);
    return Column{name, std::move(data)};
  } else if (buf.format == py::format_descriptor<double>::format()) {
    std::vector<double> data(static_cast<double*>(buf.ptr), static_cast<double*>(buf.ptr) + buf.size);
    return Column{name, std::move(data)};
  } else if (buf.format == py::format_descriptor<int32_t>::format()) {
    std::vector<int32_t> data(static_cast<int32_t*>(buf.ptr), static_cast<int32_t*>(buf.ptr) + buf.size);
    return Column{name, std::move(data)};
  } else if (buf.format == py::format_descriptor<uint32_t>::format()) {
    std::vector<uint32_t> data(static_cast<uint32_t*>(buf.ptr), static_cast<uint32_t*>(buf.ptr) + buf.size);
    return Column{name, std::move(data)};
  } else if (buf.format == py::format_descriptor<int16_t>::format()) {
    std::vector<int16_t> data(static_cast<int16_t*>(buf.ptr), static_cast<int16_t*>(buf.ptr) + buf.size);
    return Column{name, std::move(data)};
  } else if (buf.format == py::format_descriptor<uint16_t>::format()) {
    std::vector<uint16_t> data(static_cast<uint16_t*>(buf.ptr), static_cast<uint16_t*>(buf.ptr) + buf.size);
    return Column{name, std::move(data)};
  } else if (buf.format == py::format_descriptor<int8_t>::format()) {
    std::vector<int8_t> data(static_cast<int8_t*>(buf.ptr), static_cast<int8_t*>(buf.ptr) + buf.size);
    return Column{name, std::move(data)};
  } else if (buf.format == py::format_descriptor<uint8_t>::format()) {
    std::vector<uint8_t> data(static_cast<uint8_t*>(buf.ptr), static_cast<uint8_t*>(buf.ptr) + buf.size);
    return Column{name, std::move(data)};
  }
  throw std::runtime_error("Unsupported numpy array type: " + buf.format);
}

// Helper: Convert Column to numpy array
static py::array column_to_numpy(const Column& col) {
  switch (col.getType()) {
    case ColumnType::FLOAT32: {
      const auto& vec = col.asVector<float>();
      auto arr = py::array_t<float>(vec.size());
      std::memcpy(arr.mutable_data(), vec.data(), vec.size() * sizeof(float));
      return arr;
    }
    case ColumnType::FLOAT64: {
      const auto& vec = col.asVector<double>();
      auto arr = py::array_t<double>(vec.size());
      std::memcpy(arr.mutable_data(), vec.data(), vec.size() * sizeof(double));
      return arr;
    }
    case ColumnType::INT32: {
      const auto& vec = col.asVector<int32_t>();
      auto arr = py::array_t<int32_t>(vec.size());
      std::memcpy(arr.mutable_data(), vec.data(), vec.size() * sizeof(int32_t));
      return arr;
    }
    case ColumnType::UINT32: {
      const auto& vec = col.asVector<uint32_t>();
      auto arr = py::array_t<uint32_t>(vec.size());
      std::memcpy(arr.mutable_data(), vec.data(), vec.size() * sizeof(uint32_t));
      return arr;
    }
    case ColumnType::INT16: {
      const auto& vec = col.asVector<int16_t>();
      auto arr = py::array_t<int16_t>(vec.size());
      std::memcpy(arr.mutable_data(), vec.data(), vec.size() * sizeof(int16_t));
      return arr;
    }
    case ColumnType::UINT16: {
      const auto& vec = col.asVector<uint16_t>();
      auto arr = py::array_t<uint16_t>(vec.size());
      std::memcpy(arr.mutable_data(), vec.data(), vec.size() * sizeof(uint16_t));
      return arr;
    }
    case ColumnType::INT8: {
      const auto& vec = col.asVector<int8_t>();
      auto arr = py::array_t<int8_t>(vec.size());
      std::memcpy(arr.mutable_data(), vec.data(), vec.size() * sizeof(int8_t));
      return arr;
    }
    case ColumnType::UINT8: {
      const auto& vec = col.asVector<uint8_t>();
      auto arr = py::array_t<uint8_t>(vec.size());
      std::memcpy(arr.mutable_data(), vec.data(), vec.size() * sizeof(uint8_t));
      return arr;
    }
  }
  throw std::runtime_error("Unknown column type");
}

// Read a column from DataTable as numpy array
static py::array get_column_data(const DataTable& dt, int idx) {
  if (idx < 0 || idx >= static_cast<int>(dt.getNumColumns())) {
    throw std::out_of_range("Column index out of range");
  }
  return column_to_numpy(dt.getColumn(idx));
}

static py::array get_column_data_by_name(const DataTable& dt, const std::string& name) {
  return column_to_numpy(dt.getColumnByName(name));
}

// Create DataTable from dict of {name: numpy_array}
static std::unique_ptr<DataTable> datatable_from_dict(const py::dict& d) {
  std::vector<Column> cols;
  cols.reserve(d.size());
  for (auto [key, val] : d) {
    std::string name = py::cast<std::string>(key);
    py::array arr = py::cast<py::array>(val);
    cols.push_back(numpy_to_column(name, arr));
  }
  return std::make_unique<DataTable>(cols);
}

// Convert DataTable to dict of {name: numpy_array}
static py::dict datatable_to_dict(const DataTable& dt) {
  py::dict d;
  for (size_t i = 0; i < dt.getNumColumns(); i++) {
    const auto& col = dt.getColumn(i);
    d[col.name.c_str()] = column_to_numpy(col);
  }
  return d;
}

// Get a row as tuple
static py::tuple get_row_tuple(const DataTable& dt, size_t idx) {
  if (idx >= dt.getNumRows()) {
    throw std::out_of_range("Row index out of range");
  }
  py::tuple t(dt.getNumColumns());
  for (size_t j = 0; j < dt.getNumColumns(); j++) {
    t[j] = dt.getColumn(j).getValue<float>(idx);
  }
  return t;
}

// Set row from tuple
static void set_row_tuple(DataTable& dt, size_t idx, const py::tuple& t) {
  if (idx >= dt.getNumRows()) {
    throw std::out_of_range("Row index out of range");
  }
  if (t.size() != dt.getNumColumns()) {
    throw std::runtime_error("Tuple size doesn't match column count");
  }
  Row row;
  for (size_t j = 0; j < dt.getNumColumns(); j++) {
    row[dt.getColumn(j).name] = py::cast<float>(t[j]);
  }
  dt.setRow(idx, row);
}

// Get column type as string
static std::string column_type_str(const Column& col) {
  switch (col.getType()) {
    case ColumnType::INT8:
      return "int8";
    case ColumnType::UINT8:
      return "uint8";
    case ColumnType::INT16:
      return "int16";
    case ColumnType::UINT16:
      return "uint16";
    case ColumnType::INT32:
      return "int32";
    case ColumnType::UINT32:
      return "uint32";
    case ColumnType::FLOAT32:
      return "float32";
    case ColumnType::FLOAT64:
      return "float64";
  }
  return "unknown";
}

PYBIND11_MODULE(splat_transform_cpp, m) {
  m.doc() = "Python bindings for the splat library - 3D Gaussian Splatting I/O and operations";

  // ========== Column class ==========
  py::class_<Column>(m, "Column")
      .def_readonly("name", &Column::name)
      .def_property_readonly("dtype", &column_type_str)
      .def_property_readonly("length", &Column::length)
      .def("get_data", &column_to_numpy, "Get column data as numpy array")
      .def("__len__", &Column::length)
      .def("__repr__", [](const Column& c) {
        return "<Column name='" + c.name + "' dtype=" + column_type_str(c) + " len=" + std::to_string(c.length()) + ">";
      });

  // ========== DataTable class ==========
  py::class_<DataTable>(m, "DataTable")
      .def(py::init<>(), "Create empty DataTable")
      .def_readonly("columns", &DataTable::columns)

      // Properties
      .def_property_readonly("num_rows", &DataTable::getNumRows)
      .def_property_readonly("num_columns", &DataTable::getNumColumns)
      .def_property_readonly("column_names", &DataTable::getColumnNames)
      .def_property_readonly("column_types",
                             [](const DataTable& dt) {
                               std::vector<std::string> types;
                               for (size_t i = 0; i < dt.getNumColumns(); i++) {
                                 types.push_back(column_type_str(dt.getColumn(i)));
                               }
                               return types;
                             })

      // Column access
      .def("get_column", py::overload_cast<size_t>(&DataTable::getColumn, py::const_), "Get column by index")
      .def("get_column_by_name", py::overload_cast<const std::string&>(&DataTable::getColumnByName, py::const_),
           "Get column by name")
      .def("has_column", &DataTable::hasColumn, "Check if column exists")
      .def("get_column_index", &DataTable::getColumnIndex, "Get column index by name")
      .def("get_column_data", &get_column_data, "Get column data as numpy array by index")
      .def("get_column_data_by_name", &get_column_data_by_name, "Get column data as numpy array by name")

      // Row access
      .def(
          "get_row",
          [](const DataTable& dt, size_t idx) -> std::map<std::string, float> {
            Row row;
            dt.getRow(idx, row);
            return row;
          },
          "Get row as dict")
      .def("get_row_tuple", &get_row_tuple, "Get row as tuple of floats")
      .def("set_row_tuple", &set_row_tuple, "Set row from tuple of floats")

      // Modification
      .def(
          "add_column",
          [](DataTable& dt, const std::string& name, const py::array& arr) {
            dt.addColumn(numpy_to_column(name, arr));
          },
          "Add column from numpy array")
      .def("remove_column", &DataTable::removeColumn, "Remove column by name")
      .def("clone", &DataTable::clone, py::arg("column_names") = std::vector<std::string>{}, "Create a deep copy")

      // Utility
      .def(
          "permute_rows",
          [](const DataTable& dt, const std::vector<uint32_t>& indices) { return dt.permuteRows(indices); },
          "Create new table with permuted rows")
      .def("to_dict", &datatable_to_dict, "Convert to dict of {column_name: numpy_array}")
      .def_static("from_dict", &datatable_from_dict, "Create from dict of {column_name: numpy_array}")
      .def("__repr__",
           [](const DataTable& dt) {
             return "<DataTable rows=" + std::to_string(dt.getNumRows()) +
                    " cols=" + std::to_string(dt.getNumColumns()) + ">";
           })
      .def(
          "__getitem__",
          [](const DataTable& dt, const std::string& name) -> py::array {
            return column_to_numpy(dt.getColumnByName(name));
          },
          "Get column data by name (dt['name'])")
      .def(
          "__contains__", [](const DataTable& dt, const std::string& name) -> bool { return dt.hasColumn(name); },
          "Check if column exists (name in dt)");

  // ========== Rotation SH ==========
  py::class_<RotateSH>(m, "RotateSH")
      .def(py::init<const Eigen::Matrix3f&>(), "Create from 3x3 rotation matrix")
      .def(
          "apply", [](RotateSH& self, std::vector<float>& result, std::vector<float> src) { self.apply(result, src); },
          py::arg("result"), py::arg("src") = std::vector<float>{}, "Apply rotation to SH coefficients")
      .def_readwrite("sh1", &RotateSH::sh1)
      .def_readwrite("sh2", &RotateSH::sh2)
      .def_readwrite("sh3", &RotateSH::sh3);

  // ========== sigmoid ==========
  m.def("sigmoid", &sigmoid<float>, "Sigmoid activation function");

  // ========== I/O Functions ==========

  // Readers
  m.def("read_ply", &readPly, py::arg("filename"), "Read PLY file → DataTable");
  m.def("read_splat", &readSplat, py::arg("filename"), "Read .splat binary file → DataTable");
  m.def("read_sog", &readSog, py::arg("file"), py::arg("source_name"), "Read SOG file → DataTable");
  m.def("read_spz", &readSpz, py::arg("filename"), "Read SPZ file → DataTable");
  m.def("read_ksplat", &readKsplat, py::arg("filename"), "Read .ksplat file → DataTable");
  m.def("read_lcc", &readLcc, py::arg("filename"), py::arg("source_name"), py::arg("options"),
        "Read LCC file → List[DataTable]");
  m.def("read_voxel", &readVoxel, py::arg("voxel_json_path"), "Read voxel file → DataTable");

  // Writers
  m.def("write_ply", &writePly, py::arg("filename"), py::arg("ply_data"), "Write PLY file");
  m.def("write_splat", &writeSplat, py::arg("datatable"), py::arg("filepath"), "Write .splat binary file");
  m.def("write_sog", &writeSog, py::arg("filename"), py::arg("data_table"), py::arg("bundle") = true,
        py::arg("iterations") = 10, py::arg("indices") = std::vector<uint32_t>{}, "Write SOG file");
  m.def("write_csv", &writeCSV, py::arg("path"), py::arg("data_table"), "Write CSV file");
  m.def("write_glb", &writeGlb, py::arg("filename"), py::arg("data_table"), "Write GLB file");

  // ========== Operations ==========
  m.def("combine", &combine, py::arg("data_tables"), "Merge multiple DataTables into one");
  m.def("simplify_gaussians", &simplifyGaussians, py::arg("data_table"), py::arg("target_count"),
        "Simplify splat count using pairwise merging");
  m.def(
      "sort_morton_order", [](const DataTable* dt, std::vector<uint32_t>& indices) { sortMortonOrder(dt, indices); },
      py::arg("data_table"), py::arg("indices"), "Sort indices in Morton order (in-place)");
  m.def(
      "transform",
      [](DataTable* dt, const Eigen::Vector3f& t, const Eigen::Quaternionf& r, float s) { transform(dt, t, r, s); },
      py::arg("data_table"), py::arg("translation"), py::arg("rotation"), py::arg("scale"),
      "Apply translation, rotation, and scaling");

  // ========== WriteGlbOptions ==========
  py::class_<WriteGlbOptions>(m, "WriteGlbOptions")
      .def(py::init<>())
      .def_readwrite("filename", &WriteGlbOptions::filename)
      .def_readwrite("data_table", &WriteGlbOptions::data_table);

  // ========== WriteVoxelOptions ==========
  py::class_<WriteVoxelOptions>(m, "WriteVoxelOptions")
      .def(py::init<>())
      .def_readwrite("filename", &WriteVoxelOptions::filename)
      .def_readwrite("data_table", &WriteVoxelOptions::data_table)
      .def_readwrite("voxel_resolution", &WriteVoxelOptions::voxel_resolution)
      .def_readwrite("opacity_cutoff", &WriteVoxelOptions::opacity_cutoff)
      .def_readwrite("cuda_device_index", &WriteVoxelOptions::cuda_device_index)
      .def_readwrite("collision_mesh", &WriteVoxelOptions::collision_mesh)
      .def_readwrite("mesh_simplify", &WriteVoxelOptions::mesh_simplify);

  m.def("write_voxel", &writeVoxel, py::arg("options"), "Write voxel file (requires CUDA)");

  // ========== Utility ==========
  m.def("get_splat_version", []() -> std::string { return "0.1.0"; }, "Get library version string");
}
