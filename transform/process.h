#include <splat/models/data-table.h>

#include <Eigen/Dense>
#include <string>
#include <variant>

namespace splat {

struct Translate {
  Eigen::Vector3f value;
};

struct Rotate {
  Eigen::Vector3f value;
};

struct Scale {
  float value;
};

struct FilterNaN {};

struct FilterByValue {
  std::string columnName;
  std::string comparator;  // lt, lte, gt, gte, eq, neq
  float value;
};

struct FilterBands {
  int value;  // 0, 1, 2, 3
};

struct FilterBox {
  Eigen::Vector3f min;
  Eigen::Vector3f max;
};

struct FilterSphere {
  Eigen::Vector3f center;
  float radius;
};

struct Param {
  std::string name;
  std::string value;
};

struct Lod {
  int value;
};

struct Decimate {
  int count = -1;
  float percent = -1.0f;
};

using ProcessAction =
    std::variant<Translate, Rotate, Scale, FilterNaN, FilterByValue, FilterBands, FilterBox, FilterSphere, Param, Lod,
                 Decimate>;

std::unique_ptr<DataTable> processDataTable(DataTable* dataTable, const std::vector<ProcessAction>& processActions);

}  // namespace splat
