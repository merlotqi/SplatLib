#include "process.h"

#include <cmath>
#include <functional>
#include <stdexcept>

#include <splat/op/decimate.h>


namespace splat {

namespace {

Column* findColumnByName(DataTable* dataTable, const std::string& name) {
  if (!dataTable) {
    return nullptr;
  }

  const int index = dataTable->getColumnIndex(name);
  if (index < 0) {
    return nullptr;
  }

  return &dataTable->getColumn(static_cast<size_t>(index));
}

Column& ensureLodColumn(DataTable* dataTable) {
  Column* lodColumn = findColumnByName(dataTable, "lod");
  if (!lodColumn) {
    dataTable->addColumn({"lod", std::vector<float>(dataTable->getNumRows())});
    lodColumn = findColumnByName(dataTable, "lod");
  }

  return *lodColumn;
}

int computeKeepCount(size_t numRows, const Decimate& action) {
  int keepCount = static_cast<int>(numRows);

  if (action.count >= 0) {
    keepCount = action.count;
  } else if (action.percent >= 0.0f) {
    keepCount = static_cast<int>(
        std::lround(static_cast<double>(numRows) * static_cast<double>(action.percent) / 100.0));
  } else {
    throw std::runtime_error("Decimate action missing count and percent");
  }

  if (keepCount < 0) {
    keepCount = 0;
  }

  const int maxRows = static_cast<int>(numRows);
  if (keepCount > maxRows) {
    keepCount = maxRows;
  }

  return keepCount;
}

}  // namespace

static std::unique_ptr<DataTable> filter(const DataTable* dataTable,
                                         std::function<bool(const Row&, size_t)> predicate) {
  std::vector<uint32_t> indices;
  const size_t numRows = dataTable->getNumRows();
  indices.reserve(numRows);

  size_t index = 0;
  Row row;
  for (size_t i = 0; i < dataTable->getNumRows(); i++) {
    dataTable->getRow(i, row);
    if (predicate && predicate(row, i)) {
      indices.push_back(static_cast<uint32_t>(i));
    }
  }

  return dataTable->permuteRows(indices);
}

std::unique_ptr<DataTable> processDataTable(DataTable* dataTable, const std::vector<ProcessAction>& processActions) {
  assert(dataTable);
  std::unique_ptr<DataTable> result;
  result.reset(dataTable);

  for (const auto& action : processActions) {
    if (std::holds_alternative<Param>(action)) {
      continue;
    }

    if (const auto* lod = std::get_if<Lod>(&action)) {
      auto& lodValues = ensureLodColumn(result.get()).asVector<float>();
      const float lodValue = static_cast<float>(lod->value);
      for (float& value : lodValues) {
        value = lodValue;
      }
      continue;
    }

    if (const auto* decimate = std::get_if<Decimate>(&action)) {
      result = simplifyGaussians(*result, computeKeepCount(result->getNumRows(), *decimate));
      continue;
    }

    throw std::runtime_error("Unsupported process action in current translation");
  }

  return result;
}

}  // namespace splat
