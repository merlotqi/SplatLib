/**
 * @file splat/op/transform.h
 * @brief Apply rigid transforms to splat tables.
 *
 */
 
#pragma once

#include <Eigen/Dense>

namespace splat {

class DataTable;

/**
 * @brief Apply transformation to data table
 * @param dataTable Table to transform
 * @param t Translation vector
 * @param r Rotation quaternion
 * @param s Scale factor
 *
 * Applies translation, rotation, and scaling transformation to relevant
 * columns in the data table.
 */
void transform(DataTable* dataTable, const Eigen::Vector3f& t, const Eigen::Quaternionf& r, float s);

}  // namespace splat
