/**
 * @file splat/models/ply.h
 * @brief PLY-related helpers and types.
 *
 * References:
 * - [PLY format](http://paulbourke.net/dataformats/ply/)
 */

#pragma once

#include <splat/models/splatcloud.h>

namespace splat {

/**
 * @struct PlyProperty
 * @brief Represents a single property definition in a PLY element
 *
 * Describes a property within a PLY file element, including its name,
 * data type as string, and the corresponding ColumnType for internal use.
 */
struct PlyProperty {
  std::string name;     ///< Property name (e.g., 'x', 'f_dc_0', 'opacity')
  std::string type;     ///< PLY type string (e.g., 'float', 'uchar', 'int32')
  ColumnType dataType;  ///< Internal ColumnType representation
};

/**
 * @struct PlyElement
 * @brief Represents a PLY element (group of properties) in the header
 *
 * Defines an element in a PLY file, such as 'vertex' or 'face', containing
 * multiple properties and the count of items of this element type.
 */
struct PlyElement {
  std::string name;                     ///< Element name (e.g., 'vertex', 'face')
  size_t count;                         ///< Number of items of this element type
  std::vector<PlyProperty> properties;  ///< List of properties in this element
};

/**
 * @struct PlyHeader
 * @brief Complete PLY file header information
 *
 * Contains all metadata from a PLY file header, including comments
 * and element definitions.
 */
struct PlyHeader {
  std::vector<std::string> comments;  ///< Comment lines from the PLY header
  std::vector<PlyElement> elements;   ///< Element definitions in the PLY file
};

/**
 * @struct PlyElementData
 * @brief Contains actual data for a PLY element along with its name
 *
 * Associates a SplatCloud containing the actual property values with
 * the corresponding element name from the PLY file.
 */
struct PlyElementData {
  std::string name;                       ///< Element name (must match PlyElement::name)
  std::unique_ptr<SplatCloud> dataTable;  ///< Data table containing property values
};

/**
 * @struct PlyData
 * @brief Complete in-memory representation of PLY file data
 *
 * Contains both the original comments and all element data from
 * a parsed PLY file. This is the primary structure for working with
 * PLY data in memory.
 */
struct PlyData {
  std::vector<std::string> comments;     ///< Comments from the PLY file
  std::vector<PlyElementData> elements;  ///< Data for each element in the PLY file
};

}  // namespace splat
