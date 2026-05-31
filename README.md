# SplatLib

A modern C++ library for reading, writing, and processing 3D Gaussian Splatting files, designed for real-time neural rendering applications.

SplatLib provides comprehensive support for various Gaussian splat formats, enabling efficient conversion, manipulation, and optimization of 3D scene data for web-based real-time rendering and neural graphics applications.

## Core Features

### Multiple Format Support
- **PLY**: Industry-standard uncompressed format for training, editing, and archival storage
- **SOG**: Compressed format optimized for web delivery (15-20× smaller than PLY)
- **KSPLAT, SPZ, LCC**: Additional specialized formats for different use cases

### GPU Acceleration
- CUDA-based high-performance processing
- GPU-accelerated SOG compression and spatial algorithms
- Support for multiple CUDA compute capabilities (7.5, 8.0, 8.9)

### Spatial Data Structures
- **Octree**: Hierarchical spatial partitioning
- **K-D Tree**: Nearest neighbor searches
- **B-Tree**: Efficient data organization
- **Morton Encoding**: Spatial coherence optimization

### Mathematical Operations
- K-means clustering (CPU/GPU dual implementations)
- Spherical harmonic (SH) rotation and manipulation
- Coordinate transformations and data processing

### Advanced Features
- **Level of Detail (LOD)**: Chunk-based organization supporting multi-resolution representations
- **Data Compression**: Integrated WebP codec for efficient texture compression
- **VTK Visualization**: Optional `SplatGaussianProp` and `SplatVisualizer` APIs for Qt, ImGui, and VTK applications
- **Parallel Processing**: Built-in thread pools and parallel algorithms
- **Python Bindings**: pybind11-based Python interface

## Architecture Design

SplatLib follows a modular design with core components organized into the following modules:

### include/splat/ - Public API Interface

The library is organized into focused modules that provide specific functionality:

#### io/ - Input/Output Modules
- `ply_reader.h/ply_writer.h` - PLY format reading/writing
- `sog_reader.h/sog_writer.h` - SOG format reading/writing
- `compressed_ply_writer.h` - Compressed PLY writing
- `ksplat_reader.h/spz_reader.h/lcc_reader.h` - Specialized format readers
- `csv_writer.h` - CSV format output
- `lod_writer.h` - LOD data writing

#### models/ - Data Models
- `ply.h` - PLY file structure definitions (PlyHeader, PlyElement, PlyData)
- `sog.h` - SOG metadata structures (Meta, SHN, etc.)
- `data-table.h` - Generic data table structure supporting multiple column types

#### spatial/ - Spatial Data Structures
- `octree.h` - Octree implementation for spatial partitioning and queries
- `kdtree.h` - K-D tree implementation for nearest neighbor searches
- `btree.h` - B-tree implementation for efficient data organization
- `kmeans.h` - K-means clustering (CPU/GPU implementations)

#### maths/ - Mathematical Utilities
- `maths.h` - Basic mathematical operations
- `rotate-sh.h` - Spherical harmonic rotation operations

#### op/ - Operations Module
- `transform.h` - Coordinate transformation operations
- `combine.h` - Data combination operations
- `morton-order.h` - Morton spatial encoding

#### utils/ - Utilities Module
- `logger.h` - Logging infrastructure
- `threadpool.h` - Thread pool implementation
- `webp-codec.h` - WebP image encoding/decoding
- `zip-reader.h/zip-writer.h` - ZIP compression support
- `crc.h` - CRC checksum validation

### src/ - Implementation Details
- Mirrors the include/ directory structure with corresponding implementation files
- CUDA implementations located in files like `spatial/kmeans.cu`

## Design Principles

1. **Modularity**: Each functional module is self-contained for easy testing and maintenance
2. **Zero-Copy**: Prioritizes views and references to avoid unnecessary data copying
3. **Type Safety**: Uses strong typing to reduce runtime errors
4. **Performance-First**: GPU acceleration, parallel processing, and memory optimization
5. **Extensibility**: Plugin-style format support makes adding new file formats straightforward

## API Usage Examples

### Basic File I/O

```cpp
#include <splat/splat.h>

// Read a PLY file
auto data = splat::readPly("scene.ply");

// Write to compressed SOG format
splat::writeSog("scene.sog", *data);
```

### Spatial Operations

```cpp
#include <splat/spatial/octree.h>
#include <splat/models/data-table.h>

// Build spatial index
auto octree = std::make_unique<splat::Octree>(data.get(), /*maxPoints=*/32, /*maxDepth=*/8);

// Query operations
// ... spatial queries using the octree
```

### Mathematical Operations

```cpp
#include <splat/maths/rotate-sh.h>
#include <splat/op/transform.h>

// Apply spherical harmonic rotation
Eigen::Matrix3f rotation_matrix = /* ... */;
auto rotated_sh = splat::rotateSHCoefficients(original_sh, rotation_matrix);

// Coordinate transformation
auto transformed_data = splat::transform(*data, transformation_matrix);
```

### VTK Visualization

```cpp
#include <splat/splat.h>
#include <splat/visualization/splat_gaussian_prop.h>

#include <vtkRenderer.h>
#include <vtkSmartPointer.h>

auto data = splat::readPly("scene.ply");
auto prop = vtkSmartPointer<splat::SplatGaussianProp>::New();
prop->SetInputData(std::move(data));

vtkRenderer* renderer = /* owned by your Qt, ImGui, or VTK application */;
renderer->AddViewProp(prop);
```

## Extensibility and Future Development

### Adding New File Formats

To add support for a new file format:

1. **Define the data model** in `include/splat/models/`
   - Create structures for format-specific metadata
   - Define any custom data types needed

2. **Implement I/O operations** in `include/splat/io/`
   - Create reader/writer header files following existing patterns
   - Implement parsing/writing logic in corresponding src/ files

3. **Update the main header** (`include/splat/splat.h`)
   - Add includes for new headers
   - Ensure new functions are properly namespaced

4. **Add format detection** if needed
   - Extend file type detection logic
   - Add format-specific validation

### Extending Spatial Structures

When adding new spatial data structures:

1. **Design the interface** in `include/splat/spatial/`
   - Follow patterns established by existing structures (Octree, KdTree)
   - Consider memory layout and cache efficiency

2. **Implement core algorithms** in src/
   - Support both CPU and GPU implementations where applicable
   - Include comprehensive error handling

3. **Add performance optimizations**
   - Vectorization for CPU operations
   - GPU acceleration for compute-intensive tasks
   - Memory pool management for large datasets

### Mathematical Extensions

For new mathematical operations:

1. **Create utility functions** in `include/splat/maths/`
   - Use Eigen for matrix operations
   - Support both float and double precision

2. **Add operation modules** in `include/splat/op/`
   - Implement composable operations
   - Support batched processing

3. **GPU acceleration** where beneficial
   - CUDA kernels for compute-intensive operations
   - Memory transfer optimization

## Dependencies

### Required
- **CUDA** (compute capability 7.5, 8.0, or 8.9) - GPU acceleration
- **Eigen3** - Linear algebra library
- **WebP** - Image compression library
- **nlohmann_json** - JSON parsing
- **Abseil** - C++ utilities
- **ZLIB** - Compression library

### Optional
- **Doxygen** - API documentation generation
- **pybind11** - Python bindings

## Building

```bash
# Clone the repository
git clone https://github.com/merlotqi/SplatLib.git
cd SplatLib

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -DBUILD_SPLAT_TRANSFORM_TOOL=ON -DBUILD_PYTHON_BINDINGS=OFF

# Build
make -j$(nproc)
```

### CMake Options
- `BUILD_SPLAT_TRANSFORM_TOOL` - Build command-line transform utility (default: OFF)
- `BUILD_SPLAT_VISUALIZATION` - Build VTK visualization helpers (default: OFF)
- `BUILD_SPLAT_PLAYCANVAS_VIEWER` - Build standalone PlayCanvas-style OpenGL viewer (default: OFF)
- `BUILD_PYTHON_BINDINGS` - Build Python bindings (default: OFF)
- `ENABLE_CLANG_TIDY` - Enable clang-tidy static analysis (default: OFF)

## Project Structure

```
SplatLib/
├── include/splat/          # Public API headers
├── src/                   # Implementation files
├── python/                # Python bindings
├── transform/             # Command-line tool (optional)
├── docs/                  # Documentation
├── data/                  # Example data
├── thirdparty/            # External dependencies
└── cmake/                 # CMake utilities
```

## Documentation

- [PLY Format Specification](docs/ply.md) - Industry standard for Gaussian splats
- [SOG Format Specification](docs/sog.md) - Compressed format for web delivery
- [Visualization](docs/visualization.md) - VTK prop and managed viewer integration
- [API Documentation](docs/index.md) - Library overview and format comparison

Generate Doxygen documentation:
```bash
cd build && make doc
# Open docs/html/index.html
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

## License

Licensed under GNU General Public License v3.0. See [LICENSE](LICENSE) and [COPYRIGHT](COPYRIGHT) for details.

## Version

Current version: **1.2.0**

See [ChangeLog](ChangeLog) for version history.
