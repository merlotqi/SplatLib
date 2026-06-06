# Contributing to SplatLib

Thank you for your interest in contributing to SplatLib! This document provides guidelines and information for contributors.

## Development Setup

### Prerequisites

- **C++17 compatible compiler** (GCC 7+, Clang 5+, MSVC 2017+)
- **CUDA 11.0+** with compute capability 7.5, 8.0, or 8.9
- **CMake 3.10+**
- **Git**

### Building from Source

```bash
# Clone the repository
git clone https://github.com/merlotqi/SplatLib.git
cd SplatLib

# Create build directory
mkdir build && cd build

# Configure (enable all development features)
cmake .. -DBUILD_SPLAT_TRANSFORM_TOOL=ON \
         -DBUILD_PYTHON_BINDINGS=ON \
         -DENABLE_CLANG_TIDY=ON

# Build
make -j$(nproc)
```

### Development Tools

Enable development tools for better code quality:

```bash
# Enable clang-tidy for static analysis
cmake .. -DENABLE_CLANG_TIDY=ON

# Generate compile commands for IDE integration
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

## Code Organization

### Directory Structure

```
include/splat/          # Public API headers
├── io/                 # File I/O operations
├── models/             # Data structures and models
├── spatial/            # Spatial data structures
├── maths/              # Mathematical utilities
├── op/                 # Operations and transformations
└── utils/              # Utility functions

src/                    # Implementation files
├── io/                 # I/O implementation
├── models/             # Model implementation
├── spatial/            # Spatial structure implementation
├── maths/              # Math implementation
├── op/                 # Operations implementation
└── utils/              # Utilities implementation
```

### Module Organization

Each module follows a consistent pattern:

- **Header files** (`include/splat/module/`) define the public API
- **Implementation files** (`src/module/`) contain the actual code
- **Tests** should be added for new functionality
- **Documentation** should be updated for API changes

## Coding Standards

### C++ Standards

- **C++17** as the minimum standard
- **CUDA C++17** for GPU code
- Use **smart pointers** (`std::unique_ptr`, `std::shared_ptr`) instead of raw pointers
- **RAII** (Resource Acquisition Is Initialization) for resource management
- **Exception safety** where appropriate

### Naming Conventions

- **Classes and structs**: `PascalCase` (e.g., `SplatCloud`, `PlyHeader`)
- **Functions and methods**: `camelCase` (e.g., `readPly()`, `writeSog()`)
- **Variables**: `camelCase` (e.g., `dataTable`, `filePath`)
- **Constants**: `SCREAMING_SNAKE_CASE` (e.g., `MAX_BUFFER_SIZE`)
- **Namespaces**: `lowercase` (e.g., `namespace splat`)

### Code Style

- **Include guards**: Use `#pragma once` for header files
- **Header includes**: Group by type (standard library, third-party, local)
- **Function documentation**: Use Doxygen format for public APIs
- **Error handling**: Use exceptions for exceptional cases, return codes for expected failures
- **Memory management**: Prefer stack allocation, use smart pointers for heap allocation

### Example

```cpp
#pragma once

#include <memory>
#include <string>
#include <vector>

namespace splat {

/**
 * @brief Reads PLY file and returns data table
 * @param filename Path to PLY file
 * @return Unique pointer to data table containing vertex data
 * @throws std::runtime_error if file cannot be read
 */
std::unique_ptr<SplatCloud> readPly(const std::string& filename);

}  // namespace splat
```

## Adding New Features

### 1. File Format Support

To add support for a new file format:

#### Define Data Structures
```cpp
// include/splat/models/new_format.h
struct NewFormatHeader {
  int version;
  size_t numElements;
  // ... format-specific fields
};

struct NewFormatData {
  NewFormatHeader header;
  std::unique_ptr<SplatCloud> dataTable;
};
```

#### Implement I/O Operations
```cpp
// include/splat/io/new_format_reader.h
std::unique_ptr<SplatCloud> readNewFormat(const std::string& filename);

// include/splat/io/new_format_writer.h
void writeNewFormat(const std::string& filename, const SplatCloud& data);
```

#### Update Main Header
```cpp
// include/splat/splat.h
#include <splat/io/new_format_reader.h>
#include <splat/io/new_format_writer.h>
```

#### Add Implementation
```cpp
// src/io/new_format_reader.cpp
// Implementation of readNewFormat()

// src/io/new_format_writer.cpp
// Implementation of writeNewFormat()
```

### 2. Spatial Data Structures

When adding new spatial data structures:

#### Design Interface
```cpp
// include/splat/spatial/new_structure.h
class NewStructure {
 public:
  explicit NewStructure(SplatCloud* table, /* parameters */);

  // Core operations
  void build();
  std::vector<size_t> query(const Query& q) const;

 private:
  SplatCloud* dataTable_;
  // Internal data structures
};
```

#### Implementation Considerations
- Support both CPU and GPU implementations where applicable
- Include memory-efficient data structures
- Provide configurable parameters for performance tuning
- Implement proper error handling and validation

### 3. Mathematical Operations

For new mathematical operations:

#### Utility Functions
```cpp
// include/splat/maths/new_math.h
namespace maths {

// Mathematical utility functions
Eigen::Vector3f computeSomething(const Eigen::Vector3f& input);

}  // namespace maths
```

#### Operation Modules
```cpp
// include/splat/op/new_operation.h
namespace op {

// Data transformation operations
std::unique_ptr<SplatCloud> applyNewOperation(
    const SplatCloud& input,
    const OperationParameters& params);

}  // namespace op
```

## Testing

### Unit Tests

- Add unit tests for new functionality
- Use Google Test framework (if available) or implement simple test functions
- Test both success and failure cases
- Include edge cases and boundary conditions

### Integration Tests

- Test complete workflows (read → process → write)
- Verify data integrity through format conversions
- Test performance characteristics
- Validate GPU/CPU consistency

### Test Data

- Use the `data/` directory for test datasets
- Include small, representative test files
- Document test data sources and licenses

## Documentation

### API Documentation

- Use Doxygen comments for all public APIs
- Document parameters, return values, and exceptions
- Provide usage examples where helpful

```cpp
/**
 * @brief Performs operation on data table
 *
 * Detailed description of what the function does,
 * how it works, and any important considerations.
 *
 * @param[in] input Input data table
 * @param[in] params Operation parameters
 * @return Processed data table
 *
 * @throws std::invalid_argument if parameters are invalid
 *
 * @note This function modifies the input data in-place
 *
 * Example usage:
 * @code
 * auto result = processData(input, params);
 * @endcode
 */
```

### Format Documentation

- Update `docs/` with new format specifications
- Include file format details, encoding schemes, and limitations
- Provide examples and reference implementations

## Performance Considerations

### GPU Acceleration

- Identify compute-intensive operations suitable for GPU acceleration
- Use CUDA streams for concurrent operations
- Minimize host-device memory transfers
- Profile and optimize kernel performance

### Memory Management

- Prefer stack allocation for small objects
- Use memory pools for frequent allocations
- Implement streaming for large datasets
- Monitor memory usage in performance-critical code

### Parallel Processing

- Use thread pools for CPU parallelism
- Implement concurrent algorithms where beneficial
- Avoid false sharing in multi-threaded code
- Profile contention and synchronization overhead

## Pull Request Process

### Before Submitting

1. **Code Review**: Self-review your code before requesting review
2. **Tests**: Ensure all tests pass and add new tests for new functionality
3. **Documentation**: Update documentation for API changes
4. **Style**: Ensure code follows project conventions
5. **Performance**: Verify performance is acceptable and document any trade-offs

### PR Description

Provide a clear description including:

- **What** the change does
- **Why** the change is needed
- **How** the change is implemented
- **Testing** performed
- **Performance impact** (if any)
- **Breaking changes** (if any)

### Review Process

- Address review comments promptly
- Be open to feedback and suggestions
- Keep discussions focused and productive
- Iterate on implementation as needed

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help create a positive community
- Follow the [GNU GPL v3.0](LICENSE) license terms

## Getting Help

- **Issues**: Use GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub discussions for questions and design discussions
- **Documentation**: Check existing docs before asking questions

Thank you for contributing to SplatLib!
