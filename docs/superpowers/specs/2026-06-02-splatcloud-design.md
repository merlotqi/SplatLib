# SplatCloud — Core Data Model Design

## Context

SplatLib currently uses `DataTable` (a generic columnar store with 8-type `std::variant`) as the universal intermediate format for all IO operations. `DataTable` was originally designed to align with the PlayCanvas viewer's GPU buffer layout, not as a domain model for 3D Gaussian splatting.

**Problems with the current approach:**

1. **String-keyed access**: `table.getColumnByName("x").getValue<float>(i)` — typos are runtime crashes, not compile errors.
2. **8-way variant**: 7 of 8 types are unused. All splat attributes are float32.
3. **Column name duplication**: Every reader/writer hardcodes the same column name strings independently.
4. **No ABI boundary**: Adding a column forces recompilation of all consumers.
5. **SH is shoehorned in**: `f_rest_0` through `f_rest_N` column names vs. a simple `shDegree` + flat buffer.

## Design Goals

1. **Compile-time safety** for core splat attributes (position, rotation, scale, color, opacity)
2. **pimpl isolation** — consumers see only the public header; implementation changes don't cascade
3. **Float-only storage** — all attributes are `std::vector<float>`; no variant
4. **Clean extension point** for transient/non-standard columns (lod, index, cluster_id)
5. **Simple IO signatures** — value semantics with efficient move (pimpl is a pointer swap)

---

## Public API

### Header: `include/splat/models/splat-cloud.h`

```cpp
#pragma once

#include <memory>
#include <string>
#include <vector>

namespace splat {

enum class CoordinateSystem : uint32_t;  // forward declare

class SplatCloud {
public:
  SplatCloud();
  ~SplatCloud();

  // Move
  SplatCloud(SplatCloud&&) noexcept;
  SplatCloud& operator=(SplatCloud&&) noexcept;

  // Copy (deep)
  SplatCloud(const SplatCloud&);
  SplatCloud& operator=(const SplatCloud&);

  // ── Capacity ──────────────────────────────────────────────────
  size_t size() const;
  bool   empty() const;
  void   resize(size_t n);
  void   clear();

  // ── Core attributes (compile-time safe, guaranteed present) ───

  // Position — 3 × float per point
  std::vector<float>&       x();
  std::vector<float>&       y();
  std::vector<float>&       z();
  const std::vector<float>& x() const;
  const std::vector<float>& y() const;
  const std::vector<float>& z() const;

  // Rotation quaternion — 4 × float per point (w, x, y, z)
  std::vector<float>&       rot_w();
  std::vector<float>&       rot_x();
  std::vector<float>&       rot_y();
  std::vector<float>&       rot_z();
  const std::vector<float>& rot_w() const;
  const std::vector<float>& rot_x() const;
  const std::vector<float>& rot_y() const;
  const std::vector<float>& rot_z() const;

  // Scale (log space) — 3 × float per point
  std::vector<float>&       scale_x();
  std::vector<float>&       scale_y();
  std::vector<float>&       scale_z();
  const std::vector<float>& scale_x() const;
  const std::vector<float>& scale_y() const;
  const std::vector<float>& scale_z() const;

  // Color (SH DC, f_dc space) — 3 × float per point
  std::vector<float>&       f_dc_r();
  std::vector<float>&       f_dc_g();
  std::vector<float>&       f_dc_b();
  const std::vector<float>& f_dc_r() const;
  const std::vector<float>& f_dc_g() const;
  const std::vector<float>& f_dc_b() const;

  // Opacity (logit space) — 1 × float per point
  std::vector<float>&       opacity();
  const std::vector<float>& opacity() const;

  // ── Spherical harmonics (variable dimension) ──────────────────
  //
  // Flat storage:  size() × shDim × 3  floats.
  // Layout: inner axis = channel (R,G,B), outer axis = coefficient.
  // shDim for each degree:  0→0, 1→3, 2→8, 3→15.
  //
  int  shDegree() const;
  void setShDegree(int degree);

  std::vector<float>&       sh();
  const std::vector<float>& sh() const;

  // ── Extras (transient / operation-specific columns) ───────────
  //
  // For columns that are not part of the core splat definition:
  //   lod, index, cluster_id, norm_x, distance, etc.
  // These are stored and accessed by name, but isolated from core API.
  //
  bool        hasExtra(const std::string& name) const;
  std::vector<float>&       extra(const std::string& name);
  const std::vector<float>& extra(const std::string& name) const;
  void        addExtra(const std::string& name);
  void        removeExtra(const std::string& name);
  std::vector<std::string>  extraNames() const;

  // ── Metadata ──────────────────────────────────────────────────
  CoordinateSystem coordinateSystem() const;
  void             setCoordinateSystem(CoordinateSystem cs);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace splat
```

### Implementation structure (private)

```cpp
// src/models/splat-cloud.cpp

namespace splat {

struct SplatCloud::Impl {
  // Core columns (guaranteed always present, same length)
  std::vector<float> _x, _y, _z;
  std::vector<float> _rot_w, _rot_x, _rot_y, _rot_z;
  std::vector<float> _scale_x, _scale_y, _scale_z;
  std::vector<float> _f_dc_r, _f_dc_g, _f_dc_b;
  std::vector<float> _opacity;

  // SH
  int                _shDegree = 0;
  std::vector<float> _sh;

  // Extensions (not part of core splat definition)
  // Note: all values are also float; no need for variant
  std::map<std::string, std::vector<float>> _extras;

  // Metadata
  CoordinateSystem _coordSystem = CoordinateSystem::RUB;
};

// Implementation is straightforward — delegate to impl
SplatCloud::SplatCloud() : impl_(std::make_unique<Impl>()) {}
SplatCloud::~SplatCloud() = default;

SplatCloud::SplatCloud(SplatCloud&&) noexcept = default;
SplatCloud& SplatCloud::operator=(SplatCloud&&) noexcept = default;

SplatCloud::SplatCloud(const SplatCloud& other)
    : impl_(std::make_unique<Impl>(*other.impl_)) {}
SplatCloud& SplatCloud::operator=(const SplatCloud& other) {
  *impl_ = *other.impl_;
  return *this;
}

size_t SplatCloud::size() const { return impl_->_x.size(); }

void SplatCloud::resize(size_t n) {
  impl_->_x.resize(n); impl_->_y.resize(n); impl_->_z.resize(n);
  impl_->_rot_w.resize(n); impl_->_rot_x.resize(n);
  impl_->_rot_y.resize(n); impl_->_rot_z.resize(n);
  impl_->_scale_x.resize(n); impl_->_scale_y.resize(n);
  impl_->_scale_z.resize(n);
  impl_->_f_dc_r.resize(n); impl_->_f_dc_g.resize(n);
  impl_->_f_dc_b.resize(n);
  impl_->_opacity.resize(n);
  if (impl_->_shDegree > 0) {
    int shDim = 0;
    switch (impl_->_shDegree) {
      case 1: shDim = 3; break;
      case 2: shDim = 8; break;
      case 3: shDim = 15; break;
      default: shDim = 0;
    }
    impl_->_sh.resize(n * shDim * 3);
  }
  for (auto& [_, v] : impl_->_extras) v.resize(n);
}

// ... remaining accessors follow the same delegate pattern
```

---

## Migration Path

### Phase 1: Introduce SplatCloud (non-breaking)

- Add `include/splat/models/splat-cloud.h` and `src/models/splat-cloud.cpp`
- Add `SplatCloud DataTable::toSplatCloud()` and `static DataTable SplatCloud::toDataTable(const SplatCloud&)` for interop
- Existing reader/writer APIs unchanged

### Phase 2: Reader outputs SplatCloud

- Add `SplatCloud readPlyV2(path)` (new function, old one stays)
- New readers return `SplatCloud` directly
- Coordinate conversion integrated: `cloud.setCoordinateSystem(...)` metadata field

### Phase 3: Writer inputs SplatCloud

- Add `void writeSpzV2(path, const SplatCloud&)`
- Encode functions operate on SplatCloud references

### Phase 4: Deprecate DataTable-based APIs

- Mark old `readPly(DataTable*)` functions as deprecated
- Remove after migration window

---

## IO API (Post-Migration)

```cpp
// include/splat/io/ply_reader.h
SplatCloud readPly(const std::filesystem::path& path);

// include/splat/io/spz_reader.h
SplatCloud readSpz(const std::filesystem::path& path);

// include/splat/io/spz_writer.h
void writeSpz(const std::filesystem::path& path, const SplatCloud& cloud);

// include/splat/io/sog_reader.h
SplatCloud readSog(const std::filesystem::path& path);

// include/splat/io/lcc_reader.h
std::vector<SplatCloud> readLcc(const std::filesystem::path& path);  // multi-LOD
```

---

## Verification

1. **Unit test**: Create SplatCloud, resize, write core+SH+extra columns, verify sizes
2. **Unit test**: Move and copy semantics (move leaves source empty, copy is deep)
3. **Unit test**: extra() lifecycle — add, hasExtra, access, remove, extraNames
4. **Integration**: readPly → SplatCloud → writeSpz → readSpz → verify attributes match
5. **Benchmark**: SplatCloud access vs DataTable access for 10M splat iteration
