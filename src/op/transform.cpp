#include <splat/maths/rotate-sh.h>
#include <splat/models/splatcloud.h>
#include <splat/op/transform.h>

#include <iostream>

namespace splat {

// Pre-define SH coefficient column names (f_rest_0 to f_rest_44)
const std::vector<std::string> shNames = []() {
  std::vector<std::string> names(45);
  for (int i = 0; i < 45; ++i) {
    names[i] = "f_rest_" + std::to_string(i);
  }
  return names;
}();

/**
 * @brief Applies translation, rotation, and scale to all Gaussian points in a SplatCloud.
 * * @param dataTable The SplatCloud containing the Gaussian data (positions, rotations, scales, SH).
 * @param t Global translation vector (Vec3).
 * @param r Global rotation quaternion (Quat).
 * @param s Global uniform scale factor (float).
 * @throws std::runtime_error if dataTable operations fail.
 */
void transform(SplatCloud* dataTable, const Eigen::Vector3f& t, const Eigen::Quaternionf& r, float s) {
  assert(dataTable);

  // 1. Pre-calculate global transformation matrices and SH rotation utility

  // Mat4: Global Transformation Matrix (TRS)
  // Eigen uses Column-Major by default. PlayCanvas's setTRS results in a matrix that,
  // when applied to a vector (v' = M * v), performs T * R * S.
  Eigen::Matrix4f mat = Eigen::Matrix4f::Identity();
  mat.block<3, 3>(0, 0) = r.toRotationMatrix() * s;  // R * S
  mat.block<3, 1>(0, 3) = t;                         // T (translation column)

  // Mat3: Pure Rotation Matrix for SH rotation
  Eigen::Matrix3f mat3 = r.toRotationMatrix();
  RotateSH rotateSH(mat3.cast<float>());  // Use float for Eigen's RotateSH as per assumed definition

  // 2. Determine which components exist in the SplatCloud (Optimization)

  const bool hasTranslation = dataTable->hasColumn("x") && dataTable->hasColumn("y") && dataTable->hasColumn("z");
  const bool hasRotation = dataTable->hasColumn("rot_0") && dataTable->hasColumn("rot_1") &&
                           dataTable->hasColumn("rot_2") && dataTable->hasColumn("rot_3");
  const bool hasScale =
      dataTable->hasColumn("scale_0") && dataTable->hasColumn("scale_1") && dataTable->hasColumn("scale_2");

  // Determine SH bands and coefficient count
  // The original logic finds the first missing f_rest_i column to infer the band count.
  // Index: 0-8 (9 coeffs, L0+L1), 0-23 (24 coeffs, L0-L2), 0-44 (45 coeffs, L0-L3)
  int missingIndex = -1;
  for (int i = 0; i < 45; ++i) {
    if (!dataTable->hasColumn(shNames[i])) {
      missingIndex = i;
      break;
    }
  }

  // Mapping from (first missing index) to (SH Band Index: 0=None, 1=L1, 2=L2, 3=L3)
  // 45 is total max, 9 is L0+L1, 24 is L0-L2.
  int shBands = 0;           // 0=None, 1=L1 (9 coeffs), 2=L2 (24 coeffs), 3=L3 (45 coeffs)
  if (missingIndex == -1) {  // All 45 present (L0-L3)
    shBands = 3;
  } else if (missingIndex <= 9) {  // Only L0 (9/3=3 coeffs) or L1 (9 coeffs)
    // NOTE: The original logic is ambiguous for L0 (1 coeff). Assuming if f_rest_0 to f_rest_8 are present, it is L1.
    if (missingIndex > 8) {  // f_rest_0 to f_rest_8 are present
      shBands = 1;
    } else if (missingIndex > 23) {  // f_rest_0 to f_rest_23 are present
      shBands = 2;
    }
  }

  // Number of coefficients per color channel (R, G, B) for the SH bands being rotated (excluding L0)
  // Total L0-L3 coeffs: 1, 3, 8, 15 (per channel)
  // L0: 1 (L0 only)
  // L1: 3 (L1 only)
  // L2: 8 (L2 only)
  // L3: 15 (L3 only)

  // The coefficients to be rotated start from L1 (index 1) for a total of 3, 8, or 15 coeffs.
  // Total coeffs are 1 (L0) + 3*L1 + 5*L2 + 7*L3...
  // The `f_rest_i` columns hold the L1+ coefficients (total: 3* (n^2 - 1) coeffs)
  // The original JS uses 3, 8, 15 which corresponds to L1, L2, L3 bands *excluding L0*.
  // The array `shCoeffs` holds the coefficients *per color channel* (L1, L2, L3).
  const int shCoeffsPerChannel = (shBands == 1) ? 3 : (shBands == 2) ? 8 : (shBands == 3) ? 15 : 0;

  if (shBands > 0) {
    std::cout << "Applying SH rotation with " << shBands << " band(s) (" << shCoeffsPerChannel
              << " coeffs per channel)." << "\n";
  }

  // Temporary buffer for SH coefficients of one color channel
  std::vector<float> shCoeffs(shCoeffsPerChannel);

  // 3. Iterate and Transform Rows
  Row row;
  for (size_t i = 0; i < dataTable->getNumRows(); ++i) {
    // Use a temporary map to hold row data for read/write
    dataTable->getRow(i, row);

    // --- A. Translation (Position) ---
    if (hasTranslation) {
      Eigen::Vector3f pos(row["x"], row["y"], row["z"]);

      // Transform point: v' = M * v
      // Since Eigen's Mat4 * Vec3 (implicitly converting Vec3 to Vec4(v, 1))
      // already performs the correct projective transform (v' = T*R*S*v), we use that.
      Eigen::Vector4f pos4(pos.x(), pos.y(), pos.z(), 1.0f);
      pos4 = mat * pos4;

      row["x"] = pos4.x() / pos4.w();
      row["y"] = pos4.y() / pos4.w();
      row["z"] = pos4.z() / pos4.w();
    }

    // --- B. Rotation ---
    if (hasRotation) {
      // Note: The original code uses (rot_1, rot_2, rot_3, rot_0) -> (x, y, z, w) Playcanvas convention,
      // then multiplies by the global rotation 'r' (Quat).
      // Original: q.set(row.rot_1, row.rot_2, row.rot_3, row.rot_0).mul2(r, q);
      // Eigen stores Quat as (x, y, z, w) in memory/coefficients.
      Eigen::Quaternionf q_local(row["rot_0"],  // w
                                 row["rot_1"],  // x
                                 row["rot_2"],  // y
                                 row["rot_3"]   // z
      );

      // The combined rotation: q_global * q_local
      Eigen::Quaternionf q_combined = r * q_local;

      // Re-normalize might be a good idea, though quaternion multiplication should maintain unit length
      q_combined.normalize();

      // Store back using the original (w, x, y, z) column convention
      row["rot_0"] = q_combined.w();
      row["rot_1"] = q_combined.x();
      row["rot_2"] = q_combined.y();
      row["rot_3"] = q_combined.z();
    }

    // --- C. Scale ---
    if (hasScale) {
      // Scale is stored as log(scale). Scale transformation: exp(log(s_old)) * s_global = s_new.
      // log(s_new) = log(exp(log(s_old)) * s_global) = log(s_old) + log(s_global)
      float log_s = std::log(s);

      row["scale_0"] = row["scale_0"] + log_s;
      row["scale_1"] = row["scale_1"] + log_s;
      row["scale_2"] = row["scale_2"] + log_s;
    }

    // --- D. Spherical Harmonics (SH) Rotation ---
    if (shBands > 0) {
      // Iterate over the three color channels (R, G, B)
      for (int j = 0; j < 3; ++j) {
        // 1. Load SH coefficients for one channel (R, G, or B)
        for (int k = 0; k < shCoeffsPerChannel; ++k) {
          const std::string& colName = shNames[k + j * shCoeffsPerChannel];
          shCoeffs[k] = row[colName];
        }

        // 2. Apply rotation (RotateSH is assumed to operate on the float vector)
        rotateSH.apply(shCoeffs);

        // 3. Store rotated SH coefficients back to the row
        for (int k = 0; k < shCoeffsPerChannel; ++k) {
          const std::string& colName = shNames[k + j * shCoeffsPerChannel];
          row[colName] = shCoeffs[k];
        }
      }
    }

    // --- E. Final Update ---
    dataTable->setRow(i, row);
  }
}

}  // namespace splat
