/***********************************************************************************
 *
 * splat - A C++ library for reading and writing 3D Gaussian Splatting (splat) files.
 *
 * This library provides functionality to convert, manipulate, and process
 * 3D Gaussian splatting data formats used in real-time neural rendering.
 *
 * This file is part of splat.
 *
 * splat is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * splat is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 * For more information, visit the project's homepage or contact the author.
 *
 ***********************************************************************************/

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/flags/usage.h>
#include <absl/strings/match.h>
#include <absl/strings/numbers.h>
#include <absl/strings/str_split.h>
#include <absl/strings/strip.h>
#include <splat/splat.h>

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "gpudevice.h"
#include "options.h"
#include "process.h"

namespace fs = std::filesystem;

using namespace splat;

extern void writeFile(const std::string& filename, DataTable* dataTable, DataTable* envDataTable,
                      const Options& options);
extern std::vector<std::unique_ptr<DataTable>> readFile(const std::string& filename, const Options& options,
                                                        const std::vector<Param>& params);
extern std::string getOutputFormat(std::string filename);

struct File {
  std::string filename;
  std::vector<ProcessAction> processActions;
};

ABSL_FLAG(bool, overwrite, false, "Overwrite output file if it exists");
ABSL_FLAG(bool, help, false, "Show help and exit");
ABSL_FLAG(bool, version, false, "Show version and exit");
ABSL_FLAG(bool, quiet, false, "Suppress non-error output");
ABSL_FLAG(bool, list_gpus, false, "List available GPU adapters and exit");
ABSL_FLAG(bool, unbundled, false, "Generate unbundled HTML viewer with separate files");

ABSL_FLAG(int32_t, iterations, 10, "Iterations for SOG SH compression (more=better)");
ABSL_FLAG(int32_t, lod_chunk_count, 512, "Approximate number of Gaussians per LOD chunk in K");
ABSL_FLAG(int32_t, lod_chunk_extent, 16, "Approximate size of an LOD chunk in world units (m)");

ABSL_FLAG(std::string, gpu, "-1", "Select device for SOG compression: GPU adapter index | 'cpu'");
ABSL_FLAG(std::string, lod_select, "", "Comma-separated LOD levels to read from LCC input");
ABSL_FLAG(std::string, viewer_settings, "", "HTML viewer settings JSON file");

ABSL_FLAG(float, lod, {}, "Specify the level of detail, n >= 0. Can be repeated");

static std::tuple<std::vector<File>, Options> parseArguments(int argc, char** argv) {
  auto parseInteger = [](absl::string_view value) {
    int result;
    if (!absl::SimpleAtoi(value, &result)) {
      throw std::runtime_error("Invalid number value" + std::string(value));
    }
    return result;
  };

  absl::SetProgramUsageMessage(
      "Transform and Filter Gaussian Splats\nUSAGE: SplatTransform [GLOBAL] input [ACTIONS] ... output [ACTIONS]");

  std::vector<char*> remaining_args = absl::ParseCommandLine(argc, argv);

  Options options;
  options.overwrite = absl::GetFlag(FLAGS_overwrite);
  options.quiet = absl::GetFlag(FLAGS_quiet);
  options.listGpus = absl::GetFlag(FLAGS_list_gpus);
  options.unbundled = absl::GetFlag(FLAGS_unbundled);
  options.viewerSettingsPath = absl::GetFlag(FLAGS_viewer_settings);
  options.iterations = absl::GetFlag(FLAGS_iterations);
  options.lodChunkCount = absl::GetFlag(FLAGS_lod_chunk_count);
  options.lodChunkExtent = absl::GetFlag(FLAGS_lod_chunk_extent);

  // Parse gpu option - can be a number or "cpu"
  std::string gpu_val = absl::GetFlag(FLAGS_gpu);
  if (gpu_val == "cpu") {
    options.device = -2;
  } else {
    options.device = parseInteger(gpu_val);
  }

  std::string lod_s = absl::GetFlag(FLAGS_lod_select);
  if (!lod_s.empty()) {
    for (auto s : absl::StrSplit(lod_s, ',', absl::SkipEmpty())) {
      options.lodSelect.push_back(parseInteger(s));
    }
  }

  std::vector<File> files;
  for (size_t i = 1; i < remaining_args.size(); ++i) {
    absl::string_view arg = remaining_args[i];

    if (!absl::StartsWith(arg, "-")) {
      files.push_back({std::string(arg), {}});
    } else if (!files.empty()) {
      File& current = files.back();

      absl::string_view name = arg;
      while (absl::ConsumePrefix(&name, "-")) {
      }
    }
  }

  return {files, options};
}

static bool isGSDataTable(const DataTable* dataTable) {
  static std::vector<std::string> required_columns = {"x",      "y",      "z",       "rot_0",   "rot_1",
                                                      "rot_2",  "rot_3",  "scale_0", "scale_1", "scale_2",
                                                      "f_dc_0", "f_dc_1", "f_dc_2",  "opacity"};

  bool gs = std::all_of(required_columns.begin(), required_columns.end(),
                        [&](const std::string& c) { return dataTable->hasColumn(c); });
  return gs;
}

static std::unique_ptr<DataTable> combine(std::vector<std::unique_ptr<DataTable>>& dataTables) {
  if (dataTables.empty()) {
    return nullptr;
  }
  if (dataTables.size() == 1) {
    return std::move(dataTables.at(0));
  }

  return nullptr;
}

int main(int argc, char** argv) {
  std::chrono::time_point startTime = std::chrono::high_resolution_clock::now();

  auto [files, options] = parseArguments(argc, argv);
  if (absl::GetFlag(FLAGS_help)) {
    std::cout << "Usage: SplatTransform [OPTIONS] input_file output_file\n\n";
    std::cout << "GLOBAL OPTIONS:\n";
    std::cout << "  --help                       Show this help and exit\n";
    std::cout << "  --version                    Show version and exit\n";
    std::cout << "  --quiet                      Suppress non-error output\n";
    std::cout << "  --overwrite                  Overwrite output file if it exists\n";
    std::cout << "  --iterations <n>             Iterations for SOG SH compression (more=better). Default: 10\n";
    std::cout << "  --list-gpus                  List available GPU adapters and exit\n";
    std::cout << "  --gpu <n|cpu>                Select device for SOG compression: GPU adapter index | 'cpu'\n";
    std::cout << "  --viewer-settings <file>     HTML viewer settings JSON file\n";
    std::cout << "  --unbundled                  Generate unbundled HTML viewer with separate files\n";
    std::cout << "  --lod-select <n,n,...>       Comma-separated LOD levels to read from LCC input\n";
    std::cout << "  --lod-chunk-count <n>        Approximate number of Gaussians per LOD chunk in K. Default: 512\n";
    std::cout << "  --lod-chunk-extent <n>       Approximate size of an LOD chunk in world units (m). Default: 16\n";
    std::cout << "\nFILE ACTIONS (can be specified between files):\n";
    std::cout << "  --lod <n>                    Specify the level of detail, n >= 0\n";
    std::cout << "  --params <key=value,...>     Additional parameters\n";
    return 0;
  }

  if (absl::GetFlag(FLAGS_version)) {
    std::cout << "SplatTransform Version " << splat::version << "\n";
    return 0;
  }

  Logger::instance().setQuiet(options.quiet);
  LOG_INFO("SplatTransform v%s", splat::version);

  // show version and exit
  if (options.version) {
    exit(0);
  }

  if (options.listGpus) {
    LOG_INFO("Enumerating available GPU adapters...\n");
    try {
      std::vector<AdapterInfo> adapters = enumerateAdapters();

      if (adapters.empty()) {
        LOG_INFO("No GPU adapters found.");
        LOG_INFO("This could mean:");
        LOG_INFO("  - Graphics drivers need to be updated");
        LOG_INFO("  - Your system does not support the required graphics API");
      } else {
        for (const auto& adapter : adapters) {
          LOG_INFO("[%d] %s", adapter.index, adapter.name.c_str());
        }
        LOG_INFO("\nUse -g <index> to select a specific GPU adapter.");
      }
    } catch (const std::exception& err) {
      LOG_ERROR("Failed to enumerate GPU adapters: %s", err.what());
    }
    std::exit(0);
  }

  std::vector<File> inputArgs(files.begin(), files.end() - 1);
  File outputArg = files.back();

  fs::path outputFilename = fs::absolute(outputArg.filename);
  std::string outputFormat = getOutputFormat(outputFilename.string());

  if (options.overwrite) {
    std::error_code ec;
    fs::create_directories(outputFilename.parent_path(), ec);
    if (ec) {
      LOG_ERROR("Failed to create directory: %s", ec.message().c_str());
      std::exit(1);
    }
  } else {
    if (fs::exists(outputFilename)) {
      LOG_ERROR("File '%s' already exists. Use -w option to overwrite.", outputFilename.string().c_str());
      std::exit(1);
    }
  }

  try {
    std::vector<std::unique_ptr<DataTable>> inputDataTables;

    for (const auto& inputArg : inputArgs) {
      std::vector<Param> params;

      std::vector<std::unique_ptr<DataTable>> dts = readFile(inputArg.filename, options, params);

      for (auto&& dt : dts) {
        if (dt->getNumRows() == 0 || !isGSDataTable(dt.get())) {
          throw std::runtime_error("Unsupported data in file: " + inputArg.filename);
        }

        dt = processDataTable(dt.release(), inputArg.processActions);
        inputDataTables.emplace_back(dt.release());
      }
    }

    std::vector<std::unique_ptr<DataTable>> envDataTables;
    std::vector<std::unique_ptr<DataTable>> nonEnvDataTables;

    // special-case the environment dataTable
    for (auto&& dt : inputDataTables) {
      if (dt->hasColumn("lod") && dt->getColumnByName("lod").every<float>(-1.0f))
        envDataTables.emplace_back(dt.release());
      if (!dt->hasColumn("lod") || (dt->hasColumn("lod") && dt->getColumnByName("lod").some<float>(-1.0f)))
        nonEnvDataTables.emplace_back(dt.release());
    }

    // combine inputs into a single output dataTable
    std::unique_ptr<DataTable> dataTable;
    if (!nonEnvDataTables.empty()) {
      dataTable.reset(processDataTable(combine(nonEnvDataTables).release(), outputArg.processActions).release());
    }

    if (!dataTable || dataTable->getNumRows() == 0) {
      throw std::runtime_error("No splats to write");
    }

    std::unique_ptr<DataTable> envDataTable;
    if (!envDataTables.empty()) {
      envDataTable = processDataTable(combine(envDataTables).release(), outputArg.processActions);
    }

    LOG_INFO("Loaded %llu gaussians", (unsigned long long)dataTable->getNumRows());

    writeFile(outputFilename.string(), dataTable.release(), envDataTable ? envDataTable.release() : nullptr, options);

  } catch (const std::exception& e) {
    LOG_ERROR("%s", e.what());
    std::exit(1);
  }

  auto endTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = endTime - startTime;
  LOG_INFO("done in %.6fs", elapsed.count());

  std::exit(0);
}
