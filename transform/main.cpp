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
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "gpudevice.h"
#include "options.h"
#include "process.h"

namespace fs = std::filesystem;

using namespace splat;

extern void writeFile(const std::filesystem::path& filename, DataTable* dataTable, DataTable* envDataTable,
                      const Options& options);
extern std::vector<std::unique_ptr<DataTable>> readFile(const std::filesystem::path& filename, const Options& options,
                                                        const std::vector<Param>& params);
extern std::string getOutputFormat(const std::filesystem::path& filename);

struct File {
  std::filesystem::path filename;
  std::vector<ProcessAction> processActions;
};

ABSL_FLAG(bool, overwrite, false, "Overwrite output file if it exists");
ABSL_FLAG(bool, help, false, "Show help and exit");
ABSL_FLAG(bool, version, false, "Show version and exit");
ABSL_FLAG(bool, quiet, false, "Suppress non-error output");
ABSL_FLAG(bool, list_gpus, false, "List available GPU adapters and exit");
ABSL_FLAG(bool, unbundled, false, "Generate unbundled HTML viewer with separate files");

ABSL_FLAG(int32_t, iterations, 10, "Iterations for SOG SH compression (more=better)");
ABSL_FLAG(int32_t, lod_chunk_count, 64, "Approximate number of Gaussians per LOD chunk in K");
ABSL_FLAG(int32_t, lod_chunk_extent, 16, "Approximate size of an LOD chunk in world units (m)");

ABSL_FLAG(std::string, gpu, "-1", "Select device for SOG compression: GPU adapter index | 'cpu'");
ABSL_FLAG(std::string, lod_select, "", "Comma-separated LOD levels to read from LCC input");
ABSL_FLAG(std::string, viewer_settings, "", "HTML viewer settings JSON file");

static std::tuple<std::vector<File>, Options> parseArguments(int argc, char** argv) {
  auto parseInteger = [](absl::string_view value) {
    int result;
    if (!absl::SimpleAtoi(value, &result)) {
      throw std::runtime_error("Invalid number value: " + std::string(value));
    }
    return result;
  };

  auto parseNonNegativeInteger = [&](absl::string_view value, absl::string_view optionName) {
    int result = parseInteger(value);
    if (result < 0) {
      throw std::runtime_error(std::string(optionName) + " must be a non-negative integer");
    }
    return result;
  };

  auto stripOptionPrefix = [](absl::string_view arg) {
    while (absl::ConsumePrefix(&arg, "-")) {
    }
    return arg;
  };

  auto optionName = [&](absl::string_view arg) {
    absl::string_view name = stripOptionPrefix(arg);
    const size_t equals = name.find('=');
    if (equals != absl::string_view::npos) {
      name = name.substr(0, equals);
    }
    return name;
  };

  auto normalizedOptionName = [&](absl::string_view arg) {
    std::string name(optionName(arg));
    for (char& c : name) {
      if (c == '-') {
        c = '_';
      }
    }
    return name;
  };

  auto optionValue = [&](absl::string_view arg, int& i, absl::string_view optionName) {
    const size_t equals = arg.find('=');
    if (equals != absl::string_view::npos) {
      return arg.substr(equals + 1);
    }
    if (i + 1 >= argc) {
      throw std::runtime_error(std::string(optionName) + " requires a value");
    }
    return absl::string_view(argv[++i]);
  };

  auto parseParams = [&](absl::string_view value, File& file) {
    for (absl::string_view param : absl::StrSplit(value, ',', absl::SkipEmpty())) {
      const size_t equals = param.find('=');
      if (equals == absl::string_view::npos || equals == 0) {
        throw std::runtime_error("--params values must be key=value pairs");
      }
      file.processActions.push_back(Param{std::string(param.substr(0, equals)), std::string(param.substr(equals + 1))});
    }
  };

  auto parseDecimate = [&](absl::string_view value) {
    if (absl::ConsumeSuffix(&value, "%")) {
      float percent = -1.0f;
      if (!absl::SimpleAtof(value, &percent) || percent < 0.0f || percent > 100.0f) {
        throw std::runtime_error("--decimate percentage must be between 0 and 100");
      }
      return Decimate{-1, percent};
    }

    return Decimate{parseNonNegativeInteger(value, "--decimate"), -1.0f};
  };

  absl::SetProgramUsageMessage(
      "Transform and Filter Gaussian Splats\nUSAGE: SplatTransform [GLOBAL] input [ACTIONS] ... output [ACTIONS]");

  const std::unordered_set<std::string> globalValueOptions = {
      "iterations", "lod_chunk_count", "lod_chunk_extent", "gpu", "lod_select", "viewer_settings"};
  const std::unordered_set<std::string> globalBoolOptions = {"overwrite", "help",      "version",
                                                             "quiet",     "list_gpus", "unbundled"};

  std::vector<File> files;
  std::vector<char*> sanitizedArgv;
  sanitizedArgv.push_back(argv[0]);

  for (int i = 1; i < argc; ++i) {
    absl::string_view arg(argv[i]);
    if (!absl::StartsWith(arg, "-") || arg == "-") {
      files.push_back({std::filesystem::u8path(std::string(arg)), {}});
      continue;
    }

    const std::string normalizedName = normalizedOptionName(arg);
    const bool hasInlineValue = arg.find('=') != absl::string_view::npos;

    if (normalizedName == "params" || normalizedName == "lod" || normalizedName == "decimate") {
      if (files.empty()) {
        throw std::runtime_error("--" + normalizedName + " must follow a positional file");
      }

      File& current = files.back();
      if (normalizedName == "params") {
        parseParams(optionValue(arg, i, "--params"), current);
      } else if (normalizedName == "lod") {
        current.processActions.push_back(Lod{parseNonNegativeInteger(optionValue(arg, i, "--lod"), "--lod")});
      } else {
        current.processActions.push_back(parseDecimate(optionValue(arg, i, "--decimate")));
      }
      continue;
    }

    sanitizedArgv.push_back(argv[i]);
    const bool isGlobalValueOption = globalValueOptions.find(normalizedName) != globalValueOptions.end();
    const bool isGlobalBoolOption = globalBoolOptions.find(normalizedName) != globalBoolOptions.end();

    if (!hasInlineValue && isGlobalValueOption) {
      if (i + 1 >= argc) {
        throw std::runtime_error("--" + normalizedName + " requires a value");
      }
      sanitizedArgv.push_back(argv[++i]);
    } else if (!isGlobalBoolOption && !isGlobalValueOption) {
      // Let Abseil preserve its normal unknown-flag diagnostics for options outside this translated subset.
    }
  }

  absl::ParseCommandLine(static_cast<int>(sanitizedArgv.size()), sanitizedArgv.data());

  Options options;
  options.overwrite = absl::GetFlag(FLAGS_overwrite);
  options.help = absl::GetFlag(FLAGS_help);
  options.version = absl::GetFlag(FLAGS_version);
  options.quiet = absl::GetFlag(FLAGS_quiet);
  options.listGpus = absl::GetFlag(FLAGS_list_gpus);
  options.unbundled = absl::GetFlag(FLAGS_unbundled);
  options.viewerSettingsPath = std::filesystem::path(absl::GetFlag(FLAGS_viewer_settings));
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

  if (!options.help && !options.version && !options.listGpus && files.size() < 2) {
    throw std::runtime_error("Expected at least one input file and one output file");
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

int main(int argc, char** argv) {
  std::chrono::time_point startTime = std::chrono::high_resolution_clock::now();

  std::vector<File> files;
  Options options;
  try {
    std::tie(files, options) = parseArguments(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }

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
    std::cout << "  --decimate <n|n%>            Keep n splats or n percent of splats\n";
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
  std::string outputFormat = getOutputFormat(outputFilename);

  if (options.overwrite) {
    std::error_code ec;
    if (outputFormat == "lod") {
      // LOD format: remove the entire parent directory
      if (fs::exists(outputFilename.parent_path())) {
        fs::remove_all(outputFilename.parent_path(), ec);
        if (ec) {
          LOG_ERROR("Failed to remove directory: %s", ec.message().c_str());
          std::exit(1);
        }
      }
    } else {
      // Other formats: remove the single file
      if (fs::exists(outputFilename)) {
        fs::remove(outputFilename, ec);
        if (ec) {
          LOG_ERROR("Failed to remove file: %s", ec.message().c_str());
          std::exit(1);
        }
      }
    }
    // Ensure output directory exists
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
      for (const auto& action : inputArg.processActions) {
        if (const auto* param = std::get_if<Param>(&action)) {
          params.push_back(*param);
        }
      }

      std::vector<std::unique_ptr<DataTable>> dts = readFile(inputArg.filename, options, params);

      for (auto&& dt : dts) {
        if (dt->getNumRows() == 0 || !isGSDataTable(dt.get())) {
          throw std::runtime_error("Unsupported data in file: " + inputArg.filename.u8string());
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

    writeFile(outputFilename, dataTable.get(), envDataTable ? envDataTable.get() : nullptr, options);

  } catch (const std::exception& e) {
    LOG_ERROR("%s", e.what());
    std::exit(1);
  }

  auto endTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = endTime - startTime;
  LOG_INFO("done in %.6fs", elapsed.count());

  std::exit(0);
}
