#include <splat/splat.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Options {
  bool help = false;
  bool overwrite = false;
  int iterations = 10;
  size_t lodChunkCount = 64;
  size_t lodChunkExtent = 16;
  fs::path input;
  fs::path output;
  std::vector<double> levels;
};

bool startsWith(const std::string& value, const std::string& prefix) { return value.rfind(prefix, 0) == 0; }

bool matchesValueOption(const std::string& arg, const std::string& optionName) {
  return arg == optionName || startsWith(arg, optionName + "=");
}

bool endsWithIgnoreCase(const std::string& value, const std::string& suffix) {
  if (suffix.size() > value.size()) {
    return false;
  }

  return std::equal(suffix.rbegin(), suffix.rend(), value.rbegin(), [](char a, char b) {
    return std::tolower(static_cast<unsigned char>(a)) == std::tolower(static_cast<unsigned char>(b));
  });
}

std::string optionValue(const std::string& arg, int& index, int argc, char** argv, const std::string& optionName) {
  const size_t equals = arg.find('=');
  if (equals != std::string::npos) {
    return arg.substr(equals + 1);
  }
  if (index + 1 >= argc) {
    throw std::runtime_error(optionName + " requires a value");
  }
  return argv[++index];
}

int parseInt(const std::string& value, const std::string& optionName) {
  size_t consumed = 0;
  int result = 0;
  try {
    result = std::stoi(value, &consumed);
  } catch (const std::exception&) {
    throw std::runtime_error(optionName + " must be an integer");
  }
  if (consumed != value.size()) {
    throw std::runtime_error(optionName + " must be an integer");
  }
  if (result < 0) {
    throw std::runtime_error(optionName + " must be a non-negative integer");
  }
  return result;
}

double parsePercent(std::string value) {
  if (!value.empty() && value.back() == '%') {
    value.pop_back();
  }
  if (value.empty()) {
    throw std::runtime_error("--levels contains an empty percentage");
  }

  size_t consumed = 0;
  double result = 0.0;
  try {
    result = std::stod(value, &consumed);
  } catch (const std::exception&) {
    throw std::runtime_error("Invalid --levels percentage: " + value);
  }
  if (consumed != value.size() || result < 0.0 || result > 100.0) {
    throw std::runtime_error("--levels percentages must be between 0% and 100%");
  }
  return result;
}

void appendLevels(const std::string& value, std::vector<double>& levels) {
  std::stringstream stream(value);
  std::string part;
  while (std::getline(stream, part, ',')) {
    levels.push_back(parsePercent(part));
  }
}

Options parseArguments(int argc, char** argv) {
  Options options;
  std::vector<fs::path> positionals;

  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);

    if (arg == "--help") {
      options.help = true;
      continue;
    }
    if (arg == "--overwrite") {
      options.overwrite = true;
      continue;
    }
    if (matchesValueOption(arg, "--iterations")) {
      options.iterations = parseInt(optionValue(arg, i, argc, argv, "--iterations"), "--iterations");
      continue;
    }
    if (matchesValueOption(arg, "--lod-chunk-count")) {
      options.lodChunkCount =
          static_cast<size_t>(parseInt(optionValue(arg, i, argc, argv, "--lod-chunk-count"), "--lod-chunk-count"));
      continue;
    }
    if (matchesValueOption(arg, "--lod-chunk-extent")) {
      options.lodChunkExtent =
          static_cast<size_t>(parseInt(optionValue(arg, i, argc, argv, "--lod-chunk-extent"), "--lod-chunk-extent"));
      continue;
    }
    if (matchesValueOption(arg, "--levels")) {
      appendLevels(optionValue(arg, i, argc, argv, "--levels"), options.levels);
      while (i + 1 < argc && !startsWith(argv[i + 1], "--")) {
        appendLevels(argv[++i], options.levels);
      }
      continue;
    }
    if (startsWith(arg, "--")) {
      throw std::runtime_error("Unknown option: " + arg);
    }

    positionals.emplace_back(fs::u8path(arg));
  }

  if (options.help) {
    return options;
  }
  if (positionals.size() != 2) {
    throw std::runtime_error("Expected exactly one input file and one output lod-meta.json file");
  }
  if (options.levels.empty()) {
    throw std::runtime_error("--levels requires at least one percentage");
  }

  options.input = positionals[0];
  options.output = positionals[1];
  if (!endsWithIgnoreCase(options.output.u8string(), "lod-meta.json")) {
    throw std::runtime_error("Output file must end with lod-meta.json");
  }
  return options;
}

void printHelp() {
  std::cout << "Usage: PlaycanvasLOD [OPTIONS] <input-file> <output-lod-meta.json> --levels 100% 50% 25%\n\n";
  std::cout << "Options:\n";
  std::cout << "  --help                         Show this help and exit\n";
  std::cout << "  --overwrite                    Remove the output parent directory if it exists\n";
  std::cout << "  --iterations <n>               SOG SH compression iterations. Default: 10\n";
  std::cout << "  --lod-chunk-count <n>          Approximate Gaussians per LOD chunk in K. Default: 64\n";
  std::cout << "  --lod-chunk-extent <n>         Approximate LOD chunk extent in meters. Default: 16\n";
  std::cout << "  --levels 100% 50% 25%          LOD percentages, each between 0% and 100%\n";
  std::cout << "  --levels=100%,50%,25%          Inline comma-separated level form\n";
}

std::unique_ptr<splat::SplatCloud> readInput(const fs::path& filename) {
  const std::string u8 = filename.u8string();
  if (endsWithIgnoreCase(u8, ".ksplat")) {
    return splat::readKsplat(filename);
  }
  if (endsWithIgnoreCase(u8, ".splat")) {
    return splat::readSplat(filename);
  }
  if (endsWithIgnoreCase(u8, ".sog") || endsWithIgnoreCase(u8, "meta.json")) {
    return splat::readSog(filename, filename);
  }
  if (endsWithIgnoreCase(u8, ".ply")) {
    return splat::readPly(filename);
  }
  if (endsWithIgnoreCase(u8, ".spz")) {
    return splat::readSpz(filename);
  }
  if (endsWithIgnoreCase(u8, ".voxel.json")) {
    return splat::readVoxel(filename);
  }

  throw std::runtime_error("Unsupported input file type: " + u8);
}

bool isGaussianTable(const splat::SplatCloud& dataTable) {
  static const std::vector<std::string> requiredColumns = {"x",      "y",      "z",       "rot_0",   "rot_1",
                                                           "rot_2",  "rot_3",  "scale_0", "scale_1", "scale_2",
                                                           "f_dc_0", "f_dc_1", "f_dc_2",  "opacity"};

  return std::all_of(requiredColumns.begin(), requiredColumns.end(),
                     [&](const std::string& column) { return dataTable.hasColumn(column); });
}

int computeKeepCount(size_t sourceRows, double percent) {
  const double scaled = static_cast<double>(sourceRows) * percent / 100.0;
  const double rounded = std::round(scaled);
  if (rounded <= 0.0) {
    return 0;
  }
  if (rounded >= static_cast<double>(sourceRows)) {
    return static_cast<int>(sourceRows);
  }
  return static_cast<int>(rounded);
}

std::string formatPercent(double value) {
  std::ostringstream stream;
  const double rounded = std::round(value);
  if (std::abs(value - rounded) < 1e-9) {
    stream << static_cast<long long>(rounded);
  } else {
    stream.setf(std::ios::fixed);
    stream.precision(2);
    stream << value;
  }
  stream << '%';
  return stream.str();
}

std::string formatLevels(const std::vector<double>& levels) {
  std::ostringstream stream;
  for (size_t i = 0; i < levels.size(); ++i) {
    if (i > 0) {
      stream << ", ";
    }
    stream << formatPercent(levels[i]);
  }
  return stream.str();
}

void setLodColumn(splat::SplatCloud& dataTable, int levelIndex) {
  if (!dataTable.hasColumn("lod")) {
    dataTable.addColumn({"lod", std::vector<float>(dataTable.getNumRows())});
  }

  auto& lodValues = dataTable.getColumnByName("lod").asVector<float>();
  std::fill(lodValues.begin(), lodValues.end(), static_cast<float>(levelIndex));
}

void prepareOutput(const fs::path& output, bool overwrite) {
  const fs::path outputFile = fs::absolute(output);
  const fs::path outputDir = outputFile.parent_path();
  std::error_code ec;

  if (overwrite) {
    if (fs::exists(outputDir)) {
      fs::remove_all(outputDir, ec);
      if (ec) {
        throw std::runtime_error("Failed to remove output directory '" + outputDir.u8string() + "': " + ec.message());
      }
    }
    return;
  }

  if (fs::exists(outputDir)) {
    throw std::runtime_error("Output directory already exists: " + outputDir.u8string() + " (use --overwrite)");
  }
  if (fs::exists(outputFile)) {
    throw std::runtime_error("Output file already exists: " + outputFile.u8string() + " (use --overwrite)");
  }
}

int run(int argc, char** argv) {
  const Options options = parseArguments(argc, argv);
  if (options.help) {
    printHelp();
    return 0;
  }

  const std::string inputPath = fs::absolute(options.input).u8string();
  const std::string outputPath = fs::absolute(options.output).u8string();
  const std::string levelsText = formatLevels(options.levels);

  LOG_INFO("PlaycanvasLOD start");
  LOG_INFO("input='%s' output='%s' overwrite=%s iterations=%d lodChunkCount=%zu lodChunkExtent=%zu levels=[%s]",
           inputPath.c_str(), outputPath.c_str(), options.overwrite ? "true" : "false", options.iterations,
           options.lodChunkCount, options.lodChunkExtent, levelsText.c_str());

  prepareOutput(options.output, options.overwrite);
  LOG_INFO("output path prepared");

  LOG_INFO("reading input '%s'", inputPath.c_str());
  std::unique_ptr<splat::SplatCloud> source = readInput(options.input);
  if (!source || source->getNumRows() == 0 || !isGaussianTable(*source)) {
    throw std::runtime_error("Unsupported Gaussian data in file: " + options.input.u8string());
  }

  const size_t sourceRows = source->getNumRows();
  LOG_INFO("loaded source rows=%zu", sourceRows);

  std::vector<std::unique_ptr<splat::SplatCloud>> levels;
  levels.reserve(options.levels.size());

  for (size_t levelIndex = 0; levelIndex < options.levels.size(); ++levelIndex) {
    const double levelPercent = options.levels[levelIndex];
    const int keepCount = computeKeepCount(sourceRows, levelPercent);
    const std::string levelText = formatPercent(levelPercent);

    LOG_INFO("building lodIndex=%zu level=%s targetRows=%d", levelIndex, levelText.c_str(), keepCount);
    std::unique_ptr<splat::SplatCloud> level = splat::simplifyGaussians(*source, keepCount);
    setLodColumn(*level, static_cast<int>(levelIndex));
    LOG_INFO("lodIndex=%zu ready rows=%zu", levelIndex, level->getNumRows());
    levels.emplace_back(std::move(level));
  }

  LOG_INFO("combining %zu LOD tables", levels.size());
  std::unique_ptr<splat::SplatCloud> merged = splat::combine(levels);
  if (!merged || merged->getNumRows() == 0) {
    throw std::runtime_error("No splats to write");
  }
  LOG_INFO("combined rows=%zu", merged->getNumRows());

  LOG_INFO("writing LOD output to '%s'", outputPath.c_str());
  splat::writeLod(fs::absolute(options.output), merged.get(), nullptr, true, options.iterations, options.lodChunkCount,
                  options.lodChunkExtent);
  LOG_INFO("PlaycanvasLOD finished successfully");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
