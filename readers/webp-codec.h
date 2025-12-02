#pragma once

#include <webp/decode.h>
#include <webp/encode.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

struct DecodedImage {
  std::vector<uint8_t> rgba;
  int width;
  int height;
};

DecodedImage decodeRGBA(const uint8_t* data, size_t size) {
  int width, height;
  uint8_t* output = WebPDecodeRGBA(data, size, &width, &height);
  if (!output) {
    throw std::runtime_error("Failed to decode image");
  }

  DecodedImage info;
  info.width = width;
  info.height = height;
  info.rgba.assign(output, output + width * height * 4);
}

std::vector<uint8_t> encodeLosslessRGBA(const uint8_t* rgba, int width, int height, int stride = 0) {
  if (stride == 0) {
    stride = width * 4;
  }

  if (!rgba) {
    throw std::runtime_error("Failed to encode image");
  }

  if (width <= 0 || height <= 0) {
    throw std::runtime_error("Invalid image dimensions");
  }

  if (stride < width * 4) {
    throw std::runtime_error("Invalid stride");
  }

  WebPConfig config;
  if (!WebPConfigInit(&config)) {
    throw std::runtime_error("Failed to initialize WebPConfig");
  }

  config.lossless = 1;
  config.quality = 100;
  config.method = 6;

  if (!WebPValidateConfig(&config)) {
    throw std::runtime_error("Invalid WebPConfig");
  }

  uint8_t* output = nullptr;
  size_t output_size = WebPEncodeLosslessBGRA(rgba, width, height, stride, &output);
  if (output_size == 0 || !output) {
    throw std::runtime_error("Failed to encode image");
  }

  std::vector<uint8_t> result(output, output + output_size);
  WebPFree(output);

  return result;
}