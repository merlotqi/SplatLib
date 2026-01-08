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

#pragma once

#include <string>
#include <vector>

struct AdapterInfo {
  int index;
  std::string name;
};

#ifdef _WIN32
#include <Windows.h>
#include <dxgi.h>
#pragma comment(lib, "dxgi.lib")

std::vector<AdapterInfo> enumerateAdapters() {
  std::vector<AdapterInfo> adapters;
  IDXGIFactory* pFactory = nullptr;
  if (FAILED(CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)&pFactory))) {
    return adapters;
  }

  IDXGIAdapter* pAdapter = nullptr;
  for (UINT i = 0; pFactory->EnumAdapters(i, &pAdapter) != DXGI_ERROR_NOT_FOUND; ++i) {
    DXGI_ADAPTER_DESC desc;
    pAdapter->GetDesc(&desc);

    std::wstring ws(desc.Description);
    std::string name(ws.begin(), ws.end());

    adapters.push_back({(int)i, name});
    pAdapter->Release();
  }
  pFactory->Release();
  return adapters;
}

#else

#include <vulkan/vulkan.h>

static std::vector<AdapterInfo> enumerateAdapters() {
  std::vector<AdapterInfo> adapters;

  VkInstanceCreateInfo createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;

  VkInstance instance;
  if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
    return adapters;
  }

  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

  if (deviceCount > 0) {
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    for (uint32_t i = 0; i < deviceCount; ++i) {
      VkPhysicalDeviceProperties deviceProperties;
      vkGetPhysicalDeviceProperties(devices[i], &deviceProperties);

      adapters.push_back({(int)i, std::string(deviceProperties.deviceName)});
    }
  }

  vkDestroyInstance(instance, nullptr);
  return adapters;
}

#endif
