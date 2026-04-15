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
