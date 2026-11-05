// Standalone Vulkan fragment output fixture. No game data or renderer code is
// used.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
#define VK(x)                                                                  \
  do {                                                                         \
    VkResult r_ = (x);                                                         \
    if (r_ != VK_SUCCESS) {                                                    \
      fprintf(stderr, "%s: %d\n", #x, (int)r_);                                \
      exit(2);                                                                 \
    }                                                                          \
  } while (0)
#define CHECK(x)                                                               \
  do {                                                                         \
    if (!(x)) {                                                                \
      fprintf(stderr, "check failed: %s at %d\n", #x, __LINE__);               \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
static int errors;
static VKAPI_ATTR VkBool32 VKAPI_CALL debug_cb(
    VkDebugUtilsMessageSeverityFlagBitsEXT s, VkDebugUtilsMessageTypeFlagsEXT t,
    const VkDebugUtilsMessengerCallbackDataEXT *d, void *u) {
  (void)t;
  (void)u;
  if (s & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
    errors++;
  fprintf(stderr, "validation %s: %s\n",
          s & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT ? "error"
                                                            : "warning",
          d->pMessage);
  return VK_FALSE;
}
typedef struct {
  VkBuffer b;
  VkDeviceMemory m;
  void *ptr;
} Buffer;
typedef struct {
  VkImage i;
  VkDeviceMemory m;
  VkImageView v;
} Image;
static VkDevice dev;
static VkPhysicalDeviceMemoryProperties mem;
static uint32_t memtype(uint32_t bits, VkMemoryPropertyFlags flags) {
  for (uint32_t i = 0; i < mem.memoryTypeCount; i++)
    if ((bits & (1u << i)) &&
        (mem.memoryTypes[i].propertyFlags & flags) == flags)
      return i;
  fprintf(stderr, "memory type unavailable\n");
  exit(2);
}
static Buffer buffer(VkDeviceSize size, VkBufferUsageFlags usage) {
  Buffer x = {0};
  VkBufferCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                           .size = size,
                           .usage = usage};
  VK(vkCreateBuffer(dev, &ci, 0, &x.b));
  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(dev, x.b, &mr);
  VkMemoryAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr.size,
      .memoryTypeIndex =
          memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
  VK(vkAllocateMemory(dev, &ai, 0, &x.m));
  VK(vkBindBufferMemory(dev, x.b, x.m, 0));
  VK(vkMapMemory(dev, x.m, 0, size, 0, &x.ptr));
  memset(x.ptr, 0, (size_t)size);
  return x;
}
static Image image(void) {
  Image x = {0};
  VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                          .imageType = VK_IMAGE_TYPE_2D,
                          .format = VK_FORMAT_R32G32B32A32_SFLOAT,
                          .extent = {8, 1, 1},
                          .mipLevels = 1,
                          .arrayLayers = 1,
                          .samples = VK_SAMPLE_COUNT_1_BIT,
                          .tiling = VK_IMAGE_TILING_OPTIMAL,
                          .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                   VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
  VK(vkCreateImage(dev, &ci, 0, &x.i));
  VkMemoryRequirements mr;
  vkGetImageMemoryRequirements(dev, x.i, &mr);
  VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                             .allocationSize = mr.size,
                             .memoryTypeIndex = memtype(mr.memoryTypeBits, 0)};
  VK(vkAllocateMemory(dev, &ai, 0, &x.m));
  VK(vkBindImageMemory(dev, x.i, x.m, 0));
  VkImageViewCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = x.i,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = ci.format,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
  VK(vkCreateImageView(dev, &vi, 0, &x.v));
  return x;
}
static VkShaderModule shader(const char *path) {
  FILE *f = fopen(path, "rb");
  CHECK(f);
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  CHECK(n > 0 && !(n % 4));
  rewind(f);
  uint32_t *w = malloc((size_t)n);
  CHECK(w && fread(w, 1, (size_t)n, f) == (size_t)n);
  fclose(f);
  VkShaderModuleCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = (size_t)n,
      .pCode = w};
  VkShaderModule m;
  VK(vkCreateShaderModule(dev, &ci, 0, &m));
  free(w);
  return m;
}
static VkPipeline pipeline(VkPipelineLayout l, VkShaderModule vs,
                           VkShaderModule fs) {
  VkPipelineShaderStageCreateInfo st[2] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_VERTEX_BIT,
       .module = vs,
       .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
       .module = fs,
       .pName = "main"}};
  VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
  VkViewport vp = {0, 0, 8, 1, 0, 1};
  VkRect2D sc = {{0, 0}, {8, 1}};
  VkPipelineViewportStateCreateInfo vps = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .pViewports = &vp,
      .scissorCount = 1,
      .pScissors = &sc};
  VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_NONE,
      .lineWidth = 1};
  VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
  VkPipelineColorBlendAttachmentState ba[2] = {{.colorWriteMask = 15},
                                               {.colorWriteMask = 15}};
  VkPipelineColorBlendStateCreateInfo bs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 2,
      .pAttachments = ba};
  VkDynamicState dyn[] = {VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo ds = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 1,
      .pDynamicStates = dyn};
  VkFormat fmt[2] = {VK_FORMAT_R32G32B32A32_SFLOAT,
                     VK_FORMAT_R32G32B32A32_SFLOAT};
  VkPipelineRenderingCreateInfo ri = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 2,
      .pColorAttachmentFormats = fmt};
  VkGraphicsPipelineCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &ri,
      .stageCount = 2,
      .pStages = st,
      .pVertexInputState = &vi,
      .pInputAssemblyState = &ia,
      .pViewportState = &vps,
      .pRasterizationState = &rs,
      .pMultisampleState = &ms,
      .pColorBlendState = &bs,
      .pDynamicState = &ds,
      .layout = l};
  VkPipeline p;
  VK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &ci, 0, &p));
  return p;
}
typedef struct {
  uint32_t draw, cap, pad[2], a[4], b[4];
} Push;
int main(int argc, char **argv) {
  CHECK(argc == 6);
  int set = atoi(argv[5]);
  CHECK(set == 0 || set == 1);
  const char *layer = "VK_LAYER_KHRONOS_validation",
             *ext = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
  VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                           .pApplicationName = "fragment-nonfinite-fixture",
                           .apiVersion = VK_API_VERSION_1_3};
  VkDebugUtilsMessengerCreateInfoEXT dbg = {
      .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
      .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT,
      .messageType = 7,
      .pfnUserCallback = debug_cb};
  VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                              .pNext = &dbg,
                              .pApplicationInfo = &app,
                              .enabledLayerCount = 1,
                              .ppEnabledLayerNames = &layer,
                              .enabledExtensionCount = 1,
                              .ppEnabledExtensionNames = &ext};
  VkInstance inst;
  VK(vkCreateInstance(&ici, 0, &inst));
  PFN_vkCreateDebugUtilsMessengerEXT mk =
      (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
          inst, "vkCreateDebugUtilsMessengerEXT");
  VkDebugUtilsMessengerEXT messenger;
  CHECK(mk);
  VK(mk(inst, &dbg, 0, &messenger));
  uint32_t n = 0;
  VK(vkEnumeratePhysicalDevices(inst, &n, 0));
  CHECK(n && n <= 32);
  VkPhysicalDevice phys[32], pd = 0;
  VK(vkEnumeratePhysicalDevices(inst, &n, phys));
  for (uint32_t i = 0; i < n; i++) {
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(phys[i], &p);
    if (strstr(p.deviceName, "RTX 5060")) {
      pd = phys[i];
      printf("device=%s driver=%u\n", p.deviceName, p.driverVersion);
      break;
    }
  }
  CHECK(pd);
  VkPhysicalDeviceFeatures feat;
  vkGetPhysicalDeviceFeatures(pd, &feat);
  CHECK(feat.fragmentStoresAndAtomics);
  uint32_t nq = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, 0);
  CHECK(nq <= 64);
  VkQueueFamilyProperties qp[64];
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qp);
  uint32_t q = 0;
  while (q < nq && !(qp[q].queueFlags & VK_QUEUE_GRAPHICS_BIT))
    q++;
  CHECK(q < nq);
  float priority = 1;
  VkDeviceQueueCreateInfo qi = {.sType =
                                    VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                .queueFamilyIndex = q,
                                .queueCount = 1,
                                .pQueuePriorities = &priority};
  VkPhysicalDeviceVulkan13Features f13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .dynamicRendering = VK_TRUE};
  VkPhysicalDeviceFeatures enabled = {.fragmentStoresAndAtomics = VK_TRUE};
  VkDeviceCreateInfo di = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                           .pNext = &f13,
                           .queueCreateInfoCount = 1,
                           .pQueueCreateInfos = &qi,
                           .pEnabledFeatures = &enabled};
  VK(vkCreateDevice(pd, &di, 0, &dev));
  VkQueue queue;
  vkGetDeviceQueue(dev, q, 0, &queue);
  vkGetPhysicalDeviceMemoryProperties(pd, &mem);
  Buffer report = buffer(8 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
         readback = buffer(2 * 2 * 8 * 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  uint32_t *flags = report.ptr;
  flags[6] = 0xdeadbeef;
  flags[7] = 0xcafef00d;
  VkDescriptorSetLayout empty, report_layout;
  VkDescriptorSetLayoutCreateInfo eli = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  VK(vkCreateDescriptorSetLayout(dev, &eli, 0, &empty));
  VkDescriptorSetLayoutBinding binding = {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT};
  VkDescriptorSetLayoutCreateInfo rli = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1,
      .pBindings = &binding};
  VK(vkCreateDescriptorSetLayout(dev, &rli, 0, &report_layout));
  VkDescriptorSetLayout sets[2] = {set ? empty : report_layout, report_layout};
  VkPushConstantRange range = {.stageFlags = VK_SHADER_STAGE_VERTEX_BIT |
                                             VK_SHADER_STAGE_FRAGMENT_BIT,
                               .offset = 0,
                               .size = sizeof(Push)};
  VkPipelineLayoutCreateInfo pli = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = (uint32_t)set + 1,
      .pSetLayouts = sets,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &range};
  VkPipelineLayout layout;
  VK(vkCreatePipelineLayout(dev, &pli, 0, &layout));
  VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
  VkDescriptorPoolCreateInfo pci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1,
      .poolSizeCount = 1,
      .pPoolSizes = &ps};
  VkDescriptorPool pool;
  VK(vkCreateDescriptorPool(dev, &pci, 0, &pool));
  VkDescriptorSetAllocateInfo dai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = pool,
      .descriptorSetCount = 1,
      .pSetLayouts = &report_layout};
  VkDescriptorSet desc;
  VK(vkAllocateDescriptorSets(dev, &dai, &desc));
  VkDescriptorBufferInfo db = {report.b, 0, 8 * 4};
  VkWriteDescriptorSet wr = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             .dstSet = desc,
                             .dstBinding = 0,
                             .descriptorCount = 1,
                             .descriptorType =
                                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             .pBufferInfo = &db};
  vkUpdateDescriptorSets(dev, 1, &wr, 0, 0);
  VkShaderModule vs = shader(argv[1]), plain = shader(argv[2]),
                 instrumented = shader(argv[3]),
                 instrumented2 = shader(argv[4]);
  VkPipeline pipes[3] = {pipeline(layout, vs, plain),
                         pipeline(layout, vs, instrumented),
                         pipeline(layout, vs, instrumented2)};
  Image imgs[4] = {image(), image(), image(), image()};
  VkCommandPoolCreateInfo cpi = {.sType =
                                     VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                 .queueFamilyIndex = q};
  VkCommandPool cp;
  VK(vkCreateCommandPool(dev, &cpi, 0, &cp));
  VkCommandBufferAllocateInfo cai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = cp,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1};
  VkCommandBuffer cb;
  VK(vkAllocateCommandBuffers(dev, &cai, &cb));
  VkCommandBufferBeginInfo bi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  VK(vkBeginCommandBuffer(cb, &bi));
  for (int phase = 0; phase < 2; phase++) {
    VkImageMemoryBarrier ib[2];
    VkRenderingAttachmentInfo att[2];
    for (int a = 0; a < 2; a++) {
      Image *im = &imgs[phase * 2 + a];
      ib[a] = (VkImageMemoryBarrier){
          .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
          .srcAccessMask = 0,
          .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
          .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
          .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .image = im->i,
          .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
      att[a] = (VkRenderingAttachmentInfo){
          .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
          .imageView = im->v,
          .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
          .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
          .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
    }
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, 0,
                         0, 0, 2, ib);
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                          .renderArea = {{0, 0}, {8, 1}},
                          .layerCount = 1,
                          .colorAttachmentCount = 2,
                          .pColorAttachments = att};
    vkCmdBeginRendering(cb, &ri);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipes[phase]);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, layout,
                            (uint32_t)set, 1, &desc, 0, 0);
    for (uint32_t d = 0; d < 7; d++) {
      if (phase)
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          pipes[d % 2 ? 2 : 1]);
      Push p = {.draw = d,
                .cap = 6,
                .a = {0x3f000000, 0x3e800000, 0x3f400000, 0x3f800000},
                .b = {0x3e000000, 0x3f000000, 0x3e800000, 0x3f800000}};
      if (d == 1)
        p.a[0] = 0x7fc00000;
      if (d == 2)
        p.a[1] = 0x7f800000;
      if (d == 3)
        p.b[2] = 0xff800000;
      if (d == 4)
        p.b[0] = 0x7fc00000;
      if (d == 5)
        p.b[0] = 0x7fc00000;
      if (d == 6)
        p.a[0] = 0x7fc00000;
      VkRect2D sc = {{(int)d, 0}, {1, 1}};
      vkCmdSetScissor(cb, 0, 1, &sc);
      vkCmdPushConstants(
          cb, layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
          0, sizeof p, &p);
      vkCmdDraw(cb, 3, 1, 0, 0);
    }
    vkCmdEndRendering(cb);
    for (int a = 0; a < 2; a++) {
      ib[a].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      ib[a].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      ib[a].oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      ib[a].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    }
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, 0, 0, 0, 2, ib);
    for (int a = 0; a < 2; a++) {
      VkBufferImageCopy cr = {
          .bufferOffset = (VkDeviceSize)(phase * 2 + a) * 8 * 16,
          .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
          .imageExtent = {8, 1, 1}};
      vkCmdCopyImageToBuffer(cb, imgs[phase * 2 + a].i,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback.b,
                             1, &cr);
    }
  }
  VkMemoryBarrier host = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                          .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT |
                                           VK_ACCESS_TRANSFER_WRITE_BIT,
                          .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
  vkCmdPipelineBarrier(cb,
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                           VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, 0, 0, 0);
  VK(vkEndCommandBuffer(cb));
  VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                     .commandBufferCount = 1,
                     .pCommandBuffers = &cb};
  VK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
  VK(vkQueueWaitIdle(queue));
  uint32_t expected[8] = {0, 1, 2, 8, 0, 4, 0xdeadbeef, 0xcafef00d};
  for (int i = 0; i < 8; i++) {
    printf("report[%d]=%#x expected=%#x\n", i, flags[i], expected[i]);
    CHECK(flags[i] == expected[i]);
  }
  float *pixel = readback.ptr;
  for (int a = 0; a < 2; a++)
    for (int c = 0; c < 4; c++) {
      float expected_color[2][4] = {{.5f, .25f, .75f, 1},
                                    {.125f, .5f, .25f, 1}};
      float x = pixel[a * 8 * 4 + c], y = pixel[(2 + a) * 8 * 4 + c];
      printf("finite attachment=%d channel=%d original=%g diagnostic=%g\n", a,
             c, x, y);
      CHECK(x == y && x == expected_color[a][c]);
    }
  CHECK(!errors);
  printf("PASS set=%d validation_errors=%d\n", set, errors);
  for (int i = 0; i < 3; i++)
    vkDestroyPipeline(dev, pipes[i], 0);
  vkDestroyShaderModule(dev, vs, 0);
  vkDestroyShaderModule(dev, plain, 0);
  vkDestroyShaderModule(dev, instrumented, 0);
  vkDestroyShaderModule(dev, instrumented2, 0);
  vkDestroyCommandPool(dev, cp, 0);
  vkDestroyDescriptorPool(dev, pool, 0);
  vkDestroyPipelineLayout(dev, layout, 0);
  vkDestroyDescriptorSetLayout(dev, report_layout, 0);
  vkDestroyDescriptorSetLayout(dev, empty, 0);
  for (int i = 0; i < 4; i++) {
    vkDestroyImageView(dev, imgs[i].v, 0);
    vkDestroyImage(dev, imgs[i].i, 0);
    vkFreeMemory(dev, imgs[i].m, 0);
  }
  vkUnmapMemory(dev, report.m);
  vkDestroyBuffer(dev, report.b, 0);
  vkFreeMemory(dev, report.m, 0);
  vkUnmapMemory(dev, readback.m);
  vkDestroyBuffer(dev, readback.b, 0);
  vkFreeMemory(dev, readback.m, 0);
  vkDestroyDevice(dev, 0);
  PFN_vkDestroyDebugUtilsMessengerEXT destroy =
      (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
          inst, "vkDestroyDebugUtilsMessengerEXT");
  destroy(inst, messenger, 0);
  vkDestroyInstance(inst, 0);
  return 0;
}
