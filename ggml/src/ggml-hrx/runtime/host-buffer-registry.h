#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

typedef struct hrx_buffer_s * hrx_buffer_t;

namespace ggml::hrx {

class HostBufferRef {
  public:
    HostBufferRef() = default;
    ~HostBufferRef();

    HostBufferRef(HostBufferRef && other) noexcept;
    HostBufferRef & operator=(HostBufferRef && other) noexcept;

    HostBufferRef(const HostBufferRef &)             = delete;
    HostBufferRef & operator=(const HostBufferRef &) = delete;

    bool valid() const { return buffer_ != nullptr; }

    hrx_buffer_t buffer() const { return buffer_; }

    size_t offset() const { return offset_; }
    size_t capacity() const { return capacity_; }
    bool direct_bindable() const { return direct_bindable_; }
    void * base() const { return reinterpret_cast<void *>(base_); }
    uint64_t identity() const { return identity_; }
    std::shared_ptr<std::atomic<uint64_t>> generation() const { return generation_; }

  private:
    hrx_buffer_t buffer_ = nullptr;
    size_t       offset_ = 0;
    size_t       capacity_ = 0;
    bool         direct_bindable_ = false;
    uintptr_t    base_ = 0;
    uint64_t     identity_ = 0;
    std::shared_ptr<std::atomic<uint64_t>> generation_;

    HostBufferRef(hrx_buffer_t buffer, size_t offset, size_t capacity, bool direct_bindable,
                  uintptr_t base, uint64_t identity, std::shared_ptr<std::atomic<uint64_t>> generation);
    friend class HostBufferRegistry;
};

// Tracks mapped HRX host buffers so pointer-based GGML transfers can remain stream ordered and handle based.
class HostBufferRegistry {
  public:
    void add(hrx_buffer_t buffer, void * base, size_t size, bool direct_bindable,
             uint64_t identity, std::shared_ptr<std::atomic<uint64_t>> generation);
    void remove(hrx_buffer_t buffer);

    HostBufferRef find(const void * data, size_t size) const;

  private:
    struct Entry {
        hrx_buffer_t buffer = nullptr;
        uintptr_t    base   = 0;
        size_t       size   = 0;
        bool         direct_bindable = false;
        uint64_t     identity = 0;
        std::shared_ptr<std::atomic<uint64_t>> generation;
    };

    mutable std::mutex mutex_;
    std::vector<Entry> entries_;
};

}  // namespace ggml::hrx
