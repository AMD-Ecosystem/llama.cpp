#include "host-buffer-registry.h"

#include "hrx_runtime.h"

#include <algorithm>
#include <utility>

namespace ggml::hrx {

HostBufferRef::HostBufferRef(hrx_buffer_t buffer, size_t offset, size_t capacity, bool direct_bindable,
                             uintptr_t base, uint64_t identity,
                             std::shared_ptr<std::atomic<uint64_t>> generation) :
    buffer_(buffer), offset_(offset), capacity_(capacity), direct_bindable_(direct_bindable),
    base_(base), identity_(identity), generation_(std::move(generation)) {}

HostBufferRef::~HostBufferRef() {
    if (buffer_ != nullptr) {
        hrx_buffer_release(buffer_);
    }
}

HostBufferRef::HostBufferRef(HostBufferRef && other) noexcept :
    buffer_(std::exchange(other.buffer_, nullptr)),
    offset_(other.offset_),
    capacity_(other.capacity_),
    direct_bindable_(other.direct_bindable_), base_(other.base_), identity_(other.identity_),
    generation_(std::move(other.generation_)) {
    other.offset_ = 0;
    other.capacity_ = 0;
    other.direct_bindable_ = false;
}

HostBufferRef & HostBufferRef::operator=(HostBufferRef && other) noexcept {
    if (this != &other) {
        if (buffer_ != nullptr) {
            hrx_buffer_release(buffer_);
        }
        buffer_       = std::exchange(other.buffer_, nullptr);
        offset_       = other.offset_;
        capacity_     = other.capacity_;
        direct_bindable_ = other.direct_bindable_;
        base_ = other.base_;
        identity_ = other.identity_;
        generation_ = std::move(other.generation_);
        other.offset_ = 0;
        other.capacity_ = 0;
        other.direct_bindable_ = false;
    }
    return *this;
}

void HostBufferRegistry::add(hrx_buffer_t buffer, void * base, size_t size, bool direct_bindable,
                             uint64_t identity, std::shared_ptr<std::atomic<uint64_t>> generation) {
    if (buffer == nullptr || base == nullptr || size == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.push_back({ buffer, reinterpret_cast<uintptr_t>(base), size, direct_bindable,
                         identity, std::move(generation) });
}

void HostBufferRegistry::remove(hrx_buffer_t buffer) {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.erase(std::remove_if(entries_.begin(), entries_.end(),
                                  [buffer](const Entry & entry) { return entry.buffer == buffer; }),
                   entries_.end());
}

HostBufferRef HostBufferRegistry::find(const void * data, size_t size) const {
    if (data == nullptr) {
        return {};
    }
    const uintptr_t             address = reinterpret_cast<uintptr_t>(data);
    std::lock_guard<std::mutex> lock(mutex_);
    for (const Entry & entry : entries_) {
        if (address < entry.base) {
            continue;
        }
        const size_t offset = static_cast<size_t>(address - entry.base);
        if (offset > entry.size || size > entry.size - offset) {
            continue;
        }
        hrx_buffer_retain(entry.buffer);
        return HostBufferRef(entry.buffer, offset, entry.size, entry.direct_bindable,
                             entry.base, entry.identity, entry.generation);
    }
    return {};
}

}  // namespace ggml::hrx
