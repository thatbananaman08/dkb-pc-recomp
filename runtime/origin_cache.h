// A lock-free direct-mapped cache from a lowered code buffer's address to its stage name (nvn2_lower.cpp nvn2_origin).
// Lowered buffers live for the process and their addresses never name another stage, so an entry never goes stale; a miss
// (and a name of 0, which is not cached) takes the locked map. A reader accepts an entry only if its address word was the
// same before and after it read the name, so a writer replacing the entry at that moment cannot give it another stage's name.
#pragma once
#include <atomic>
#include <cstdint>

struct OriginCache {
    static constexpr unsigned kSlots = 1024;
    struct Slot {
        std::atomic<const void *> code{nullptr};
        std::atomic<uint64_t> origin{0};
    };
    Slot slot[kSlots];

    static unsigned index(const void *code) {
        return static_cast<unsigned>((reinterpret_cast<uintptr_t>(code) >> 4) * 0x9E3779B97F4A7C15ull >> 54) & (kSlots - 1);
    }
    bool get(const void *code, uint64_t *origin) const {
        const Slot &s = slot[index(code)];
        if (s.code.load(std::memory_order_acquire) != code) return false;
        uint64_t o = s.origin.load(std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_acquire);
        if (s.code.load(std::memory_order_relaxed) != code) return false;
        *origin = o;
        return true;
    }
    void put(const void *code, uint64_t origin) {
        Slot &s = slot[index(code)];
        s.code.store(nullptr, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
        s.origin.store(origin, std::memory_order_relaxed);
        s.code.store(code, std::memory_order_release);
    }
};
