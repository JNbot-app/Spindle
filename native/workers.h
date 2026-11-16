#pragma once
#include <atomic>
#include <cfenv>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

namespace spindle {
// Synchronous, allocation-free row jobs. Kernels must not throw or submit nested jobs.
// Decode projections take tens of microseconds, less than a condition-variable
// wake-up, so idle workers spin briefly before they sleep.
//
// A job is split into one piece per thread and the pieces are claimed from a shared counter.
// A thread that is slow to start (asleep, on a busy core) then finds its piece already taken:
// the caller never waits for a worker that has not begun. Each row is computed on its own, so
// results do not depend on which thread took which piece.
class ProjectionWorkers {
  public:
    using Function = void (*)(void *, size_t, size_t);
    explicit ProjectionWorkers(size_t threads);
    ~ProjectionWorkers();
    ProjectionWorkers(const ProjectionWorkers &) = delete;
    ProjectionWorkers &operator=(const ProjectionWorkers &) = delete;
    // Pieces are contiguous row ranges whose bounds are multiples of grain.
    void run(size_t rows, void *context, Function function, size_t grain = 4);

  private:
    std::vector<std::thread> workers;
    std::mutex mutex;
    std::condition_variable ready;
    std::atomic<bool> stopping{false};
    // Each counter sits on its own cache line: workers poll `generation` while others write.
    // Odd while the caller writes the next job; workers only enter on an even value.
    alignas(128) std::atomic<uint64_t> generation{0};
    alignas(128) std::atomic<size_t> inside{0};
    alignas(128) std::atomic<size_t> next{0};
    alignas(128) std::atomic<size_t> finished{0};
    alignas(128) std::atomic<size_t> sleepers{0};
    size_t count = 0, piece = 4, pieces = 0;
    void *context = nullptr;
    Function function = nullptr;
    std::fenv_t environment;
    void work();
    void take();
    void wake();
    void stop();
};
// Team size for "auto" (0): half the performance cores, at most four. Every job waits for its
// slowest thread, so a team as large as the machine slows down as soon as another program
// needs a core.
size_t default_threads();
// body(i) for every i in [0, count): on the team when there is more than one piece of `grain`
// items, otherwise on the caller. The body must only write data that belongs to its own i.
template <class Body>
void parallel_for(ProjectionWorkers *workers, size_t count, size_t grain, Body &&body) {
    using Plain = typename std::remove_reference<Body>::type;
    auto call = [](void *context, size_t begin, size_t end) {
        auto &function = *static_cast<Plain *>(context);
        for (size_t i = begin; i < end; ++i)
            function(i);
    };
    if (workers && count > grain)
        workers->run(count, const_cast<void *>(static_cast<const void *>(&body)), call, grain);
    else
        call(const_cast<void *>(static_cast<const void *>(&body)), 0, count);
}
} // namespace spindle
