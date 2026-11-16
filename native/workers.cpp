#include "workers.h"
#include <algorithm>
#include <chrono>
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace spindle {
namespace {
// Idle time after which a worker stops spinning and sleeps until the next job.
constexpr auto spin_budget = std::chrono::microseconds(500);

inline void relax() {
#if defined(_MSC_VER)
    YieldProcessor();
#elif defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#elif defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("yield");
#endif
}
} // namespace

size_t default_threads() {
    size_t cores = std::max(1u, std::thread::hardware_concurrency());
#ifdef __APPLE__
    int performance = 0;
    size_t bytes = sizeof(performance);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &performance, &bytes, nullptr, 0) == 0 && performance > 0)
        cores = size_t(performance);
#endif
    return std::min(size_t(4), std::max(size_t(1), cores / 2));
}

ProjectionWorkers::ProjectionWorkers(size_t threads) {
    threads = std::min(threads, size_t(std::max(1u, std::thread::hardware_concurrency())));
    workers.reserve(threads > 0 ? threads - 1 : 0);
    try {
        for (size_t rank = 1; rank < threads; ++rank)
            workers.emplace_back([this] { work(); });
    } catch (...) {
        stop();
        throw;
    }
}
ProjectionWorkers::~ProjectionWorkers() { stop(); }

void ProjectionWorkers::wake() {
    // Pairs with the sleeper count a worker publishes before its final check.
    if (sleepers.load() == 0)
        return;
    { std::lock_guard<std::mutex> lock(mutex); }
    ready.notify_all();
}

void ProjectionWorkers::stop() {
    stopping.store(true);
    { std::lock_guard<std::mutex> lock(mutex); }
    ready.notify_all();
    for (auto &worker : workers)
        worker.join();
}

// Takes pieces of the current job until none are left.
void ProjectionWorkers::take() {
    for (;;) {
        const size_t index = next.fetch_add(1, std::memory_order_relaxed);
        if (index >= pieces)
            return;
        const size_t begin = index * piece;
        function(context, begin, std::min(count, begin + piece));
        finished.fetch_add(1, std::memory_order_release);
    }
}

void ProjectionWorkers::run(size_t rows, void *job, Function fn, size_t step) {
    if (workers.empty() || rows <= step) {
        fn(job, 0, rows);
        return;
    }
    // Close the previous job: after the odd value is visible no worker enters, and those
    // already inside leave at once because its pieces are used up.
    generation.fetch_add(1);
    while (inside.load())
        relax();
    const size_t team = workers.size() + 1, units = (rows + step - 1) / step;
    context = job;
    function = fn;
    count = rows;
    piece = step * ((units + team - 1) / team);
    pieces = (rows + piece - 1) / piece;
    std::fegetenv(&environment);
    next.store(0);
    finished.store(0);
    generation.fetch_add(1);
    wake();
    take();
    while (finished.load(std::memory_order_acquire) < pieces)
        relax();
}

void ProjectionWorkers::work() {
    uint64_t seen = 0;
    for (;;) {
        auto idle = std::chrono::steady_clock::now();
        auto open = [&] { const uint64_t g = generation.load(); return g != seen && !(g & 1); };
        for (size_t spins = 0; !open() && !stopping.load(std::memory_order_relaxed); ++spins) {
            relax();
            if ((spins & 1023) == 1023 && std::chrono::steady_clock::now() - idle > spin_budget) {
                std::unique_lock<std::mutex> lock(mutex);
                sleepers.fetch_add(1);
                ready.wait(lock, [&] { return stopping.load() || open(); });
                sleepers.fetch_sub(1);
                idle = std::chrono::steady_clock::now();
            }
        }
        if (stopping.load())
            return;
        inside.fetch_add(1);
        const uint64_t current = generation.load();
        if (current != seen && !(current & 1)) {
            seen = current;
            std::fenv_t saved;
            std::fegetenv(&saved);
            std::fesetenv(&environment);
            take();
            std::fesetenv(&saved);
        }
        inside.fetch_sub(1);
    }
}
} // namespace spindle
