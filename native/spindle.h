#ifndef SPINDLE_H
#define SPINDLE_H
#include <stddef.h>
#include <stdint.h>
#if defined(_WIN32)
#if defined(SPINDLE_STATIC)
#define NP_API
#elif defined(SPINDLE_BUILD)
#define NP_API __declspec(dllexport)
#else
#define NP_API __declspec(dllimport)
#endif
#else
#define NP_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif
typedef struct np_handle np_handle;
typedef int32_t (*np_trace_callback)(void *, int32_t, uint32_t, const char *, int32_t,
                                     const float *, size_t);
NP_API uint32_t np_abi_version(void);
// threads controls packed projection workers; dense BLAS owns its scheduling.
// C callers must finish all operations before freeing a handle.
NP_API np_handle *np_load(const char *, uint64_t, uint64_t, int32_t, char *, size_t);
// Clone shares immutable mapped weights; request state and scratch are independent.
NP_API np_handle *np_clone(np_handle *, char *, size_t);
NP_API void np_free(np_handle *);
NP_API void np_reset(np_handle *);
NP_API int32_t np_tensor_info(np_handle *, int32_t, uint32_t *, uint32_t *, char *, size_t);
NP_API int32_t np_row(np_handle *, int32_t, uint32_t, float *, size_t, char *, size_t);
NP_API int32_t np_matvec(np_handle *, int32_t, const float *, size_t, float *, size_t, char *,
                         size_t);
// Stateless full prefill. Does not alter the cached request.
NP_API int32_t np_forward(np_handle *, const int32_t *, size_t, float *, size_t, float *, size_t,
                          char *, size_t);
NP_API int32_t np_forward_trace(np_handle *, const int32_t *, size_t, float *, size_t,
                                np_trace_callback, void *, char *, size_t);
// Cached prefill or continuation. np_decode requires exactly one token.
NP_API int32_t np_eval(np_handle *, const int32_t *, size_t, float *, size_t, char *, size_t);
NP_API int32_t np_decode(np_handle *, int32_t, float *, size_t, char *, size_t);
NP_API int32_t np_engram(np_handle *, const int32_t *, size_t, float *, size_t, char *, size_t);
NP_API int32_t np_replay_norm(np_handle *, int32_t, const float *, size_t, float *, size_t, char *,
                              size_t);
NP_API int32_t np_replay_projection(np_handle *, int32_t, const float *, size_t, float *, size_t,
                                    char *, size_t);
NP_API int32_t np_replay_convolution(np_handle *, int32_t, const float *, size_t, size_t, float *,
                                     size_t, char *, size_t);
NP_API int32_t np_replay_rope(np_handle *, uint32_t, const float *, size_t, float *, size_t, char *,
                              size_t);
NP_API int32_t np_replay_a8(const float *, size_t, float *, size_t, char *, size_t);
// Greedy selection with a caller-supplied grammar mask. Ties choose lowest ID.
NP_API int32_t np_select_token(const float *, const uint8_t *, size_t);
NP_API size_t np_decoded_buffer_bytes(np_handle *);
// Allocated numerical-buffer capacities, excluding allocator/handle overhead.
// Mapped weights are shared by clones; the other fields are per request.
typedef struct np_memory_info {
    uint64_t mapped_weights_bytes;
    uint64_t decoded_workspace_bytes;
    uint64_t kv_bytes;
    uint64_t scratch_bytes;
} np_memory_info;
NP_API int32_t np_memory_stats(np_handle *, np_memory_info *, char *, size_t);
NP_API void np_profile_enable(int32_t);
NP_API void np_profile_reset(void);
NP_API int32_t np_profile_dump(char *, size_t);
NP_API void np_profile_print(void);
// Final normalized hidden state of the last evaluated token.
NP_API int32_t np_last_hidden(np_handle *, float *, size_t, char *, size_t);
// Scores the cached prompt and completion using the archive's confidence head.
// Returns 1 with a score, 0 when no head is present, or -1 on error.
NP_API int32_t np_confidence(np_handle *, float *, char *, size_t);

#ifdef __cplusplus
}
#endif
#endif
