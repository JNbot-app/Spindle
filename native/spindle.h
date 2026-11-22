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
// threads is the worker team size (0: automatic); dense BLAS owns its scheduling.
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
// KV window in tokens (0: the archive's own). Changing it drops the cached request state.
NP_API int32_t np_set_window(np_handle *, size_t, char *, size_t);
// Keep the state after the first `count` prompt tokens. A later prompt starting with the same
// tokens skips their prefill; a cache hit and a miss give the same result.
NP_API int32_t np_set_prefix(np_handle *, size_t, char *, size_t);
// Decode skip: single-token decode steps skip the layers in `mask` (bit i = layer i; 0 = off, the
// default). Prompt tokens always run every layer. The archive's header fields 45-46 hold the list
// the model was trained to skip; any mask that keeps the last layer is accepted.
NP_API int32_t np_set_decode_skip(np_handle *, uint64_t, char *, size_t);
// Prompt tokens restored from the prefix cache by the last prefill.
NP_API size_t np_prefix_reused(np_handle *);
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

// Token choice for tool calling (grammar.cpp): grammar-constrained greedy decoding, copy guard and
// enum debiasing, shared by every frontend. Create one decoder per model with its vocabulary (the bytes
// of every token, `offsets` has count + 1 entries; special = 1 for control/unknown tokens), give it the
// tools (a JSON list of {name, parameters} with local $refs inlined, plus a JSON object mapping each
// enum label to its first token), then per request: start, and step with each new logits vector until
// the state is NP_DECODER_DONE. np_decoder_result writes {"complete", "calls", "probabilities"} as JSON
// and returns its length. Enum debiasing learns across requests, so keep one decoder per model.
typedef struct np_decoder np_decoder;
#define NP_DECODER_REQUIRE_CALL 1
#define NP_DECODER_REASONING 2
#define NP_DECODER_DEBIAS 4
#define NP_DECODER_DONE 3
NP_API np_decoder *np_decoder_new(const uint8_t *pieces, const uint32_t *offsets, const uint8_t *special,
                                  size_t count, int32_t open_id, int32_t close_id, int32_t think_id,
                                  const int32_t *stop_ids, size_t stop_count, char *, size_t);
NP_API void np_decoder_free(np_decoder *);
NP_API int32_t np_decoder_set_tools(np_decoder *, const char *tools_json, const char *labels_json,
                                    char *, size_t);
NP_API int32_t np_decoder_start(np_decoder *, const char *query, size_t query_size, int32_t flags,
                                float copy_margin, char *, size_t);
// Picks, records and takes the next token; returns it, or -1 on error.
NP_API int32_t np_decoder_step(np_decoder *, const float *logits, size_t count, char *, size_t);
// 0 before the calls (or in a note), 1 in the calls, 2 after them, 3 done.
NP_API int32_t np_decoder_state(np_decoder *);
NP_API int64_t np_decoder_result(np_decoder *, char *buffer, size_t size);
#ifdef __cplusplus
}
#endif
#endif
