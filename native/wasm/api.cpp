// Browser entry points. The model file is written into the in-memory file system by JavaScript;
// this wrapper only exists because np_load takes 64-bit arguments, which JavaScript passes badly.
#include "../spindle.h"
#include <emscripten.h>

extern "C" EMSCRIPTEN_KEEPALIVE np_handle *sp_load(const char *path, char *error, int capacity) {
    return np_load(path, 0, 0, 1, error, size_t(capacity));
}

