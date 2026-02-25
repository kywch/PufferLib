#include "g2048.h"

#define Env Game
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->scaffolding_ratio = unpack(kwargs, "scaffolding_ratio");
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "merge_score", log->merge_score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "lifetime_max_tile", log->lifetime_max_tile);
    assign_to_dict(dict, "reached_16384", log->reached_16384);
    assign_to_dict(dict, "reached_32768", log->reached_32768);
    assign_to_dict(dict, "reached_65536", log->reached_65536);
    assign_to_dict(dict, "reached_131072", log->reached_131072);
    return 0;
}
