#include "g2048.h"

#define Env Game
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->scaffolding_ratio = unpack(kwargs, "scaffolding_ratio");
    env->scaffolding_ratio = max(min(env->scaffolding_ratio, 0.9f), 0.0f);
    env->snake_reward_weight = unpack(kwargs, "snake_reward_weight");
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "merge_score", log->merge_score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "snake_reward", log->snake_reward);
    return 0;
}