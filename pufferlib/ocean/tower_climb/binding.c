#include "tower_climb.h"

#define Env CTowerClimb
#define MY_SHARED
#include "../env_binding.h"

static PyObject* my_shared(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char* path = "resources/tower_climb/maps.bin";
    int num_maps = 0;

    Level* levels = load_levels_from_file(&num_maps, path);
    if (levels == NULL) {
        PyErr_SetString(PyExc_IOError, "Failed to load maps from maps.bin. Did you run './tower_climb' to pregenerate them?");
        return NULL;
    }

    PuzzleState* puzzle_states = calloc(num_maps, sizeof(PuzzleState));

    for (int i = 0; i < num_maps; i++) {
        init_puzzle_state(&puzzle_states[i]);
        levelToPuzzleState(&levels[i], &puzzle_states[i]);
    }

    PyObject* levels_handle = PyLong_FromVoidPtr(levels);
    PyObject* puzzles_handle = PyLong_FromVoidPtr(puzzle_states);
    PyObject* num_maps_obj = PyLong_FromLong(num_maps);
    PyObject* state = PyDict_New();
    PyDict_SetItemString(state, "levels", levels_handle);
    PyDict_SetItemString(state, "puzzles", puzzles_handle);
    PyDict_SetItemString(state, "num_maps", num_maps_obj);
    return PyLong_FromVoidPtr(state);
}

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    // num_maps is loaded from the file, but we can still unpack it from kwargs
    // to know how many maps were intended to be loaded.
    // This is useful for consistency checks if needed later.
    // For now, the actual number of maps is determined by load_levels.
    // unpack(kwargs, "num_maps"); // No longer needed from kwargs

    env->reward_climb_row = unpack(kwargs, "reward_climb_row");
    env->reward_fall_row = unpack(kwargs, "reward_fall_row");
    env->reward_illegal_move = unpack(kwargs, "reward_illegal_move");
    env->reward_move_block = unpack(kwargs, "reward_move_block");
    init(env);

    PyObject* handle_obj = PyDict_GetItemString(kwargs, "state");
    if (handle_obj == NULL) {
        PyErr_SetString(PyExc_KeyError, "Key 'state' not found in kwargs");
        return 1;
    }

    // Check if handle_obj is a PyLong
    if (!PyLong_Check(handle_obj)) {
        PyErr_SetString(PyExc_TypeError, "state handle must be an integer");
        return 1;
    }

    // Convert PyLong to PyObject* (state dictionary)
    PyObject* state_dict = (PyObject*)PyLong_AsVoidPtr(handle_obj);
    if (state_dict == NULL) {
        PyErr_SetString(PyExc_ValueError, "Invalid state dictionary pointer");
        return 1;
    }

    // Verify it’s a dictionary
    if (!PyDict_Check(state_dict)) {
        PyErr_SetString(PyExc_TypeError, "State pointer does not point to a dictionary");
        return 1;
    }

    // Basic validation: check reference count
    if (state_dict->ob_refcnt <= 0) {
        PyErr_SetString(PyExc_RuntimeError, "State dictionary has invalid reference count");
        return 1;
    }

    PyObject* levels_obj = PyDict_GetItemString(state_dict, "levels");
    if (levels_obj == NULL) {
        PyErr_SetString(PyExc_KeyError, "Key 'levels' not found in state");
        return 1;
    }
    if (!PyLong_Check(levels_obj)) {
        PyErr_SetString(PyExc_TypeError, "levels must be an integer");
        return 1;
    }
    env->all_levels = (Level*)PyLong_AsVoidPtr(levels_obj);

    PyObject* num_maps_obj = PyDict_GetItemString(state_dict, "num_maps");
    if (num_maps_obj == NULL) {
        PyErr_SetString(PyExc_KeyError, "Key 'num_maps' not found in state");
        return 1;
    }
    if (!PyLong_Check(num_maps_obj)) {
        PyErr_SetString(PyExc_TypeError, "'num_maps' must be an integer");
        return 1;
    }
    if (env->all_levels != NULL) {
        env->num_maps = PyLong_AsLong(num_maps_obj);
    }

    PyObject* puzzles_obj = PyDict_GetItemString(state_dict, "puzzles");
    if (!PyObject_TypeCheck(puzzles_obj, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "puzzles handle must be an integer");
        return 1;
    }
    PuzzleState* puzzles = (PuzzleState*)PyLong_AsVoidPtr(puzzles_obj);
    if (!puzzles) {
        PyErr_SetString(PyExc_ValueError, "Invalid puzzles handle");
        return 1;
    }
    env->all_puzzles = puzzles;

    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    return 0;
}
