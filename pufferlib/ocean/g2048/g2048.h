#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "raylib.h"

static inline int min(int a, int b) { return a < b ? a : b; }
static inline int max(int a, int b) { return a > b ? a : b; }

#define SIZE 4
#define EMPTY 0
#define UP 1
#define DOWN 2
#define LEFT 3
#define RIGHT 4
#define BASE_MAX_TICKS 1000

// Precomputed constants
#define REWARD_MULTIPLIER 0.0625f
#define INVALID_MOVE_PENALTY -0.05f
#define GAME_OVER_PENALTY -1.0f

// Features: 18 per cell
// 1. Normalized tile value (current_val / max_val)
// 2. One-hot for empty (1 if empty, 0 if occupied)
// 3-18. One-hot for tile values 2^1 to 2^16 (16 features)
#define NUM_FEATURES 18

// To normalize perf from 0 to 1. Only used with perf.
#define OBSERVED_MAX_TILE 16384.0f

typedef struct {
    float perf;
    float score;
    float merge_score;
    float episode_return;
    float episode_length;
    float n;
} Log;

typedef struct {
    Log log;                        // Required
    unsigned char* observations;    // Cheaper in memory if encoded in uint_8
    int* actions;                   // Required
    float* rewards;                 // Required
    unsigned char* terminals;       // Required

    float scaffolding_ratio;        // The ratio for "scaffolding" runs, in which higher blocks are spawned
    bool is_scaffolding_episode;

    int score;
    int tick;
    unsigned char grid[SIZE][SIZE];
    float episode_reward;           // Accumulate episode reward
    int moves_made;
    int max_episode_ticks;          // Dynamic max_ticks based on score
    
    // Cached values to avoid recomputation
    int empty_count;
    bool game_over_cached;
    bool grid_changed;
} Game;

// Precomputed color table for rendering optimization
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};

static Color tile_colors[12] = {
    {6, 24, 24, 255}, // Empty/background
    {187, 187, 187, 255}, // 2
    {170, 187, 187, 255}, // 4
    {150, 187, 187, 255}, // 8
    {130, 187, 187, 255},  // 16
    {110, 187, 187, 255},  // 32
    {90, 187, 187, 255},   // 64
    {70, 187, 187, 255}, // 128
    {50, 187, 187, 255},  // 256
    {30, 187, 187, 255},  // 512
    {10, 187, 187, 255},  // 1024
    {0, 187, 187, 255}   // 2048+
};

// --- Logging ---
void add_log(Game* game);

// --- Required functions for env_binding.h ---
void c_reset(Game* env);
void c_step(Game* env);
void c_render(Game* env);
void c_close(Game* env);

static inline unsigned char get_max_tile(Game* game) {
    unsigned char max_tile = 0;
    // Unroll loop for better performance
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (game->grid[i][j] > max_tile) {
                max_tile = game->grid[i][j];
            }
        }
    }
    return max_tile;
}

// Inline function for updating observations (avoid function call overhead)
static inline void update_observations(Game* game) {
    // Observation: 4x4 grid, 18 features per cell
    // 1. Normalized tile value (current_val / max_val)
    // 2. One-hot for empty (1 if empty, 0 if occupied)
    // 3. One-hot for tile values 2^1 to 2^16 (16 features)
    
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            int base_idx = (i * SIZE + j) * NUM_FEATURES;
            unsigned char grid_val = game->grid[i][j];

            // Feature 1: The original tile values ** 1.5, to make a bit superlinear within uint8
            game->observations[base_idx] = (unsigned char)pow((float)grid_val, 1.5f);

            // Feature 2: One-hot for empty
            game->observations[base_idx + 1] = (grid_val == EMPTY) ? 1 : 0;

            // Features 3-18: One-hot for tile values
            // NOTE: If this ever gets close to 131072, revisit this
            memset(&game->observations[base_idx + 2], 0, 16 * sizeof(char));
            if (grid_val > 0) {
                grid_val = min(grid_val, 16);
                game->observations[base_idx + 1 + grid_val] = 1;
            }
        }
    }
}

// Cache empty cell count during grid operations
static inline void update_empty_count(Game* game) {
    int count = 0;
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (game->grid[i][j] == EMPTY) count++;
        }
    }
    game->empty_count = count;
}

void add_log(Game* game) {
    // Scaffolding runs will distort stats, so skip logging
    if (game->is_scaffolding_episode) return;

    unsigned char s = get_max_tile(game);
    game->log.score += (float)(1 << s);
    game->log.perf += (float)(1 << s) / OBSERVED_MAX_TILE;
    game->log.merge_score += (float)game->score;
    game->log.episode_length += game->tick;
    game->log.episode_return += game->episode_reward;
    game->log.n += 1;
}

void c_reset(Game* game) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            game->grid[i][j] = EMPTY;
        }
    }

    game->score = 0;
    game->tick = 0;
    game->episode_reward = 0;
    game->empty_count = SIZE * SIZE;
    game->game_over_cached = false;
    game->grid_changed = true;
    game->moves_made = 0;
    game->max_episode_ticks = BASE_MAX_TICKS;
    
    if (game->terminals) game->terminals[0] = 0;

    // Higher tiles are spawned in scaffolding episodes
    // game->is_scaffolding_episode = (rand() / (float)RAND_MAX) < game->scaffolding_ratio;
    // NOTE: scaffolding did not work well, so not using it. Leaving it as a reference.
    game->is_scaffolding_episode = false;

    // Add two random tiles at the start - optimized version
    for (int added = 0; added < 2; ) {
        int pos = rand() % (SIZE * SIZE);
        int i = pos / SIZE;
        int j = pos % SIZE;
        if (game->grid[i][j] == EMPTY) {
            game->grid[i][j] = (rand() % 10 == 0) ? 2 : 1;
            added++;
            game->empty_count--;
        }
    }
    
    update_observations(game);
}

void add_random_tile(Game* game) {
    if (game->empty_count == 0) return;
    
    // Use reservoir sampling for better performance
    int chosen_pos = -1;
    int count = 0;
    
    for (int pos = 0; pos < SIZE * SIZE; pos++) {
        int i = pos / SIZE;
        int j = pos % SIZE;
        if (game->grid[i][j] == EMPTY) {
            count++;
            if (rand() % count == 0) {
                chosen_pos = pos;
            }
        }
    }
    
    if (chosen_pos >= 0) {
        int i = chosen_pos / SIZE;
        int j = chosen_pos % SIZE;

        unsigned char new_tile = 0;
        if (game->is_scaffolding_episode) {
            int max_tile = (int)get_max_tile(game);
            // Scaffolding: spawn tiles up to max tile (or 2^17...)
            new_tile = min(17, (rand() % max(1, max_tile)) + 1);
        } else {
            // Normal: Implement the 90% 2, 10% 4 rule
            new_tile = (rand() % 10 == 0) ? 2 : 1;
        }

        game->grid[i][j] = new_tile;
        game->empty_count--;
        game->grid_changed = true;
    }
}

// Optimized slide and merge with fewer memory operations
static inline bool slide_and_merge(unsigned char* row, float* reward, float* score_increase) {
    bool moved = false;
    int write_pos = 0;
    
    // Single pass: slide and identify merge candidates
    for (int read_pos = 0; read_pos < SIZE; read_pos++) {
        if (row[read_pos] != EMPTY) {
            if (write_pos != read_pos) {
                row[write_pos] = row[read_pos];
                row[read_pos] = EMPTY;
                moved = true;
            }
            write_pos++;
        }
    }
    
    // Merge pass
    for (int i = 0; i < SIZE - 1; i++) {
        if (row[i] != EMPTY && row[i] == row[i + 1]) {
            row[i]++;
            *reward += ((float)row[i]) * REWARD_MULTIPLIER;
            *score_increase += (float)(1 << (int)row[i]);
            // Shift remaining elements left
            for (int j = i + 1; j < SIZE - 1; j++) {
                row[j] = row[j + 1];
            }
            row[SIZE - 1] = EMPTY;
            moved = true;
        }
    }
    
    return moved;
}

bool move(Game* game, int direction, float* reward, float* score_increase) {
    bool moved = false;
    unsigned char temp[SIZE];
    
    if (direction == UP || direction == DOWN) {
        for (int col = 0; col < SIZE; col++) {
            // Extract column
            for (int i = 0; i < SIZE; i++) {
                int idx = (direction == UP) ? i : SIZE - 1 - i;
                temp[i] = game->grid[idx][col];
            }
            
            if (slide_and_merge(temp, reward, score_increase)) {
                moved = true;
                // Write back column
                for (int i = 0; i < SIZE; i++) {
                    int idx = (direction == UP) ? i : SIZE - 1 - i;
                    game->grid[idx][col] = temp[i];
                }
            }
        }
    } else {
        for (int row = 0; row < SIZE; row++) {
            // Extract row
            for (int i = 0; i < SIZE; i++) {
                int idx = (direction == LEFT) ? i : SIZE - 1 - i;
                temp[i] = game->grid[row][idx];
            }
            
            if (slide_and_merge(temp, reward, score_increase)) {
                moved = true;
                // Write back row
                for (int i = 0; i < SIZE; i++) {
                    int idx = (direction == LEFT) ? i : SIZE - 1 - i;
                    game->grid[row][idx] = temp[i];
                }
            }
        }
    }

    if (moved) {
        game->grid_changed = true;
        game->game_over_cached = false; // Invalidate cache
    }

    return moved;
}

bool is_game_over(Game* game) {
    // Use cached result if grid hasn't changed
    if (!game->grid_changed && game->game_over_cached) {
        return game->game_over_cached;
    }
    
    // Quick check: if there are empty cells, game is not over
    if (game->empty_count > 0) {
        game->game_over_cached = false;
        game->grid_changed = false;
        return false;
    }
    
    // Check for possible merges
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            unsigned char current = game->grid[i][j];
            if (i < SIZE - 1 && current == game->grid[i + 1][j]) {
                game->game_over_cached = false;
                game->grid_changed = false;
                return false;
            }
            if (j < SIZE - 1 && current == game->grid[i][j + 1]) {
                game->game_over_cached = false;
                game->grid_changed = false;
                return false;
            }
        }
    }
    
    game->game_over_cached = true;
    game->grid_changed = false;
    return true;
}

void c_step(Game* game) {
    float reward = 0.0f;
    float score_add = 0.0f;
    bool did_move = move(game, game->actions[0] + 1, &reward, &score_add);
    game->tick++;

    if (did_move) {
        game->moves_made++;
        add_random_tile(game);
        game->score += score_add;
        update_empty_count(game); // Update after adding tile
        update_observations(game); // Observations only change if the grid changes

        if (!game->is_scaffolding_episode) {
            // This is to limit infinite invalid moves during eval
            // Don't need to be tight. Don't need to show to user?
            game->max_episode_ticks = max(BASE_MAX_TICKS, game->score / 10);
        }

    } else {
        reward = INVALID_MOVE_PENALTY;
        // No need to update observations if the grid hasn't changed
    }

    bool game_over = is_game_over(game);
    bool max_ticks_reached = game->tick >= game->max_episode_ticks;
    game->terminals[0] = (game_over || max_ticks_reached) ? 1 : 0;

    // Game over penalty overrides other rewards
    if (game_over) {
        reward = GAME_OVER_PENALTY;
    }

    game->rewards[0] = reward;
    game->episode_reward += reward;

    if (game->terminals[0]) {
        add_log(game);
        c_reset(game);
    }
}

// Stepping for eval only, no reward, no reset
void step_without_reset(Game* game) {
    float score_add = 0.0f;
    float reward = 0.0f;
    bool did_move = move(game, game->actions[0] + 1, &reward, &score_add);
    game->tick++;

    if (did_move) {
        game->moves_made++;
        add_random_tile(game);
        game->score += score_add;
        update_empty_count(game); // Update after adding tile
        update_observations(game); // Observations only change if the grid changes
    }

    bool game_over = is_game_over(game);
    game->terminals[0] = (game_over) ? 1 : 0;
}


// Rendering optimizations
void c_render(Game* game) {
    static bool window_initialized = false;
    static char score_text[32];
    static const int px = 100;
    
    if (!window_initialized) {
        InitWindow(px * SIZE, px * SIZE + 50, "2048");
        SetTargetFPS(30); // Increased for smoother rendering
        window_initialized = true;
    }
    
    if (IsKeyDown(KEY_ESCAPE)) {
        CloseWindow();
        exit(0);
    }

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);

    // Draw grid
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            int val = game->grid[i][j];
            
            // Use precomputed colors
            Color color = (val == 0) ? tile_colors[0] : 
                         (val <= 11) ? tile_colors[val] : 
                         (Color){60, 60, 60, 255};
            
            DrawRectangle(j * px, i * px, px - 5, px - 5, color);
            
            if (val > 0) {
                int display_val = 1 << val; // Power of 2
                // Pre-format text to avoid repeated formatting
                snprintf(score_text, sizeof(score_text), "%d", display_val);
                if (display_val < 1000) {
                    DrawText(score_text, j * px + 30, i * px + 40, 32, PUFF_WHITE);
                } else {
                    DrawText(score_text, j * px + 20, i * px + 40, 32, PUFF_WHITE);
                }
            }
        }
    }
    
    // Draw score (format once per frame)
    snprintf(score_text, sizeof(score_text), "Score: %d", game->score);
    DrawText(score_text, 10, px * SIZE + 10, 24, PUFF_WHITE);

    snprintf(score_text, sizeof(score_text), "Moves: %d", game->moves_made);
    DrawText(score_text, 210, px * SIZE + 10, 24, PUFF_WHITE);
    
    EndDrawing();
}

void c_close(Game* game) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
