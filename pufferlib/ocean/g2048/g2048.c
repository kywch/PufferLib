#include "g2048.h"
#include "g2048_net.h"

#define OBS_DIM 288

int main() {
    srand(time(NULL));
    Game env;
    unsigned char observations[OBS_DIM] = {0};
    unsigned char terminals[1] = {0};
    int actions[1] = {0};
    float rewards[1] = {0};

    env.observations = observations;
    env.terminals = terminals;
    env.actions = actions;
    env.rewards = rewards;

    Weights* weights = load_weights("resources/g2048/g2048_weights.bin", 444933);
    G2048Net* net = make_g2048net(weights, OBS_DIM);
    c_reset(&env);
    c_render(&env);

    // Main game loop
    int frame = 0;
    int action = -1;
    while (!WindowShouldClose()) {
        c_render(&env);
        frame++;
        
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            action = -1;
            if (IsKeyDown(KEY_W) || IsKeyDown(KEY_UP)) action = UP;
            else if (IsKeyDown(KEY_S) || IsKeyDown(KEY_DOWN)) action = DOWN;
            else if (IsKeyDown(KEY_A) || IsKeyDown(KEY_LEFT)) action = LEFT;
            else if (IsKeyDown(KEY_D) || IsKeyDown(KEY_RIGHT)) action = RIGHT;
            env.actions[0] = action - 1;
        } else if (frame % 1 != 0) {
            continue;
        } else {
            action = 1;
            forward_g2048net(net, env.observations, env.actions);
        }

        if (action > 0) {
            step_without_reset(&env);
        }

        if (env.terminals[0] == 1) { 
            WaitTime(10);
            c_reset(&env);
            c_render(&env);
            frame = 0;
        }

        if (IsKeyDown(KEY_LEFT_SHIFT) && action > 0) {
            // Don't need to be super reactive
            WaitTime(0.1);
        }        
    }

    free_g2048net(net);
    c_close(&env);
    printf("Game Over! Final Max Tile: %d\n", env.score);
    return 0;
}
