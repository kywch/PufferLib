#include "puffernet.h"

typedef struct G2048Net G2048Net;
struct G2048Net {
    float* obs;
    Linear* layer1;
    GELU* gelu1;
    Linear* layer2;
    GELU* gelu2;
    Linear* layer3;
    GELU* gelu3;
    LSTM* lstm;
    Linear* actor;
    Linear* value_fn;
    Multidiscrete* multidiscrete;
};

G2048Net* make_g2048net(Weights* weights, int input_dim) {
    G2048Net* net = calloc(1, sizeof(G2048Net));
    const int num_agents = 1;
    const int num_actions = 1;
    const int atn_sum = 4;
    const int hidden_dim = 128;
    int logit_sizes[1] = {4};
    net->obs = calloc(num_agents*input_dim, sizeof(float));
    net->layer1 = make_linear(weights, num_agents, input_dim, 512);
    net->gelu1 = make_gelu(num_agents, 512);
    net->layer2 = make_linear(weights, num_agents, 512, 256);
    net->gelu2 = make_gelu(num_agents, 256);
    net->layer3 = make_linear(weights, num_agents, 256, hidden_dim);
    net->gelu3 = make_gelu(num_agents, hidden_dim);
    net->actor = make_linear(weights, num_agents, hidden_dim, atn_sum);
    net->value_fn = make_linear(weights, num_agents, hidden_dim, 1);
    net->lstm = make_lstm(weights, num_agents, hidden_dim, hidden_dim);
    net->multidiscrete = make_multidiscrete(num_agents, logit_sizes, num_actions);
    return net;
}

void free_g2048net(G2048Net* net) {
    free(net->obs);
    free(net->layer1);
    free(net->gelu1);
    free(net->layer2);
    free(net->gelu2);
    free(net->layer3);
    free(net->gelu3);
    free(net->actor);
    free(net->value_fn);
    free(net->lstm);
    free(net->multidiscrete);
    free(net);
}

void forward_g2048net(G2048Net* net, unsigned char* observations, int* actions) {
    for (int i = 0; i < net->layer1->input_dim; i++) {
        net->obs[i] = (float)observations[i];
    }

    linear(net->layer1, net->obs);
    gelu(net->gelu1, net->layer1->output);
    linear(net->layer2, net->gelu1->output);
    gelu(net->gelu2, net->layer2->output);
    linear(net->layer3, net->gelu2->output);
    gelu(net->gelu3, net->layer3->output);
    lstm(net->lstm, net->gelu3->output);
    linear(net->actor, net->lstm->state_h);
    linear(net->value_fn, net->lstm->state_h);
    softmax_multidiscrete(net->multidiscrete, net->actor->output, actions);
}
