#ifndef MAIN_H
#define MAIN_H

typedef struct
{   
    unsigned int n;
    float dt;
    float softening;
    float radius;
    float2 centre;
    unsigned int screen_width;
    unsigned int screen_height;
    int dev_id;
    dim3 block_dim;
    dim3 thread_dim;
} params_t;

int handle_args(int argc, char* argv[], params_t* params);

#endif