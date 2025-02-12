#ifndef MAIN_H
#define MAIN_H

typedef struct
{
    int n;
    float dt;
    float softening;
    float radius;
    float centre_x;
    float centre_y;
    int screen_width;
    int screen_height;
    int dev_id;
    dim3 block_dim;
    dim3 thread_dim;
} params_t;

int handle_args(int argc, char* argv[], params_t* params);

#endif