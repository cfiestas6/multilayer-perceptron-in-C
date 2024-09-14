# ifndef LAYER_H
# define LAYER_H

# include "neuron.h"

typedef struct {
	Neuron **neurons;
	int n_neurons;
} Layer;

Layer *create_layer(int n_neurons, int n_inputs);
void forward_layer(Layer *layer, double *inputs);
void backward_layer(Layer *layer, Layer *next_layer, int is_output_layer);
void free_layer(Layer *layer);

# endif
