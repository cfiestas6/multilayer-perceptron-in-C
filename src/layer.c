# include "../include/layer.h"

// =======================================
//	Layer Functions
// =======================================

Layer *create_layer(int n_neurons, int n_inputs)
{
	Layer *layer = (Layer *) malloc(sizeof(Layer));
	layer->n_neurons = n_neurons;
	layer->neurons = (Neuron **) malloc(n_neurons * sizeof(Neuron *));

	for (int i = 0; i < n_neurons; i++)
		layer->neurons[i] = create_neuron(n_inputs);

	return layer;
}

void forward_layer(Layer *layer, double *inputs)
{
	for (int i = 0; i < layer->n_neurons; i++)
		layer->neurons[i]->output = activate_neuron(layer->neurons[i], inputs);
}

void backward_layer(Layer *layer, Layer *next_layer, int is_output_layer) {
	if (is_output_layer) {
		return; // Output layer delta is calculated in train_mlp (mlp.c)
	} 
	// Hidden layer error calculation
	for (int i = 0; i < layer->n_neurons; i++) {
		Neuron *neuron = layer->neurons[i];
		double error = 0.0;
		for (int j = 0; j < next_layer->n_neurons; j++) {
			error += next_layer->neurons[j]->weights[i] * next_layer->neurons[j]->delta;
		}
		neuron->delta = error * sigmoid_derivative(neuron->output);
	}
}

void free_layer(Layer *layer)
{
	for (int i = 0; i < layer->n_neurons; i++)
		free_neuron(layer->neurons[i]);
	free(layer->neurons);
	free(layer);
}
