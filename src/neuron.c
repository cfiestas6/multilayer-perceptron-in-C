#include "../include/neuron.h"

// =======================================
//  Neuron Functions
// =======================================

Neuron *create_neuron(int n_inputs)
{
	Neuron *neuron = (Neuron *) malloc(sizeof(Neuron));
	neuron->n_inputs = n_inputs;
	neuron->weights = (double *) malloc(n_inputs * sizeof(double));
	
	for (int i = 0; i < n_inputs; i++)
		neuron->weights[i] = ((double) rand() / RAND_MAX) * 2 - 1;

	neuron->bias = ((double) rand() / RAND_MAX) * 2 - 1;
	return neuron;
}

double activate_neuron(Neuron *neuron, double *inputs) {
    neuron->inputs = inputs;
    double activation = neuron->bias;
    for (int i = 0; i < neuron->n_inputs; i++)
        activation += neuron->weights[i] * inputs[i];
    neuron->output = sigmoid(activation);
    return neuron->output;
}

double sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

double relu(double x)
{
	return (x > 0) ? x : 0;
}

double relu_derivative(double x)
{
	return (x > 0) ? 1 : 0;
}

double sigmoid_derivative(double sigmoid_output)
{
	return sigmoid_output * (1.0 - sigmoid_output);
}

void update_weights(Neuron *neuron, double learning_rate)
{
	for (int i = 0; i < neuron->n_inputs; i++)
		neuron->weights[i] -= learning_rate * neuron->delta * neuron->inputs[i];
	neuron->bias -= learning_rate * neuron->delta;
}

void free_neuron(Neuron *neuron)
{
	free(neuron->weights);
	free(neuron);
}
