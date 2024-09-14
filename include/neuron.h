# ifndef NEURON_H
# define NEURON_H

# include <math.h>
# include <stdlib.h>
# include <stdio.h>

typedef struct {
	double *weights;
	double bias;
	double output;
	double delta;
	double *inputs;
	int n_inputs;
} Neuron;

Neuron *create_neuron(int n_inputs);
double activate_neuron(Neuron *neuron, double *inputs);

double sigmoid(double x);
double sigmoid_derivative(double x);
double relu(double x);
double relu_derivative(double x);
void update_weights(Neuron *neuron, double learning_rate);
void free_neuron(Neuron *neuron);

# endif
