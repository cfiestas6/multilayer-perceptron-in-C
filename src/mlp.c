# include "../include/mlp.h"

// =======================================
//  Multilayer Perceptron Functions
// =======================================

MLP *create_mlp(int n_layers, int *layer_sizes) {
	MLP *mlp = (MLP *) malloc(sizeof(MLP));
	mlp->n_layers = n_layers;
	mlp->layers = (Layer **)malloc(n_layers * sizeof(Layer *));

	for (int i = 0; i < n_layers; i++)
		mlp->layers[i] = create_layer(
			layer_sizes[i], 
			(i == 0) ? layer_sizes[i] : layer_sizes[i - 1]
		);
	return mlp;
}

void train_mlp(MLP *mlp, double **inputs, double **outputs, int n_samples, int n_epochs, double learning_rate) {
    for (int epoch = 0; epoch < n_epochs; epoch++) {
        double total_error = 0.0;

        for (int sample = 0; sample < n_samples; sample++) {
            // Forward pass
            double *layer_input = inputs[sample];
            for (int i = 0; i < mlp->n_layers; i++) {
                forward_layer(mlp->layers[i], layer_input);

                // Prepare input for next layer
                if (i < mlp->n_layers - 1) {
                    double *next_input = (double *)malloc(mlp->layers[i]->n_neurons * sizeof(double));
                    for (int j = 0; j < mlp->layers[i]->n_neurons; j++)
                        next_input[j] = mlp->layers[i]->neurons[j]->output;

                    if (i > 0)
                        free(layer_input);

                    layer_input = next_input;
                }
            }

            // Compute error and delta for output layer neurons
            Layer *output_layer = mlp->layers[mlp->n_layers - 1];
            for (int i = 0; i < output_layer->n_neurons; i++) {
                Neuron *neuron = output_layer->neurons[i];
                double output = neuron->output;
                double target_value = outputs[sample][i];

                // Calculate loss (Cross-Entropy)
                total_error += - (target_value * log(output + 1e-15) + (1 - target_value) * log(1 - output + 1e-15));

                // Calculate delta
                neuron->delta = output - target_value;
            }

            // Backward pass
            for (int i = mlp->n_layers - 1; i >= 0; i--) {
                if (i == mlp->n_layers - 1) {
                    backward_layer(mlp->layers[i], NULL, 1);
                } else {
                    backward_layer(mlp->layers[i], mlp->layers[i + 1], 0);
                }
            }

            // Update weights
            for (int i = 0; i < mlp->n_layers; i++) {
                Layer *layer = mlp->layers[i];
                for (int j = 0; j < layer->n_neurons; j++) {
                    update_weights(layer->neurons[j], learning_rate);
                }
            }

            if (mlp->n_layers > 1)
                free(layer_input);
        }

        // Average loss over all samples
        printf("Epoch %d/%d, Loss: %f\n", epoch + 1, n_epochs, total_error / n_samples);
    }
}

double *predict_mlp(MLP *mlp, double *input)
{
    double *layer_input = input;
    double *output = NULL;

    for (int i = 0; i < mlp->n_layers; i++)
    {
        forward_layer(mlp->layers[i], layer_input);

        // Prepare input for next layer
        output = (double *)malloc(mlp->layers[i]->n_neurons * sizeof(double));
        for (int j = 0; j < mlp->layers[i]->n_neurons; j++)
        {
            output[j] = mlp->layers[i]->neurons[j]->output;
        }

        if (i > 0)
            free(layer_input);

        layer_input = output;
    }

    return output;
}

void free_mlp(MLP *mlp)
{
	for (int i = 0; i < mlp->n_layers; i++)
	{
		free_layer(mlp->layers[i]);
	}
	free(mlp->layers);
	free(mlp);
}