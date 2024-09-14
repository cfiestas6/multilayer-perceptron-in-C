# ifndef MLP_H
# define MLP_H

# include "layer.h"
# include "dataset.h"
# include <memory.h>

typedef struct {
	Layer **layers;
	int n_layers;
} MLP;

MLP *create_mlp(int n_layers, int *layer_sizes);
void train_mlp(MLP *mlp, double **inputs, double **outputs, int n_samples, int n_epochs, double learning_rate);
double *predict_mlp(MLP *mlp, double *input);
void free_mlp(MLP *mlp);

# endif
