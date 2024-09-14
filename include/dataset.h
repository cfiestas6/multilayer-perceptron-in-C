# ifndef DATASET_H
# define DATASET_H

# include <stdio.h>
# include <stdlib.h>
# include <string.h>

double **normalize_dataset(double **dataset, int n_samples, int n_features);
double **load_dataset(const char *filename, int *n_samples, int *n_features, double ***outputs, char *delimiter);

# endif
