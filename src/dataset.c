# include "../include/dataset.h"

// =======================================
//	Dataset Loading Functions
// =======================================

// Just for fun... could've done this in python xd
double **load_dataset(const char *filename, int *n_samples, int *n_features, double ***outputs, char *delimiter)
{
	FILE *file = fopen(filename, "r");
	if (!file) {
		perror("File not found!");
		return 0;
	}

	// Count the number of samples and features
	char line[1024];

	*n_samples = 0;
	*n_features = -1;

	while (fgets(line, 1024, file))
	{
		(*n_samples)++;

		if (*n_features == -1)
		{
			char *token = strtok(line, delimiter);
			while (token)
			{
				(*n_features)++;
				token = strtok(NULL, delimiter);
			}
			
			// Ignore ouput column (y)
			(*n_features)--;
		}
	}

	double **dataset = (double **) malloc((*n_samples) * sizeof(double *));
	double **output = (double **) malloc((*n_samples) * sizeof(double *));

	for (int i = 0; i < *n_samples; i++)
	{
		dataset[i] = (double *) malloc((*n_features) * sizeof(double));
		output[i] = (double *) malloc(sizeof(double *));
	}

	// Read and load
	fseek(file, 0, SEEK_SET);
	int x = 0;
	while (fgets(line, 1024, file))
	{
		char *token = strtok(line, delimiter);
		int y = 0;

		while (token)
		{
			if (y < *n_features)
				dataset[x][y] = atof(token);
			else
				output[x][0] = atof(token);
			token = strtok(NULL, delimiter);
			y++;
		}
		x++;
	}
	
	fclose(file);
	*outputs = output;
	return dataset;
}

double **normalize_dataset(double **dataset, int n_samples, int n_features)
{
	double **normalized_dataset = (double **) malloc(n_samples * sizeof(double *));
	for (int i = 0; i < n_samples; i++)
		normalized_dataset[i] = (double *) malloc(n_features * sizeof(double));

	for (int j = 0; j < n_features; j++)
	{
		double min = dataset[0][j];
		double max = dataset[0][j];

		for (int i = 1; i < n_samples; i++)
		{
			if (dataset[i][j] < min) min = dataset[i][j];
			if (dataset[i][j] > max) max = dataset[i][j];
		}

		for (int i = 0; i < n_samples; i++)
		{
			if (max - min == 0) normalized_dataset[i][j] = 0.0;
			else normalized_dataset[i][j] = (dataset[i][j] - min) / (max - min);
		}
	}
	return normalized_dataset;
}