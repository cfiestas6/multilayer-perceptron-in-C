# include "../include/mlp.h"

// =======================================
//  Example Program, training, testing
//  and evaluating.
// =======================================

int main() {
    int num_samples, num_features;
    double **outputs;
    double **dataset = load_dataset("data/customer_purchase_data.csv", &num_samples, &num_features, &outputs, ",");

    double **normalized_dataset = normalize_dataset(dataset, num_samples, num_features);

    int layer_sizes[] = {num_features, 10, 10, 1};
    MLP *mlp = create_mlp(4, layer_sizes);

    train_mlp(mlp, normalized_dataset, outputs, num_samples, 5000, 0.01);

    for (int i = 0; i < num_samples; i++) {
        free(dataset[i]);
        free(normalized_dataset[i]);
        free(outputs[i]);
    }
    free(dataset);
    free(normalized_dataset);
    free(outputs);

    int test_samples, test_features;
    double **test_outputs;
    double **test_dataset = load_dataset("data/customer_purchase_testing.csv", &test_samples, &test_features, &test_outputs, ",");

    double **normalized_test_dataset = normalize_dataset(test_dataset, test_samples, test_features);

    // Predict and evaluate
    int correct_predictions = 0;
    int true_positive = 0;
    int true_negative = 0;
    int false_positive = 0;
    int false_negative = 0;
    for (int i = 0; i < test_samples; i++) {
        double *prediction = predict_mlp(mlp, normalized_test_dataset[i]);

        int predicted_label = (prediction[0] >= 0.5) ? 1 : 0;
        int actual_label = (test_outputs[i][0] >= 0.5) ? 1 : 0;

        if (predicted_label == 1 && actual_label == 1) {
            true_positive++;
        } else if (predicted_label == 1 && actual_label == 0) {
            false_positive++;
        } else if (predicted_label == 0 && actual_label == 1) {
            false_negative++;
        } else if (predicted_label == 0 && actual_label == 0) {
            true_negative++;
        }

        if (predicted_label == actual_label) {
            correct_predictions++;
        }

        printf("Prediction for test sample %d: %f (Actual: %f)\n", i + 1, prediction[0], test_outputs[i][0]);
        free(prediction);
    }

    double accuracy = ((double)correct_predictions / test_samples) * 100.0;
    printf("\n====================================\n");
    printf("Accuracy on test dataset: %.2f%%\n", accuracy);
    printf("====================================\n");

    // Calculate precision, recall, and F1-score
    double precision = (true_positive + false_positive) > 0
                      ? (double)true_positive / (true_positive + false_positive)
                      : 0.0;
    double recall = (true_positive + false_negative) > 0
                   ? (double)true_positive / (true_positive + false_negative)
                   : 0.0;
    double f1_score = (precision + recall) > 0
                     ? 2 * (precision * recall) / (precision + recall)
                     : 0.0;

    printf("Precision: %.2f%%\n", precision * 100.0);
    printf("Recall: %.2f%%\n", recall * 100.0);
    printf("F1 Score: %.2f%%\n", f1_score * 100.0);

    printf("Confusion Matrix:\n");
    printf("TP: %d, TN: %d, FP: %d, FN: %d\n",
           true_positive, true_negative, false_positive, false_negative);

    free_mlp(mlp);

    for (int i = 0; i < test_samples; i++) {
        free(test_dataset[i]);
        free(normalized_test_dataset[i]);
        free(test_outputs[i]);
    }
    free(test_dataset);
    free(normalized_test_dataset);
    free(test_outputs);

    return 0;
}