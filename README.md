# Multilayer Perceptron Implementation in C
![image](https://github.com/user-attachments/assets/405993dc-615f-4d53-b598-a76238a143a9)

## Project Overview

This project is an implementation of a Multilayer Perceptron (MLP) neural network in C. The MLP is designed to predict customer purchase status based on various features such as age, gender, annual income, and more. The implementation includes:

- Forward and backward propagation algorithms.
- Customizable network architecture.
- Training and evaluation on provided datasets.
- Performance metrics calculation including accuracy, precision, recall, and F1 score.

## File Structure

```
.
├── data/
│   ├── customer_purchase_data.csv
│   └── customer_purchase_testing.csv
├── include/
│   ├── neuron.h
│   ├── layer.h
│   ├── mlp.h
│   └── dataset.h
├── src/
│ ├── dataset.c
│ ├── layer.c
│ ├── main.c
│ ├── mlp.c
│ └── neuron.c
├── Makefile
└── README.md
```

### Directories and Files

- **data/**: Contains the dataset files.
  - `customer_purchase_data.csv`: Training dataset.
  - `customer_purchase_testing.csv`: Testing dataset.
- **include/**: Contains header files with function prototypes and struct definitions.
  - `layer.h`: Definitions and prototypes related to the `Layer` struct.
  - `dataset.h`: Data loading and processing function prototypes.
  - `mlp.h`: Definitions and prototypes for the MLP model.
  - `neuron.h`: Definitions and prototypes for the `Neuron` struct.
- **src/**: Contains the source code files with implementations.
  - `dataset.c`: Data loading and processing.
  - `layer.c`: Layer-related functions (forward and backward propagation).
  - `main.c`: The main program file to run the application.
  - `mlp.c`: MLP model functions (creation, training, prediction).
  - `neuron.c`: Neuron-related functions (activation, weight updates).
- **Makefile**: Contains build instructions.
- **README.md**: Project documentation (this file).

## How to Run the Project

### Prerequisites

- GCC compiler installed (`gcc` command available).
- `make` utility installed.

### Steps to Run

1. **Clone the Repository**

   Clone the repository using:

   ```bash
   git clone https://github.com/cfiestas6/multilayer-perceptron-in-C.git
   ```

   Navigate to the project directory:

   ```bash
   cd multilayer-perceptron-in-C
   ```

2. **Compile the Project**

   Use the `make` command to compile the source code:

   ```bash
   make
   ```

   This will compile all the source files and generate an executable named `mlp`.

3. **Run the Executable**

   Execute the program with:

   ```bash
   ./mlp
   ```

   The program will:

   - Load and normalize the training dataset.
   - Train the MLP model.
   - Load and normalize the testing dataset.
   - Make predictions on the testing data.
   - Calculate and display performance metrics.

```
[...]
Epoch 4992/5000, Loss: 0.438557
Epoch 4993/5000, Loss: 0.438555
Epoch 4994/5000, Loss: 0.438553
Epoch 4995/5000, Loss: 0.438552
Epoch 4996/5000, Loss: 0.438550
Epoch 4997/5000, Loss: 0.438549
Epoch 4998/5000, Loss: 0.438547
Epoch 4999/5000, Loss: 0.438546
Epoch 5000/5000, Loss: 0.438544
```

```
[...]
Prediction for test sample 88: 0.344577 (Actual: 0.000000)
Prediction for test sample 89: 0.375537 (Actual: 0.000000)
Prediction for test sample 90: 0.662297 (Actual: 0.000000)
Prediction for test sample 91: 0.517423 (Actual: 1.000000)
Prediction for test sample 92: 0.044651 (Actual: 1.000000)
Prediction for test sample 93: 0.126743 (Actual: 0.000000)
Prediction for test sample 94: 0.793391 (Actual: 1.000000)
Prediction for test sample 95: 0.594724 (Actual: 1.000000)
```

```
====================================
Accuracy on test dataset: 80.00%
====================================
Precision: 82.05%
Recall: 72.73%
F1 Score: 77.11%
Confusion Matrix:
TP: 32, TN: 44, FP: 7, FN: 12
```

4. **Clean Up Build Files**

   To remove the compiled files and clean up the directory, use:

   ```bash
   make clean
   ```

## Project Details

### MLP Model

- **Architecture**: The MLP model consists of an input layer, two hidden layers with 10 neurons each, and an output layer.
- **Activation Function**: Sigmoid activation function is used for all neurons.
- **Training Parameters**:
  - **Epochs**: 5000
  - **Learning Rate**: 0.01
- **Loss Function**: Cross-entropy loss is used during training.

$$
\text{Loss} = - [y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})]
$$

- **Loss Calculation**:
  - ε is a small value (`1e-15`) added for numerical stability to prevent taking the logarithm of zero.

$$
\text{Loss} = - [y \log(\hat{y} + \varepsilon) + (1 - y) \log(1 - \hat{y} + \varepsilon)]
$$

- **Delta Calculation:** For Cross-Entropy Loss combined with the sigmoid activation function, the derivative simplifies to:

$$
\delta = \hat{y} - y
$$

### Datasets

- **Training Data**: `data/customer_purchase_data.csv`
- **Testing Data**: `data/customer_purchase_testing.csv`
- **Features**:
  - Age
  - Gender
  - Annual Income
  - Number of Purchases
  - Product Category
  - Time Spent on Website
  - Loyalty Program Participation
  - Discounts Availed
- **Target Variable**:
  - Purchase Status (`0` or `1`)

### Performance Metrics

The following metrics are calculated and displayed after testing:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**

Definitions:

- **True Positives (TP)**: Cases where the model correctly predicted the positive class.
- **True Negatives (TN)**: Cases where the model correctly predicted the negative class.
- **False Positives (FP)**: Cases where the model incorrectly predicted the positive class.
- **False Negatives (FN)**: Cases where the model incorrectly predicted the negative class.
--------------
- **Precision** measures the proportion of positive identifications that were actually correct.

$$
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
$$


- **Recall** measures the proportion of actual positives that were identified correctly.

$$
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$


- **F1 Score** is the harmonic mean of precision and recall.

$$
\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

