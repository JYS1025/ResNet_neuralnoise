# Investigating Internal Dynamics of Pre-trained ResNet Models Under Input Uncertainty

## Overview

This experiment aims to understand how a pre-trained ResNet-18 model, when subjected to increasing levels of Gaussian noise in its input images, adapts its internal representations. Specifically, it focuses on the behavior of the residual function outputs (`h(x)`) and the inputs to ReLU activation functions (`x + h(x)`) within each `BasicBlock` of the network. The investigation is centered around how the statistical properties of these intermediate tensors change, providing insights into the network's strategies for handling input uncertainty.

## Core Hypotheses

The experiment tests two competing hypotheses regarding the ResNet's behavior:

*   **H1 (Active Manipulation for Linearity):** Increased input noise leads to a higher variance in the residual values `h(x)`. This, in turn, pushes the input to the subsequent ReLU activation (`x + h(x)`) further away from zero, thereby making the activation function operate in a more linear regime (i.e., reducing activation sparsity).
*   **H0 (Signal Preservation):** Increased input noise leads to a smaller L2-norm of the residual values `h(x)`. This suggests the model attempts to minimize the influence of the residual path to preserve the original signal `x` carried through the identity connection, especially when the input is corrupted.

## Technical Stack

*   Python 3.x
*   PyTorch
*   Torchvision
*   NumPy
*   Matplotlib

## The Experiment Script (`experiment_setup.py`)

The core of this investigation is the Python script `experiment_setup.py`. Its main functionalities are:

1.  **Model and Data Loading:** Loads a pre-trained ResNet-18 model (from `torchvision.models`) and the CIFAR-10 test dataset. The model is set to evaluation mode.
2.  **Noise Injection:** Defines a range of sigma values representing different levels of Gaussian noise.
3.  **Iterative Processing:** For each defined noise level:
    *   Gaussian noise is added to the batch of input images.
    *   A forward pass is performed with the noisy images.
4.  **Hook-based Data Extraction:** Utilizes PyTorch's `register_forward_hook` mechanism to capture intermediate tensors from each `BasicBlock` without altering the model architecture. The captured tensors are:
    *   `h(x)`: The output of the final batch normalization layer (`bn2`) in the block's residual path, just before being added to the identity stream.
    *   `ReLU_input`: The tensor resulting from the addition of the identity stream and `h(x)` (i.e., `x + h(x)`), which serves as the input to the block's final ReLU activation function.
5.  **Metrics Calculation:** For each `BasicBlock` and each noise level, the script calculates:
    *   The L2-norm of `h(x)`.
    *   The variance of the elements within the `h(x)` tensor.
    *   The mean of the `ReLU_input` tensor.
    *   The activation sparsity of the `ReLU_input` after applying the ReLU function (i.e., the percentage of elements equal to zero).
6.  **Averaging and Storage:** Metrics are averaged across all batches in the test dataset for each noise level and stored.
7.  **Visualization:** Generates and saves plots for each of the four metrics. These plots show the metric's value as a function of the input noise sigma, with different lines or subplots representing different blocks or layers within the ResNet, allowing for analysis of how these dynamics evolve with network depth.

## How to Run the Experiment

1.  **Prerequisites:**
    *   Python 3.x
    *   pip (Python package installer)
2.  **Installation of Dependencies:**
    Open your terminal and run:
    ```bash
    pip install torch torchvision numpy matplotlib
    ```
3.  **Execution:**
    Navigate to the directory containing `experiment_setup.py` and run:
    ```bash
    python experiment_setup.py
    ```

## Expected Output

Upon execution, the script will:

*   Print status messages to the console, including the device being used (CPU/GPU), progress through noise levels and batches, and final confirmation messages.
*   Download the CIFAR-10 dataset if not already present in the `./data` directory.
*   Generate four PNG image files in the same directory as the script:
    *   `l2_norm_hx_vs_noise.png`: Shows the L2-norm of `h(x)` vs. input noise.
    *   `var_hx_vs_noise.png`: Shows the variance of `h(x)` vs. input noise.
    *   `mean_relu_in_vs_noise.png`: Shows the mean of the ReLU input (`x + h(x)`) vs. input noise.
    *   `sparsity_relu_out_vs_noise.png`: Shows the activation sparsity (after ReLU) vs. input noise.
    Each plot will contain multiple lines/subplots corresponding to different `BasicBlock`s within the ResNet layers, allowing for a detailed view of the effects at various depths.

## Interpreting the Results

The generated plots are key to evaluating the core hypotheses:

*   **L2-norm of h(x) (Plot: `l2_norm_hx_vs_noise.png`):**
    *   If this value tends to decrease or remain consistently low as input noise increases, it lends support to **H0**, suggesting the model dampens the residual component to protect the integrity of the identity signal.
*   **Variance of h(x) (Plot: `var_hx_vs_noise.png`):**
    *   An increase in the variance of `h(x)` with rising input noise could be indicative of **H1**, where the model might be amplifying certain features or noise components.
*   **Mean of ReLU Input (Plot: `mean_relu_in_vs_noise.png`):**
    *   Shifts in the mean of `x + h(x)` can reveal biases introduced by `h(x)` in response to noise, potentially affecting the operating point of the ReLU.
*   **Activation Sparsity (Plot: `sparsity_relu_out_vs_noise.png`):**
    *   A significant decrease in activation sparsity (meaning fewer neurons output zero) as noise increases would strongly support **H1**. This would imply that `x + h(x)` is being pushed away from zero, causing more ReLU units to activate and operate in their linear regime.

Carefully observe how these trends manifest across different layers (e.g., `layer1` vs. `layer4`) as this can indicate differing strategies or sensitivities at various stages of feature extraction.

## License

This project is licensed under the MIT License. (Assuming, consistent with typical open-source Python projects. If a specific license is required, this should be updated.)
