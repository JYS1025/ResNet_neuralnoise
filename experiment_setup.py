import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Define global constants
SIGMA_LEVELS = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
BATCH_SIZE = 128

# Auto-detect and set device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")

# Load pre-trained ResNet-18 model
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model = model.to(DEVICE)
model.eval()

# Define transformations for CIFAR-10 test dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Load CIFAR-10 test dataset
test_dataset = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)

# Create DataLoader for the test dataset
test_loader = DataLoader(
    dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)

print("ResNet-18 model loaded and CIFAR-10 test data prepared.")

def add_gaussian_noise(image_tensor, sigma):
    noise = torch.randn_like(image_tensor) * sigma
    noisy_image_tensor = image_tensor + noise
    noisy_image_tensor = torch.clamp(noisy_image_tensor, 0., 1.)
    return noisy_image_tensor

# --- Hooking Mechanism ---
captured_outputs = {}
hook_handles = [] # Global list to store hook handles

def get_capture_hook(name, store_input=False, input_index=0):
    def hook(module, input_val, output_val):
        if store_input:
            captured_outputs[name] = input_val[input_index].clone().detach()
        else:
            captured_outputs[name] = output_val.clone().detach()
    return hook

def remove_hooks():
    global hook_handles # Ensure we are modifying the global list
    for handle in hook_handles:
        handle.remove()
    print(f"Removed {len(hook_handles)} hooks successfully.")
    hook_handles = [] # Clear the list after removing handles


block_names_for_hooks = []
for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
    layer = getattr(model, layer_name)
    for block_idx, block in enumerate(layer):
        if isinstance(block, models.resnet.BasicBlock):
            prefix_name = f"{layer_name}.{block_idx}"
            block_names_for_hooks.append(prefix_name)

            bn2_hook = block.bn2.register_forward_hook(
                get_capture_hook(f"{prefix_name}.bn2_out", store_input=False)
            )
            hook_handles.append(bn2_hook)

            relu_hook = block.relu.register_forward_hook(
                get_capture_hook(f"{prefix_name}.relu_in", store_input=True, input_index=0)
            )
            hook_handles.append(relu_hook)

print(f"Registered {len(hook_handles)} hooks for {len(block_names_for_hooks)} BasicBlocks.")
print(f"Block identifiers: {block_names_for_hooks}")


# --- Metrics Calculation and Experiment Loop ---
final_results = {}
metric_names = ['l2_norm_hx', 'var_hx', 'mean_relu_in', 'sparsity_relu_out']

for sigma in SIGMA_LEVELS:
    print(f"\nProcessing for sigma: {sigma}")

    batch_metrics = {metric: {b_name: [] for b_name in block_names_for_hooks} for metric in metric_names}

    for batch_idx, (images, _) in enumerate(test_loader):
        images = images.to(DEVICE)
        captured_outputs.clear()

        images_for_model = images
        if sigma > 0:
            # As noted before, this adds noise to normalized images and clamps.
            # This might not be the ideal way to handle noise if images are expected to be [0,1] by add_gaussian_noise.
            noisy_images_normalized = add_gaussian_noise(images, sigma)
            images_for_model = noisy_images_normalized

        with torch.no_grad():
            _ = model(images_for_model)

        for block_id in block_names_for_hooks:
            h_x = captured_outputs.get(f"{block_id}.bn2_out")
            relu_input = captured_outputs.get(f"{block_id}.relu_in")

            if h_x is not None and relu_input is not None:
                batch_metrics['l2_norm_hx'][block_id].append(torch.norm(h_x, p=2).item())
                batch_metrics['var_hx'][block_id].append(torch.var(h_x).item())
                batch_metrics['mean_relu_in'][block_id].append(torch.mean(relu_input).item())

                relu_output = torch.relu(relu_input)
                batch_metrics['sparsity_relu_out'][block_id].append((relu_output == 0).float().mean().item())
            else:
                print(f"Warning: Missing captures for block {block_id} in batch {batch_idx} for sigma {sigma}")

        if batch_idx % 10 == 0:
             print(f"  Sigma {sigma}, Batch {batch_idx+1}/{len(test_loader)}") # Corrected batch_idx for 1-based display

    averaged_metrics_for_sigma = {
        metric: {
            b_name: np.mean(batch_metrics[metric][b_name]) if batch_metrics[metric][b_name] else np.nan
            for b_name in block_names_for_hooks
        }
        for metric in metric_names
    }
    final_results[sigma] = averaged_metrics_for_sigma
    print(f"Finished processing for sigma: {sigma}")

# --- Cleanup Hooks ---
remove_hooks() # Ensure this is called after all processing needing hooks is done

print("\n--- Experiment Complete ---")

print("\nExample - L2 norm of h(x) for layer1.0:")
for sigma_val, metrics in final_results.items():
    if 'layer1.0' in metrics['l2_norm_hx']: # Check if block exists (it should)
        # Check if metric value is not nan
        metric_val = metrics['l2_norm_hx']['layer1.0']
        if not np.isnan(metric_val):
            print(f"Sigma {sigma_val:.2f}: {metric_val:.4f}")
        else:
            print(f"Sigma {sigma_val:.2f}: NaN")


def plot_metrics(results_dict, b_names, metric_display_map):
    global SIGMA_LEVELS # Ensure SIGMA_LEVELS is accessible

    for metric_key, display_name in metric_display_map.items():
        plt.figure(figsize=(20, 10)) # Increased figure size for better readability

        num_layers = 4
        blocks_per_layer = {f'layer{i+1}': [] for i in range(num_layers)}
        for bn in b_names:
            layer_prefix = bn.split('.')[0]
            if layer_prefix in blocks_per_layer:
                 blocks_per_layer[layer_prefix].append(bn)

        subplot_idx = 1
        for layer_prefix, current_layer_blocks in blocks_per_layer.items():
            if not current_layer_blocks: continue

            # Create a subplot for each layer if there are blocks in it
            ax = plt.subplot(2, 2, subplot_idx) # Arranged in a 2x2 grid
            subplot_idx +=1

            for block_id in current_layer_blocks:
                metric_values = [results_dict[s][metric_key][block_id] for s in SIGMA_LEVELS if s in results_dict and block_id in results_dict[s][metric_key]]
                valid_sigmas = [s for s in SIGMA_LEVELS if s in results_dict and block_id in results_dict[s][metric_key] and not np.isnan(results_dict[s][metric_key][block_id])]
                valid_metric_values = [results_dict[s][metric_key][block_id] for s in valid_sigmas]

                if valid_metric_values: # Only plot if there's valid data
                    ax.plot(valid_sigmas, valid_metric_values, marker='o', linestyle='-', label=f'{block_id}')

            ax.set_xlabel("Sigma (Noise Level)")
            ax.set_ylabel(display_name)
            ax.set_title(f"{layer_prefix}") # Simplified title, suptitle has full metric name
            ax.legend(fontsize='x-small') # Adjusted fontsize
            ax.grid(True)

        plt.suptitle(f"{display_name} vs. Noise Level", fontsize=18) # Increased suptitle fontsize
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjusted rect for suptitle
        plot_filename = f"{metric_key}_vs_noise.png"
        plt.savefig(plot_filename)
        print(f"Plot saved to {plot_filename}")
        plt.close() # Close figure to free memory

metric_display_names = {
    'l2_norm_hx': 'L2 Norm of h(x)',
    'var_hx': 'Variance of h(x)',
    'mean_relu_in': 'Mean of ReLU Input (x_res + h(x))',
    'sparsity_relu_out': 'Sparsity of ReLU Output'
}

# Call plotting function
if final_results and block_names_for_hooks: # Ensure there are results to plot
    plot_metrics(final_results, block_names_for_hooks, metric_display_names)
else:
    print("No results to plot or block names not defined.")

print("\nScript finished.")
