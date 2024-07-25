import numpy as np
import matplotlib.pyplot as plt
import rsatoolbox

# EXERCISE 1: Data and RDM Handling
# Step 1: Generating Random Data (For Practice Purposes)
# 1a: Parameters For Random Data
n_layers = 5
n_measurements = 10
n_conditions = 50

# 1b: Generate Random RDMs
np.random.seed(27) # For reproducability
model_names = [f"layer{i+1}" for i in range(n_layers) for _ in range(n_measurements)] # Generates model names based on the number of layers
measurement_model = [f"measurement{i+1}" for _ in range(n_layers) for i in range(n_measurements)] # Generates model names based on the number of measurements
rdms_array = np.random.rand(n_layers * n_measurements, n_conditions, n_conditions) # Random array with the shape n_layers * n_measurements, n_conditions, n_conditions

# 1c: Create RDMs Object
model_rdms = rsatoolbox.rdm.RDMs(rdms_array,
    rdm_descriptors={'brain_computational_model': model_names,
                     'measurement_model': measurement_model},
    dissimilarity_measure='Euclidean')

# Step 2: Plotting RDMs
# 2a: Plotting Layer 1
layer1_rdms = model_rdms.subset('brain_computational_model', 'layer1')
fig, ax, ret_val = rsatoolbox.vis.show_rdm(layer1_rdms, rdm_descriptor='measurement_model', figsize=(10, 10))
plt.show()

# Print Info About RDMs
print(layer1_rdms)

# EXERCISE 2: Fixed Model Inference
# Step 1: Generating Random Data RDMs (Again, For Practice Purposes)
# 1a: Generating A Random Matrix
n_data_rdms = 10
rdms_matrix = np.random.rand(n_data_rdms, n_conditions, n_conditions)

# Metadata
repr_names = [f"data_model_{i}" for i in range(n_data_rdms)]
fwhms = np.random.rand(n_data_rdms)
noise_std = np.random.rand(n_data_rdms)

# 1b: Choose a Data RDM For Inference
i_rep = 2
i_noise = 1
i_fwhm = 0

repr_name = repr_names[i_rep]
print('The chosen ground truth model is:')
print(repr_name)
print('with noise level:')
print(noise_std[i_noise])
print('with averaging width (full width at half magnitude):')
print(fwhms[i_fwhm])

# 1c: Create RDMs Object
data_rdms = rsatoolbox.rdm.RDMs(rdms_matrix,
                                 rdm_descriptors={'brain_computational_model': repr_names})

# Step 2: Plotting RDMs
fig, ax, ret_val = rsatoolbox.vis.show_rdm(data_rdms, rdm_descriptor='brain_computational_model', figsize=(10, 10))
plt.show()

# Print Info About RDMs
print(data_rdms)

# Step 2: Defining And Evaluating Fixed Models
# 2a: Defining Fixed Models
def create_models(model_rdms):
    """
    Create a list of models from the model RDMs.
    """
    models = []
    for i_model in np.unique(model_rdms.rdm_descriptors['brain_computational_model']):
        rdm_m = model_rdms.subset('brain_computational_model', i_model).subset('measurement_model', 'measurement1')
        m = rsatoolbox.model.ModelFixed(i_model, rdm_m)
        models.append(m)
    return models

models = create_models(model_rdms)
results_1 = rsatoolbox.inference.eval_fixed(models, data_rdms, method='corr')
rsatoolbox.vis.plot_model_comparison(results_1)
plt.show()

# Print the results
print(results_1)

# EXERCISE 3: Cross-validation for Flexible Models
# Step 1: Defining And Evaluating Flexible Models
# 1a: Defining Flexible Models
models_flex = []
for i_model in np.unique(model_rdms.rdm_descriptors['brain_computational_model']):
    models_flex.append(rsatoolbox.model.ModelSelect(i_model, model_rdms.subset('brain_computational_model', i_model)))

# Splitting Data Into Training and Test Sets
train_set, test_set, ceil_set = rsatoolbox.inference.sets_k_fold(data_rdms, k_pattern=3, k_rdm=2)

# Cross-validation
results_3_cv = rsatoolbox.inference.crossval(models_flex, data_rdms, train_set, test_set, ceil_set=ceil_set, method='corr')
rsatoolbox.vis.plot_model_comparison(results_3_cv)
plt.show()

# Bootstrapped Cross-validation
results_3_full = rsatoolbox.inference.eval_dual_bootstrap(models_flex, data_rdms, k_pattern=4, k_rdm=2, method='corr', N=100)
rsatoolbox.vis.plot_model_comparison(results_3_full)
plt.show()