import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import rsatoolbox

def load_model_rdms(file_path):
    """
    Load model RDMs from a MATLAB file.

    Parameters:
    file_path (str): Path to the MATLAB file containing the model RDMs.

    Returns:
    model_rdms (rsatoolbox.rdm.RDMs): RDMs object containing the model RDMs.
    """
    try:
        matlab_data = io.loadmat(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

    matlab_data = matlab_data['modelRDMs']
    n_models = len(matlab_data[0])
    model_names = [matlab_data[0][i][0][0] for i in range(n_models)]
    measurement_model = [matlab_data[0][i][1][0] for i in range(n_models)]
    rdms_array = np.array([matlab_data[0][i][3][0] for i in range(n_models)])

    model_rdms = rsatoolbox.rdm.RDMs(rdms_array,
                                    rdm_descriptors={'brain_computational_model': model_names,
                                                     'measurement_model': measurement_model},
                                    dissimilarity_measure='Euclidean'
                                   )
    return model_rdms

def load_noisy_rdms(file_path):
    """
    Load noisy RDMs from a MATLAB file.

    Parameters:
    file_path (str): Path to the MATLAB file containing the noisy RDMs.

    Returns:
    rdms_data (rsatoolbox.rdm.RDMs): RDMs object containing the noisy RDMs.
    """
    try:
        matlab_data = io.loadmat(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

    repr_names_matlab = matlab_data['reprNames']
    fwhms_matlab = matlab_data['FWHMs']
    noise_std_matlab = matlab_data['relNoiseStds']
    rdms_matlab = matlab_data['noisyModelRDMs']
    repr_names = [repr_names_matlab[i][0][0] for i in range(repr_names_matlab.shape[0])]
    fwhms = fwhms_matlab.squeeze().astype('float')
    noise_std = noise_std_matlab.squeeze().astype('float')
    rdms_matrix = rdms_matlab.squeeze().astype('float')

    i_rep = 2 #np.random.randint(len(repr_names))
    i_noise = 1 #np.random.randint(len(noise_std))
    i_fwhm = 0 #np.random.randint(len(fwhms))

    repr_name = repr_names[i_rep]
    print('The chosen ground truth model is:')
    print(repr_name)
    print('with noise level:')
    print(noise_std[i_noise])
    print('with averaging width (full width at half magnitude):')
    print(fwhms[i_fwhm])

    rdms_data = rsatoolbox.rdm.RDMs(rdms_matrix[:, i_rep, i_fwhm, i_noise, :].transpose())
    return rdms_data

def create_models(model_rdms):
    """
    Create a list of models from the model RDMs.

    Parameters:
    model_rdms (rsatoolbox.rdm.RDMs): RDMs object containing the model RDMs.

    Returns:
    models (list): List of rsatoolbox.model.ModelFixed objects.
    """
    models = []
    for i_model in np.unique(model_rdms.rdm_descriptors['brain_computational_model']):
        rdm_m = model_rdms.subset('brain_computational_model', i_model).subset('measurement_model', 'complete')
        m = rsatoolbox.model.ModelFixed(i_model, rdm_m)
        models.append(m)
    return models

def evaluate_models(models, rdms_data, method='corr'):
    """
    Evaluate the models using the specified method.

    Parameters:
    models (list): List of rsatoolbox.model.ModelFixed objects.
    rdms_data (rsatoolbox.rdm.RDMs): RDMs object containing the noisy RDMs.
    method (str): Evaluation method (default: 'corr').

    Returns:
    results (rsatoolbox.inference.Results): Results object containing the evaluation results.
    """
    results = rsatoolbox.inference.eval_fixed(models, rdms_data, method=method)
    return results

def plot_model_comparison(results):
    """
    Plot the model comparison results.

    Parameters:
    results (rsatoolbox.inference.Results): Results object containing the evaluation results.
    """
    rsatoolbox.vis.plot_model_comparison(results)
    plt.show()

# Exercise 1
model_rdms = load_model_rdms('rdms_inferring/modelRDMs_A2020.mat')
if model_rdms is not None:
    conv1_rdms = model_rdms.subset('brain_computational_model', 'conv1')
    fig, ax, ret_val = rsatoolbox.vis.show_rdm(conv1_rdms, rdm_descriptor='measurement_model', figsize=(10, 10))
    plt.show()
else:
    print("Model RDMs could not be loaded.")

# Exercise 2
rdms_data = load_noisy_rdms('rdms_inferring/noisyModelRDMs_A2020.mat')
if rdms_data is not None and model_rdms is not None:
    models = create_models(model_rdms)
    results_1 = evaluate_models(models, rdms_data, method='corr')
    plot_model_comparison(results_1)
else:
    print("Noisy RDMs or Model RDMs could not be loaded.")

# Exercise 3
if model_rdms is not None:
    models_flex = []
    for i_model in np.unique(model_rdms.rdm_descriptors['brain_computational_model']):
        models_flex.append(rsatoolbox.model.ModelSelect(i_model, model_rdms.subset('brain_computational_model', i_model)))

    if rdms_data is not None:
        train_set, test_set, ceil_set = rsatoolbox.inference.sets_k_fold(rdms_data, k_pattern=3, k_rdm=2)

        results_3_cv = rsatoolbox.inference.crossval(models_flex, rdms_data, train_set, test_set, ceil_set=ceil_set, method='corr')
        plot_model_comparison(results_3_cv)

        results_3_full = rsatoolbox.inference.eval_dual_bootstrap(models_flex, rdms_data, k_pattern=4, k_rdm=2, method='corr', N=100)
        plot_model_comparison(results_3_full)
else:
    print("Model RDMs could not be loaded.")
