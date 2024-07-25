import numpy as np
from rsatoolbox.data import Dataset
from rsatoolbox.rdm import calc_rdm
from rsatoolbox.model import ModelFixed
from rsatoolbox.inference import eval_fixed
import matplotlib.pyplot as plt
from rsatoolbox.vis import show_rdm



data = np.random.rand(5, 100) # 5 conditions, 100 voxels
conditions = ['condition1', 'condition2', 'condition3', 'condition4', 'condition5']

dataset = Dataset(data, descriptors={'conditions': conditions})

rdms = calc_rdm(dataset, method='euclidean')

print(rdms)


show_rdm(rdms, patterns=conditions)
plt.show()

'''
This code generates a random dataset with 5 conditions and 100 voxels each. It then calculates the Euclidean RDM (Riemannian Distance Matrix) using the `rsatoolbox` library and displays the RDM using the `show_rdm` function from `rsatoolbox.vis`.

Note that you need to have the `rsatoolbox` library installed to run this code. Additionally, you may need to install the `matplotlib` library if you want to display the RDM plot.
'''

model_rdm = np.eye(5) # 5x5 identity matrix for model RDM
model = ModelFixed("Identity Model", model_rdm) # Create a fixed model with the identity RDM

eval = eval_fixed(model, rdms) # Evaluate the model on the RDMs
print(f'Model fit: {eval}') # Print the evaluation results

'''
This code creates a fixed model with an identity RDM and evaluates it on the provided RDMs. The `eval_fixed` function from `rsatoolbox.inference` is used for this purpose. The evaluation results, including the distance (d), similarity (s), and correlation (c) values, are printed to the console.

Note that the `eval_fixed` function assumes that the input RDMs are represented as a numpy array or a similar data structure compatible with the `rsatoolbox` library.
'''