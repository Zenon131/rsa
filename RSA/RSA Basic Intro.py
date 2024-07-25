import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


# Mock data: neural activity patterns (e.g., from different conditions)
# Each row is a different condition, each column is a different voxel (simplified)
neural_activity = np.random.rand(5, 100)  # 5 conditions, 100 voxels


# Compute the dissimilarity matrix using Euclidean distance
dissimilarity_matrix = pdist(neural_activity, metric='euclidean')


# Convert to a square form matrix
RDM = squareform(dissimilarity_matrix)


# Plot the RDM
plt.figure(figsize=(8, 6))
sns.heatmap(RDM, annot=True, cmap='viridis')
plt.title('Representational Dissimilarity Matrix (RDM)')
plt.xlabel('Conditions')
plt.ylabel('Conditions')
plt.show()