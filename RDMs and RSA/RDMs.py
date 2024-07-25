import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform

# Step 1: Generate synthetic data
np.random.seed(0)
data = np.random.rand(10, 5)  # 10 conditions, 5 features each

# Step 2: Calculate pairwise dissimilarities (Euclidean distance)
dissimilarities = pdist(data, metric='euclidean')

# Step 3: Create the RDM
RDM = squareform(dissimilarities)

# Plot the RDM
plt.figure(figsize=(8, 6))
sns.heatmap(RDM, annot=True, cmap='viridis')
plt.title('Representational Dissimilarity Matrix (RDM)')
plt.xlabel('Condition')
plt.ylabel('Condition')
plt.show()
