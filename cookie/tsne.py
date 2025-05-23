"""

Implements a NumPy-based t-SNE algorithm for dimensionality reduction.



"""


import numpy as np
import matplotlib.pyplot as plt


def pca(X, no_dims=50):
    """
    Runs PCA on the nxd array X in order to reduce its dimensionality to
    no_dims dimensions.

    Parameters
    ----------
    X : numpy.ndarray
        data input array with dimension (n,d)
    no_dims : int
        number of dimensions that PCA reduce to

    Returns
    ----------
    Y : numpy.ndarray 
        low-dimensional representation of input X (n, no_dims) 

    """
    n, d = X.shape
    X = X - X.mean(axis=0)[None, :]
    _, M = np.linalg.eig(np.dot(X.T, X))
    Y = np.real(np.dot(X, M[:, :no_dims]))
    return Y



def compute_pij(Pi_j):
    """
    Compute pij from Pi_j

    Parameters
    ----------
    P : numpy.ndarray
        distance matrix (n,n)
    
    Returns
    -------
    Pij : numpy.ndarray
        pairwise probability matrix (n,n)
    """
    pairwise_sum = Pi_j + Pi_j.T
    Pij = pairwise_sum / np.sum(pairwise_sum)

    return Pij



def compute_qij(Y,
                min_clip = 1e-12):
    """
    Compute pij from Y matrix 

    Parameters
    ----------
    Y : numpy.ndarray
        PCA results of X(n, no_dim)
    min_clip : scalar
        value for a_min in np.clip function

    Returns
    -------
    qij : numpy.ndarray
        Probability matrix of Y (n, n)
    """
    from adjustbeta import distance_matrix

    y_dist = distance_matrix(Y)
    y_dist_matrix = 1 / (1 + y_dist)

    n = Y.shape[0]
    dia = np.diag_indices(n) # indices of diagonal elements
    #dia_sum = sum(y_dist_matrix[dia]) # sum of diagonal elements
    #off_dia_sum = np.sum(y_dist_matrix) - dia_sum

    #qij = y_dist_matrix / off_dia_sum
    
    y_dist_matrix[dia] = 0 # set diagonal as 0 
    qij = y_dist_matrix / np.sum(y_dist_matrix)
    qij = np.clip(qij, a_min = min_clip, a_max = None) # clip the smallest value to 1e-12

    return qij, y_dist

def compute_y_gradient(pij, qij, Y, y_dist):
    """
    Compute pij from Y matrix 

    Parameters
    ----------
    pij : numpy.ndarray
        pair wise probability matrix generated from adjustbeta (n, n)

    qij : numpy.ndarray
        pair wise probability matrix generated from Y (n, n)

    Y : numpy.ndarray
        PCA results of X (n, no_dim)
    
    y_dist : numpy.ndarray
        distance matrix of Y (n, n)
    
    Returns
    -------
    dY : numpy.ndarray
        gradient of Y (n, no_dims)

    """
    n, d = Y.shape
    ## Firstly, calculate (pij - qij)
    pij_qij = pij - qij

    ## Secondly, calculate yi - yj
    yi_yj = np.zeros((n, n, d))
    for ii in range(n):
        yi_yj[ii] = Y[ii,:][None,:]-Y

    ## Thirdly, use the previous y_dist matrix calculate the inverse of ( 1 + y_dist)
    y_dist_matrix = 1 / (1 + y_dist)

    dY = np.sum(pij_qij[:, :, None] * yi_yj * y_dist_matrix[:, :, None], axis = 1)

    return dY


def tsne(X,
         no_dims=2, 
         perplexity=30.0, 
         initial_momentum=0.5, 
         final_momentum=0.8, 
         eta=500, 
         min_gain=0.01, 
         T=1000
         ):
    """
    Master Function for perfoming tsne on high-dimensional data
    
    Parameters
    -----------------------------------
    X : numpy.ndarray
        data input array with dimension(n, d)
    no_dims : int
        dimension of PC to keep 
    perplexity : float
        for calculating beta
    initial momentum : float
        momentum for the first 20 iterations
    final momentum : float
        momentum after the first 20 iterations
    eta : int
        for update deltaY
    min_gain : float
        value to clip the gain 
    T : int
        number of iteration

    Return
    -----------------------------------
    Y : numpy.ndarray 
        low-dimensional representation of input X (n, no_dims)

    """

    from adjustbeta import adjustbeta
    from tqdm import tqdm
    n = X.shape[0]
    # precision(beta) adjustment based on perplexity
    P, beta = adjustbeta(X, tol =  1e-5, perplexity = perplexity)

    # Compute pairwise affinities pij (equation 1 and note 5.1)
    pij = compute_pij(P)

    # Early exaggerate (multiply) p(n n) ij by 4 and clip the value to be at least 1e-12
    pij = 4 * pij
    pij = np.clip(pij, a_min = 1e-12, a_max = None)

    # Initialize low-dimensional data representation Array Y (0) using first no_dims of PCs from PCA
    Y = X[:, : no_dims]
 
    # Initialize  delta_Y (n,no_dims) = 0, gains(n, no_dims) = 1
    delta_Y = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))


    for tt in tqdm(range(T)): 
        # Compute low-dimensional affinities qij (equation 2 and note 5.2) and clip the value to be at least 10e-12
        qij, y_dist = compute_qij(Y)

        # Compute gradient dY (equation 3 and note 5.3)
        dY = compute_y_gradient(pij, qij, Y, y_dist)
        

        if tt < 19:
            momentum = initial_momentum
        else:
            momentum = final_momentum

        
        # Determine gains based on the sign of dY and  delta_Y 
        dY_sign = dY > 0
        deltaY_sign = delta_Y > 0
        gains = (gains + 0.2) * (dY_sign != deltaY_sign) + (gains * 0.8) * (dY_sign == deltaY_sign)
    
        # Clip gains to be at least min gain
        gains = np.clip(gains, a_min = min_gain, a_max = None)

        # calculate delta Y and update Y
        delta_Y = momentum * delta_Y - eta * (gains * dY)
        Y += delta_Y
        
        # remove early exaggeration
        if tt == 99:
            pij /= 4
            
    return Y
 



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D 
    import numpy as np

    no_dims = 2
    print("Run Y = tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    print("Running example on 2,500 MNIST digits...")
    X = np.loadtxt("mnist2500_X.txt")
    X = pca(X, 50)
    labels = np.loadtxt("mnist2500_labels.txt")
    Y = tsne(X, no_dims = no_dims)
    
    if no_dims == 2:
        plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
        scatter = plt.scatter(Y[:, 0], Y[:, 1], c=labels, cmap='tab20', s=20)

        handles, legend_labels = scatter.legend_elements(prop="colors", num=len(np.unique(labels)))
        plt.legend(handles=handles, labels=[str(label) for label in np.unique(labels)], title="Digit",loc='upper left', bbox_to_anchor=(0.95, 1.0))
        plt.savefig(f"mnist_tsne_{no_dims}D.png")
    
    if no_dims == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=labels, cmap='tab20', s=20)
        legend = ax.legend(*scatter.legend_elements(), title="Digit")
        ax.add_artist(legend)

        ax.set_title("3D t-SNE Visualization")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")
        plt.tight_layout()
        plt.savefig(f"mnist_tsne_{no_dims}D.png")
        plt.show()

