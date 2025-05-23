"""
Implements GraphDR, a graph-regularized dimensionality reduction method for single-cell or high-dimensional data.

"""


import pandas as pd
import numpy as np

def graphdr(pca_data, 
            lambda_ = 1, 
            no_rotation=True,
            n_neighbor = 10,
            top_d_eigenvector = 10):
    """
    Main Function of GraphDR mehtod

    Parameters
    ----------
    pca_data : numpy.ndarray
        PCA results of the original data (n, d)
    lambda_ : float
        Parameters to control the regularization on Laplacian matrix
    no_rotation : bool
        Control if the final results need to be rotated based on the eigenvector
    n_neighbor : int
        Number of neighbor when constructing neighbor graph
    top_d_eigenvector : int
        number of eigenvectors to choose when performing rotation

    Returns
    -------
    Z : numpy.ndarray
        The final graph-regularized dimension reduction results;(n, top_d_eigenvector) if no_rotation False; (n, d) else

    """
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import laplacian
    from scipy.sparse import eye
    from numpy.linalg import inv
  
    n, d = pca_data.shape
    
    X = pca_data
    I = eye(n).toarray()
    G = kneighbors_graph(X, n_neighbors = n_neighbor).toarray()
    L = laplacian(G, symmetrized=True)  # L must be symmetrical here

    K =  inv(I + lambda_ * L)
    
    if no_rotation:
        W = eye(d).toarray()
    else:
        W_ = X.T @ K @ X
        _, W = np.linalg.eig(W_)
        W = W[:,:top_d_eigenvector]

    Z = K @ X @ W

    return Z

def preprocess_data(data):
    """ 
    Preprocess and normalize the single cell data

    Parameters
    ----------
    data : numpy.ndarray
        The original single cell or sequencing data (n, d)

    Returns
    -------
    preprocessed_data : numpy.ndarray
        Data after normalization for each cell (n, d)

    """
    #import pandas as pd
    #data = pd.read_csv(data_path, sep='\t',index_col=0)

    #We will first normalize each cell by total count per cell.
    percell_sum = data.sum(axis=0)
    pergene_sum = data.sum(axis=1)

    preprocessed_data = data / percell_sum.values[None, :] * np.median(percell_sum)
    preprocessed_data = preprocessed_data.values

    #transform the preprocessed_data array by `x := log (1+x)`
    preprocessed_data = np.log(1 + preprocessed_data)

    #standard scaling
     
    
    return preprocessed_data


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.decomposition import PCA
    import plotly.express as px

    preprocessed_data = preprocess_data('./hochgerner_2018.data.gz')
    anno = pd.read_csv('./hochgerner_2018.anno',sep='\t',header=None)
    anno = anno[1].values

    #preprocess by PCA 
    pca = PCA(n_components = 20)
    pca.fit(preprocessed_data.T)
    pca_data = pca.transform(preprocessed_data.T)

    #visualize PCA result
    plt.figure(figsize=(15,10))
    sns.scatterplot(x=pca_data[:,0], y=pca_data[:,1], linewidth = 0, s=5, hue=anno)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig('after_pca.svg')
    #plt.show()
    # use plotly to visualize in 3d
    graphdr_data = graphdr(pca_data, lambda_= 5, no_rotation=True, n_neighbor = 10, top_d_eigenvector = 5)
    #compare your graphdr output with this one, it should look similar (remember to adjust lambda_ and see its effect)

    plt.figure(figsize=(15,10))
    sns.scatterplot(x=graphdr_data[:,0], y=graphdr_data[:,1], linewidth = 0, s=3, hue=anno)
    plt.xlabel('GraphDR 1')
    plt.ylabel('GraphDR 2')
    plt.savefig('after_GraphDR.svg')
    #plt.show()

    fig = px.scatter_3d(x=graphdr_data[:,0], y=graphdr_data[:,1], z=graphdr_data[:,2],color=anno,opacity = 0.5)
    fig.update_traces(marker_size=2.5)
    fig.write_html("3dscatter_plot.html")
    fig.show()

    

