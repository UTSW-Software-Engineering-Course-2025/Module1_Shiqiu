# import argparse
# import tsne 
# import GradphDR
# import os 
# import sys
# import pandas as pd
import numpy as np
import pandas as pd

def get_args_parser():
    import argparse
    parser = argparse.ArgumentParser('Dimensionality reduction and manifold learning')
    
    # dataset parameters
    parser.add_argument('--dataset', default='sample_data', type=str,
                        help='Choose using sample datasets as a tutorial/demo or customized(real) dataset for analysis')
    parser.add_argument('--dataset_name', default='mnist_digit', type=str,
                        help='Folder name that contains the training data and the label(if there is one)')
    parser.add_argument('--normalize', action="store_true", 
                        help='Folder name that contains the training data and the label(if there is one)')
    parser.add_argument('--data_T', action="store_true", 
                        help='is data needed to do transform')            
    # output parameters
    parser.add_argument('--output_dir', default='output', type=str,
                        help='the output directory, if not customized, plots and results will be saved in the output, folder names will be the time the folder is generated')
   
    
    # method parameters
    parser.add_argument('--method', default='tsne', type=str,
                        help='Options: "tsne", "GraphDR"')
    parser.add_argument('--PCA', default=True, type=bool,
                        help='Perform PCA before the non-linear dimension reduction')
    parser.add_argument('--PCA_dimension', default=50, type=int,
                        help='The number of PCA components')

    ## parameters for tsne
    parser.add_argument('--tsne_no_dims', default=2, type=int,
                        help='tsne parameters: no_dims')
    parser.add_argument('--tsne_perplexity', default=30.0, type=float,
                        help='tsne parameters: perplexity')
    parser.add_argument('--tsne_initial_momentum', default=0.5, type=float,
                        help='tsne parameters: initial_momentum')
    parser.add_argument('--tsne_final_momentum', default=0.8, type=float,
                        help='tsne parameters: final_momentum')
    parser.add_argument('--tsne_eta', default=500, type=float,
                        help='tsne parameters: eta')
    parser.add_argument('--tsne_min_gain', default=0.01, type=float,
                        help='tsne parameters: main_gain')
    parser.add_argument('--tsne_T', default=1000, type=float,
                        help='tsne parameters: T')

    ## parameters for GraphDR
    parser.add_argument('--GraphDR_lambda_', default=1., type=float,
                        help='GraphDR parameters: lambda')
    parser.add_argument('--GraphDR_no_rotation', default=True, type=bool, 
                        help='GraphDR parameters: no_rotation')
    parser.add_argument('--GraphDR_n_neighbor', default=10, type=int,
                        help='GraphDR parameters: n_neighbor')
    parser.add_argument('--GraphDR_top_d_eigenvector', default=10, type=int,
                        help='GraphDR parameters: top_d_eigenvector')


    return parser



def load_dataset(data_path):
    """ given a data path set"""

    if data_path.endswith(".gz"):
        data = pd.read_csv(data_path,sep='\t',index_col=0)
    elif data_path.endswith(".txt"):
        data = np.loadtxt(data_path)
    elif data_path.endswith(".npy"):
        data = np.load(data_path)
    # else:
    #     try:
    #         data = np.loadtxt(data_path)
    #     except:
    #         data = np.load
    return data

def load_label(label_path):
    try:
        anno = pd.read_csv(label_path,sep='\t',header=None)
        anno = anno[1].values
    except:
        anno = np.loadtxt(label_path)
    return anno

def main():
    import os 
    import sys

    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    sys.path.append("../src")
    from tsne import tsne 
    from GraphDR import graphdr, preprocess_data
    import logging

     

    args_parser = get_args_parser()
    args = args_parser.parse_args()

    ## create output dir
    output_dir = os.path.join("../output/", args.dataset_name)
    os.makedirs(output_dir, exist_ok = True)

    logging.basicConfig(
        level=logging.INFO,  # Set the level to DEBUG, INFO, WARNING, ERROR, or CRITICAL
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=f'../output/{args.dataset_name}/cli.log', 
        filemode='w'        
    )


    ## load dataset and label
    dataset_path = os.path.join("../data/",args.dataset, args.dataset_name, 'training_data')
    label_path = os.path.join("../data/",args.dataset, args.dataset_name, 'label')

    for f in  os.listdir(dataset_path):
        if f.endswith(".npy") or f.endswith(".txt") or f.endswith(".gz"):
            file_name = os.path.join(dataset_path, f)
        
    logging.info("loading dataset from here: %s", file_name)
    data = load_dataset(file_name) ## assume there's only one dataset inside each folder
    
    if len(os.listdir(label_path)) > 0:
        label = load_label(os.path.join(label_path, os.listdir(label_path)[0]))
    else:
        label = None
  


    


    ## normalize data
    if args.normalize:
        data = preprocess_data(data)    
    
    if args.data_T:
        data = data.T
    
    ## perform PCA
    if args.PCA:
        pca = PCA(n_components = args.PCA_dimension)
        pca.fit(data)
        pca_data = pca.transform(data)
        X = pca_data
    else:
        X = data

    logging.info("Sanity Check for X shape: %s", X.shape)

    if args.method == 'tsne':
        paras = {key.replace('tsne_', ""): val for key, val in vars(args).items() if key.startswith("tsne_")}
        Y = tsne(X, **paras)

    elif args.method == 'GraphDR':
        paras = {key.replace('GraphDR_', ""): val for key, val in vars(args).items() if key.startswith("GraphDR_")}
        Y = graphdr(X, **paras)

    else:
        assert "methods has not been developed yet, leave a comment and let us know which method you're interested!"


    np.save(os.path.join(output_dir, f"PCA_{args.PCA_dimension}d.npy"), pca_data)
    np.save(os.path.join(output_dir, f"{args.method}_results.npy"), Y)
    # pca_data.save(os.path.join(output_dir, f"PCA_{args.PCA_dimension}d.npy"))
    # Y.save(os.path.join(output_dir, f"{args.method}_.npy"))
        



    ## plot and save the plots of PCA
    plt.figure(figsize=(15,10))
    sns.scatterplot(x=pca_data[:,0], y=pca_data[:,1], linewidth = 0, s=5, hue=label)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig(os.path.join(output_dir,'after_pca.svg'))

    ## plot and save the plots of dimension reduction and manifold learning
    if Y.shape[1] <= 2:
        
        plt.scatter(Y[:, 0], Y[:, 1], 20, label)
        scatter = plt.scatter(Y[:, 0], Y[:, 1], c=label, cmap='tab20', s=20)
        handles, legend_labels = scatter.legend_elements(prop="colors", num=len(np.unique(label)))
        plt.legend(handles=handles, labels=[str(label) for label in np.unique(label)], title="Digit",loc='upper left', bbox_to_anchor=(0.95, 1.0))
        plt.savefig(os.path.join(output_dir, f"mnist_{args.method}_2d.svg"))



    if pca_data.shape[1] > 2:
        fig = px.scatter_3d(x=pca_data[:,0], y=pca_data[:,1], z=pca_data[:,2],color=label,opacity = 0.5)
        fig.update_traces(marker_size=2.5)
        fig.write_html(os.path.join(output_dir,"3dscatter_plot_PCA.html"))
        fig.show()

    if Y.shape[1] > 2:
        fig = px.scatter_3d(x=Y[:,0], y=Y[:,1], z=Y[:,2],color=label,opacity = 0.5)
        fig.update_traces(marker_size=2.5)
        fig.write_html(os.path.join(output_dir,f"3dscatter_plot_{args.method}.html"))
        fig.show()



    ## save all the args into log
    for key, val in vars(args).items():
        logging.info(f"{key}: {val}")
    logging.info("All results and plots have been sucessfully saved!")







if __name__ == '__main__':
    main()