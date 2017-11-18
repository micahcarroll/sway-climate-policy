import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score, recall_score, f1_score, precision_score, accuracy_score
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA

def read_data():
    df = pd.read_csv('data/initial_data.csv')
    df = df.set_index('name')
    return df

def format_data(df):
    df = df.iloc[:,6:]
    for col in df.columns:
        temp = df[col].apply(lambda x: x if x in ['Yes','No','Not Voting'] else 0)
        temp = np.where(df[col] == 'Yes',1,temp)
        temp = np.where(df[col] == 'No',-1,temp)
        temp = np.where(df[col] == 'Not Voting',0,temp)
        df[col] = temp
        df[col].fillna(0)
    return df

def plot_PCA_reduction(X, with_labels=False, labels=None):
    fig = plt.figure(1, figsize=(12, 12))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    plt.cla()
    if with_labels:
        for label in labels.unique():
            ax.text3D(X[labels == label, 0].mean(),
                      X[labels == label, 1].mean() + 1.5,
                      X[labels == label, 2].mean(), str(label),
                      horizontalalignment='center',
                      bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
        colours = np.where(labels==labels.unique()[0],'r','b')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2],c=colours, cmap=plt.cm.spectral,
                   edgecolor='k')
    else:
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], cmap=plt.cm.spectral,
                   edgecolor='k')
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    plt.show()
    pass


if __name__ == '__main__':
    # Read in data
    df = read_data()
    # Format data and take care of missing values
    votes = format_data(df)
    # Experiment with PCA
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(votes)
    # plot_PCA_reduction(reduced, with_labels=True, labels=df['party'])
    # Find axis with D + R split = axis 0 as expdcted
    # Project onto remainig axis
    X = reduced[:,1]
    y = reduced[:,2]
    colours = df['party'].apply(lambda x: 'r' if x=='R' else 'b')
    plt.scatter(X, y , c=colours)
    plt.show()
    # Identify TOP climate hawk
    ## Calculate typical climmate hawk vector
    # Identify TOP climate denier
    ## Calculate typical climate denier
