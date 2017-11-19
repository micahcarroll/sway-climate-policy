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

def find_centroids(data):
    return data.mean(axis=0)

def calculate_distance_to_point(point, data):
    dist_sqr = np.power(data - point,2)
    summed = dist_sqr.sum(axis=1)
    dist = np.sqrt(summed)
    return dist

def distance_to_nearest(points, data):
    s_distances = []
    for i in xrange(data.shape[0]):
        d = data[i,:]
        distances = calculate_distance_to_point(points, d)
        # arg_shorted = distances.argmin()
        shortest_distance =distances.min()
        s_distances.append(shortest_distance)
    s_distances = np.asarray(s_distances)
    return s_distances




if __name__ == '__main__':
    bad_guys = ['lucas','cole','nunes','mccarthy','rohrabacher','issa','turner','gohmert','wicker','cochran']
    good_guys = ['thompson','welch','blumenaver','mcnerney','eshoo','velazquez','lowey','pallone','grijalva','pelosi']
    # Read in data
    df = read_data()
    # Format data and take care of missing values
    votes = format_data(df)
    # Experiment with PCA
    pca = PCA(n_components=4)
    reduced = pca.fit_transform(votes)
        # plot_PCA_reduction(reduced, with_labels=True, labels=df['party'])
    # Find axis with D + R split = axis 0 as expdcted
    # Project onto remainig axis
    X = reduced[:,0]
    y = reduced[:,2]
    colours = df['party'].apply(lambda x: 'r' if x=='R' else 'b')
    # plt.scatter(X, y , c=colours)
    # plt.show()
    # Identify TOP climate hawk
    df2 = pd.DataFrame(data=reduced, index=df.index.values)
    surnames = []
    firstnames = []
    for name in df2.index.values:
        split=name.split(' ')
        firstnames.append(split[0].lower())
        surnames.append(split[-1].lower())
    df2['surname'] = surnames
    df2['firstname'] = firstnames
    good_guy_idx = []
    for h in good_guys:
        if h in surnames:
            idx = surnames.index(h)
            good_guy_idx.append(idx)
    bad_guy_idx = []
    for h in bad_guys:
        if h in surnames:
            idx = surnames.index(h)
            bad_guy_idx.append(idx)
    k = np.asarray(['b' for i in xrange(len(surnames))])
    k[good_guy_idx] = 'g'
    k[bad_guy_idx] = 'r'

    m = np.asarray(['.' for i in xrange(len(surnames))])
    m[good_guy_idx] = '*'
    m[bad_guy_idx] = '<'

    s = np.asarray([20 for i in xrange(len(surnames))])
    s[good_guy_idx] = 50
    s[bad_guy_idx] = 50
    # plt.scatter(X, y , marker=m,c=k)
    for _m, _c, _x, _y, _s in zip(m, colours, X, y, s):
        # plt.figure(figsize=(15,15))
        plt.scatter(_x, _y, marker=_m, c=_c, s=_s)
    # find cluster centroids
    aoi = df2.iloc[:,[0,2]].values
    good_centroid = find_centroids(aoi[good_guy_idx])
    bad_centroid = find_centroids(aoi[bad_guy_idx])
    plt.scatter(good_centroid[0],good_centroid[1],marker='*',c='g',s=100)
    plt.scatter(bad_centroid[0],bad_centroid[1],marker='<',c='g',s=100)
    plt.show()

    good_guys_location = df2.iloc[:,[0,2]].values[good_guy_idx]
    bad_guys_location = df2.iloc[:,[0,2]].values[bad_guy_idx]

    # Comparing to centroid
    # good_dist = calculate_distance_to_point(good_centroid, df2.iloc[:,[0,2]].values)
    # bad_dist = calculate_distance_to_point(bad_centroid, df2.iloc[:,[0,2]].values)
    # df2['good_distance'] = good_dist
    # df2['bad_distance'] = bad_dist
    # distances = df2.iloc[:,-2:].values
    # vect_dist = distances.sum(axis=1)
    # vect_dist = vect_dist.reshape(len(vect_dist),1).astype(float)
    # normed = np.divide(distances,vect_dist.astype(float))
    # df2['good_distance_norm'] = normed[:,0]
    # df2['bad_distance_norm'] = normed[:,1]
    #
    # y = np.zeros(len(surnames))
    # X = df2['good_distance_norm'] - 0.5
    # for _m, _c, _x, _y, _s in zip(m, colours, X, y, s):
    #     plt.scatter(_x, _y, marker=_m, c=_c, s=_s)
    # # plt.show()

    # Comparing to nearest point
    closest_good = distance_to_nearest(good_guys_location,df2.iloc[:,[0,2]].values)
    closest_bad = distance_to_nearest(bad_guys_location,df2.iloc[:,[0,2]].values)
    df2['good_closest'] = closest_good
    df2['bad_closest'] = closest_bad
    distances = df2.iloc[:,-2:].values
    vect_dist = distances.sum(axis=1)
    vect_dist = vect_dist.reshape(len(vect_dist),1).astype(float)
    normed = np.divide(distances,vect_dist.astype(float))
    df2['good_closest_norm'] = normed[:,0]
    df2['bad_closest_norm'] = normed[:,1]
    y = np.zeros(len(surnames))
    df2['score']= df2['good_closest_norm'] - 0.5
    for _m, _c, _x, _y, _s in zip(m, colours, df2['score'], y, s):
        # plt.figure(figsize=(15,15))
        plt.scatter(_x, _y, marker=_m, c=_c, s=_s)
    plt.show()
    df2['party'] = df['party']
    df2['state'] = df['state']
    df2['district'] = df['district']
    df2.to_csv('results.csv')

    # df2['scores_sqr'] = df2['scores'].apply(lambda x: x**2)
    # 
