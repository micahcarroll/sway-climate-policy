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
    df = pd.read_csv('voting_data_2000.csv')
    current = pd.read_csv('data/initial_data.csv')
    current_names = current['name'].unique()
    df = df[df['name'].isin(current_names)]
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

    col_filter = (df.values.sum(axis=0)/float(df.shape[0])<0.7)
    cols_to_keep = df.columns[col_filter]
    print cols_to_keep
    df = df[cols_to_keep]
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
    # Taken from best and worst scorer League of Conservation Voters.
    bad_guys = ['lucas','cole','nunes','mccarthy','rohrabacher','issa','turner','gohmert','wicker','cochran']
    good_guys = ['thompson','welch','blumenaver','mcnerney','eshoo','velazquez','lowey','pallone','grijalva','pelosi']
    # Read in data
    df = read_data()

    # Format data and take care of missing values
    votes = format_data(df)

    # Experiment with PCA
    pca = PCA(n_components=4)
    reduced = pca.fit_transform(votes)

    # plot_PCA_reduction(reduced[:,:3], with_labels=True, labels=df['party'])

    # New dataframe
    df2 = pd.DataFrame(data=reduced, index=df.index.values)
    locs = reduced

    # Goof guys _ Bad guys
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

    # Find centroids
    good_guys_location = locs[good_guy_idx]
    bad_guys_location = locs[bad_guy_idx]
    good_centroid = find_centroids(good_guys_location)
    bad_centroid = find_centroids(bad_guys_location)

    # Plotting two dimensions
    dimensions = (0,1)
    X = reduced[:,dimensions[0]]
    y = reduced[:,dimensions[1]]
    colours = df['party'].apply(lambda x: 'r' if x=='R' else 'b')
    # Setting colours based on good/bad/normal guy
    k = np.asarray(['b' for i in xrange(len(surnames))])
    k[good_guy_idx] = 'g'
    k[bad_guy_idx] = 'r'
    # Setting marker on good/bad/normal guy
    m = np.asarray(['.' for i in xrange(len(surnames))])
    m[good_guy_idx] = '*'
    m[bad_guy_idx] = '<'
    # Setting size on good/bad/normal guy
    s = np.asarray([20 for i in xrange(len(surnames))])
    s[good_guy_idx] = 50
    s[bad_guy_idx] = 50
    # Plotting!
    for _m, _c, _x, _y, _s in zip(m, colours, X, y, s):
        plt.scatter(_x, _y, marker=_m, c=_c, s=_s)
    plt.scatter(good_centroid[dimensions[0]],good_centroid[dimensions[1]],marker='*',c='g',s=100)
    plt.scatter(bad_centroid[dimensions[0]],bad_centroid[dimensions[1]],marker='<',c='g',s=100)
    plt.show()

    # Compute distances
    comparison = 'nearest' # or 'average'
    if comparison == 'nearest':
        # Comparing to nearest point
        closest_good = distance_to_nearest(good_guys_location,locs)
        closest_bad = distance_to_nearest(bad_guys_location,locs)
        df2['good_distance'] = closest_good
        df2['bad_distance'] = closest_bad
    elif comparison == 'average':
        good_dist = calculate_distance_to_point(good_centroid, locs)
        bad_dist = calculate_distance_to_point(bad_centroid, locs)
        df2['good_distance'] = good_dist
        df2['bad_distance'] = bad_dist

    # norm distances
    distances = df2[['good_distance','bad_distance']].values
    vect_dist = distances.sum(axis=1)
    vect_dist = vect_dist.reshape(len(vect_dist),1).astype(float)
    normed = np.divide(distances,vect_dist.astype(float))
    df2['good_distance_norm'] = normed[:,0]
    df2['bad_distance_norm'] = normed[:,1]

    # Plot!
    y = np.zeros(len(surnames))
    df2['score']= df2['good_distance_norm'] - 0.5
    for _m, _c, _x, _y, _s in zip(m, colours, df2['score'], y, s):
        plt.scatter(_x, _y, marker=_m, c=_c, s=_s)
    plt.show()

    # Output results
    df2['party'] = df['party']
    df2['state'] = df['state']
    df2['district'] = df['district']
    df2.to_csv('results.csv')

    # Calculate top 12 likely to sway!
    df2['scores_sqr'] = df2['score'].apply(lambda x: np.sqrt(x**2))
    sway_targets = df2.iloc[:,[4,5,11,12,-1]].values
    top_twelve = sorted(sway_targets, key=lambda x: x[-1])[:12]
    for person in top_twelve:
        print person

