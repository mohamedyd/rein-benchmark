import matplotlib.pyplot as plt
import numpy as np
import os

def plot_barchart(feature_names, weights, svd_id, column_name):
    N = len(feature_names)

    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()

    colors = []
    for w in range(len(weights)):
        if weights[w] > 0.0:
            colors.append('b')
        else:
            colors.append('r')

    rects1 = ax.bar(ind, weights, width, color=colors)

    ax.axhline(0, color='black')
    #ax.axvline(0, color='white')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Weights')
    ax.set_title('Weights for svd '+ str(svd_id) + ' of column ' + str(column_name))
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(feature_names)

def visualize_svd(svd_components, data, vocabulary, column_id):
    number_components = len(svd_components)
    number_features = len(svd_components[0])

    column_name = data.clean_pd.columns[column_id]

    inv_vocabulary = {v: k for k, v in vocabulary.iteritems()}

    for c in range(number_components):
        feature_names = []
        weight = []
        for f in range(number_features):
            feature_names.append(inv_vocabulary[f])
            weight.append(svd_components[c][f])
        plot_barchart(feature_names, weight, c, column_name)

        directory = '/home/felix/SequentialPatternErrorDetection/html/' + data.name + '/img'
        if not os.path.exists(directory):
            os.makedirs(directory)
        path = directory + '/' + str(column_name) + '_svd_' + str(c) + '.png'

        plt.savefig(path)


def replace_with_url(html, data):

    html_new = ''
    import re
    tokens = filter(None, re.split("[\n\t]+", html))

    directory = './img'

    for tok in tokens:
        found = False
        for column_name in data.clean_pd.columns:
            search = column_name + '_svd_'
            if search in tok:
                pos = tok.index(search) + len(search)
                c = int(tok[pos:])

                path = directory + '/' + str(column_name) + '_svd_' + str(c) + '.png'
                html_new += '<a href="' + path + '"> ' + tok + ' </a>'
                found = True
                break
        if not found:
            html_new += tok

    return html_new




