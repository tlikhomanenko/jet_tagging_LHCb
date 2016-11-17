from __future__ import print_function, division
import numpy
import pandas
from matplotlib import pyplot as plt
import matplotlib
from matplotlib import cm
from sklearn.metrics import roc_auc_score
from collections import OrderedDict
from rep.utils import get_efficiencies
from rep.plotting import ErrorPlot
from rep.utils import weighted_quantile
from sklearn.metrics import roc_curve, roc_auc_score
from collections import defaultdict




labels_names_correspondence = {0: "b jets", 1:"c jets", 2: "light jets"}
labels_names_correspondence = OrderedDict(sorted(labels_names_correspondence.items()))
names_labels_correspondence = OrderedDict(map(lambda (x, y): (y, x), labels_names_correspondence.items()))

def names_labels_correspondence_update(new_labels_names_correspondence):

    labels_names_correspondence = new_labels_names_correspondence
    labels_names_correspondence = OrderedDict(sorted(labels_names_correspondence.items()))
    names_labels_correspondence = OrderedDict(map(lambda (x, y): (y, x), labels_names_correspondence.items()))



def shrink_floats(data):
    for column in data.columns:
        if data[column].dtype == 'float64':
            data[column] = data[column].astype('float32')

def compute_weights(labels):
    """
    Compute weight (sum of weights for each class are the same - balanced data).

    Parameters
    ----------
    labels : array_like
        Label values of samples.

    Return
    ------
    weights : array_like
        Weight of the each sample.
    """

    weights = numpy.ones(len(labels))
    for label in numpy.unique(labels):
        weights[labels == label] = 1. / sum(labels == label)
    weights /= numpy.mean(weights) + 1e-10
    return weights


def roc_auc_score_one_vs_all(labels, pred, sample_weight):
    """
    Compute ROC AUC values for (one vs rest).
    
    :param array labels: labels (from 0 to 5)
    :param array pred: 1d to use it for each class, or ndim: each column corresponds to only one class
    :param array sample_weight: weights
    :return: pandas.DataFrame with ROC AUC values for each class
    """
    rocs = OrderedDict()
    if len(pred.shape) == 1:
        pred = numpy.vstack([pred] * len(names_labels_correspondence.keys())).T
    for key, label in names_labels_correspondence.items():
        rocs[key] = [roc_auc_score(labels == label, pred[:, label], sample_weight=sample_weight)]
    return pandas.DataFrame(rocs)


def roc_auc_score_one_vs_all_for_separate_algorithms(labels, pred, sample_weight):
    """
    Compute ROC AUC values for (one vs rest).
    
    :param array labels: labels (from 0 to 5)
    :param dict pred: predcitions for ech label to be signal
    :param array sample_weight: weights
    :return: pandas.DataFrame with ROC AUC values for each class
    """
    rocs = OrderedDict()
    for key, label in names_labels_correspondence.items():
        rocs[key] = [roc_auc_score(labels == label, pred[label], sample_weight=sample_weight)]
    return pandas.DataFrame(rocs)


def compute_cum_sum(data, features, prefix_name="", scale=False):
    """
    Compute cumulative sum for features from starting with the first feature.
    
    :param pandas.DataFrame data: data 
    :param list features: features
    :param str prefix_name: prefix for produced features names
    :param bool scale: scale or not feature before adding to cumulative sum
    :return: pandas.DataFrame new features
    """
    cum_sum = numpy.zeros(len(data))
    cum_features = {}
    for n, f in enumerate(features):
        column = data[f].values
        if scale:
            temp = (column - column.mean()) / numpy.sqrt(column.var())
        cum_sum += column
        cum_features[prefix_name + str(n)] = cum_sum.copy()
    return pandas.DataFrame(cum_features, index=None)


def convert_DLL_to_LL(data, features):
    """
    Compute Likelihood for each particle from the DLL=Likelihood_particle - Likelihood_pion. We assume that probabilities are sum up to 1. Actually each probability is computed independently and they should not be summed up to 1.
    
    :param pandas.DataFrame data: data with DLL features
    :param list features: DLL features
    :return: pandas.DataFrame with features names + '_LL' 
    """
    temp_data = data[features].values
    temp_data -= temp_data.max(axis=1, keepdims=True)
    temp_data = numpy.exp(temp_data)
    temp_data /= numpy.sum(temp_data, axis=1, keepdims=True)
    return pandas.DataFrame(numpy.log(numpy.clip(temp_data, 1e-6, 10)), columns=map(lambda x: x + '_LL', features))


def plot_hist_features(data, labels, features, bins=30, ignored_sideband=0.01):
    """
    Plot histogram of features with values > -500.
    
    :param pandas.DataFrame data: data with features
    :param array labels: labels (from 0 to 5)
    :param list features: plotted features
    """
    labels = numpy.array(labels)
    for n, f in enumerate(features):
        plt.subplot(int(numpy.ceil(len(features) / 6)), min(6, len(features)), n+1)
        temp_values = data[f].values
        temp_labels = numpy.array(labels)[temp_values != -999]
        temp_values = temp_values[temp_values != -999]
        v_min, v_max = numpy.percentile(temp_values, [ignored_sideband * 100, (1. - ignored_sideband) * 100])
        for key, val in names_labels_correspondence.items():  
            plt.hist(temp_values[temp_labels == val], label=key, alpha=0.2, normed=True, bins=bins, range=(v_min, v_max))
        plt.legend(loc='best')
        plt.title(f)
        
        
def __rolling_window(data, window_size):
    """
    Rolling window: take window with definite size through the array

    :param data: array-like
    :param window_size: size
    :return: the sequence of windows

    Example: data = array(1, 2, 3, 4, 5, 6), window_size = 4
        Then this function return array(array(1, 2, 3, 4), array(2, 3, 4, 5), array(3, 4, 5, 6))
    """
    shape = data.shape[:-1] + (data.shape[-1] - window_size + 1, window_size)
    strides = data.strides + (data.strides[-1],)
    return numpy.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def __cvm(subindices, total_events):
    """
    Compute Cramer-von Mises metric.
    Compared two distributions, where first is subset of second one.
    Assuming that second is ordered by ascending

    :param subindices: indices of events which will be associated with the first distribution
    :param total_events: count of events in the second distribution
    :return: cvm metric
    """
    # here we compute the same expression (using algebraic expressions for them).
    n_subindices = float(len(subindices))
    subindices = numpy.array([0] + sorted(subindices) + [total_events], dtype='int')
    # via sum of the first squares
    summand1 = total_events * (total_events + 1) * (total_events + 0.5) / 3. / (total_events ** 3)
    left_positions = subindices[:-1]
    right_positions = subindices[1:]

    values = numpy.arange(len(subindices) - 1)

    summand2 = values * (right_positions * (right_positions + 1) - left_positions * (left_positions + 1)) / 2
    summand2 = summand2.sum() * 1. / (n_subindices * total_events * total_events)

    summand3 = (right_positions - left_positions) * values ** 2
    summand3 = summand3.sum() * 1. / (n_subindices * n_subindices * total_events)

    return summand1 + summand3 - 2 * summand2


def compute_cvm(predictions, masses, n_neighbours=200, step=50):
    """
    Computing Cramer-von Mises (cvm) metric on background events: take average of cvms calculated for each mass bin.
    In each mass bin global prediction's cdf is compared to prediction's cdf in mass bin.

    :param predictions: array-like, predictions
    :param masses: array-like, in case of Kaggle tau23mu this is reconstructed mass
    :param n_neighbours: count of neighbours for event to define mass bin
    :param step: step through sorted mass-array to define next center of bin
    :return: average cvm value
    """
    predictions = numpy.array(predictions)
    masses = numpy.array(masses)
    assert len(predictions) == len(masses)

    # First, reorder by masses
    predictions = predictions[numpy.argsort(masses)]

    # Second, replace probabilities with order of probability among other events
    predictions = numpy.argsort(numpy.argsort(predictions))

    # Now, each window forms a group, and we can compute contribution of each group to CvM
    cvms = []
    for window in __rolling_window(predictions, window_size=n_neighbours)[::step]:
        cvms.append(__cvm(subindices=window, total_events=len(predictions)))
    return numpy.mean(cvms)


def plot_roc_one_vs_rest(labels, predictions_dict, weights=None, physics_notion=False, predictions_dict_comparison=None, separate_particles=False, algorithms_name=('MVA', 'baseline')):
    """
    Plot roc curves one versus rest.
    
    :param array labels: labels form 0 to 5
    :param dict(array) predictions_dict: dict of label/predictions
    :param array weights: sample weights
    """
    if separate_particles:
        plt.figure(figsize=(22, 22))
    else:
        plt.figure(figsize=(10, 8))
    for label, name in labels_names_correspondence.items():
        if separate_particles:
            plt.subplot(3, 2, label + 1)
        for preds, prefix in zip([predictions_dict, predictions_dict_comparison], algorithms_name):
            if preds is None:
                continue
            fpr, tpr, _ = roc_curve(labels == label, preds[label], sample_weight=weights)
            auc = roc_auc_score(labels == label, preds[label], sample_weight=weights)
            if physics_notion:
                plt.plot(tpr * 100, fpr * 100, label='{}, {}, AUC={:1.5f}'.format(prefix, name, auc), linewidth=2)
                plt.yscale('log', nonposy='clip')
            else:
                plt.plot(tpr, 1-fpr, label='{}, AUC={:1.5f}'.format(name, auc), linewidth=2)
        if physics_notion:
            plt.xlabel('Efficiency', fontsize=22)
            plt.ylabel('Overall MisID Efficiency', fontsize=22)
        else:
            plt.xlabel('Signal efficiency', fontsize=22)
            plt.ylabel('Background rejection', fontsize=22)
        plt.legend(loc='best', fontsize=18)
    
    
def plot_roc_one_vs_one(labels, predictions_dict, weights=None):
    """
    Plot roc curves one versus one.
    
    :param array labels: labels form 0 to 5
    :param dict(array) predictions_dict: dict of label/predictions
    :param array weights: sample weights
    """
    plt.figure(figsize=(22, 24))
    for label, name in labels_names_correspondence.items():
        plt.subplot(3, 2, label + 1)
        for label_vs, name_vs in labels_names_correspondence.items():
            if label == label_vs:
                continue
            mask = (labels == label) | (labels == label_vs)
            fpr, tpr, _ = roc_curve(labels[mask] == label, predictions_dict[label][mask], 
                                    sample_weight=weights if weights is None else weights[mask])
            auc = roc_auc_score(labels[mask] == label, predictions_dict[label][mask],
                                sample_weight=weights if weights is None else weights[mask])
            plt.plot(tpr, 1-fpr, label='{} vs {}, AUC={:1.5f}'.format(name, name_vs, auc), linewidth=2)
        plt.xlabel('Signal efficiency', fontsize=22)
        plt.ylabel('Background rejection', fontsize=22)
        plt.legend(loc='best', fontsize=18)
        
        
def compute_roc_auc_matrix(labels, predictions_dict, weights=None):
    """
    Calculate class vs class roc aucs matrix.
    
    :param array labels: labels form 0 to 5
    :param dict(array) predictions_dict: dict of label/predictions
    :param array weights: sample weights
    """

    # Calculate roc_auc_matrices

    roc_auc_matrices = numpy.ones(shape=[len(labels_names_correspondence)] * 2)
    for label, name in labels_names_correspondence.items():
        for label_vs, name_vs in labels_names_correspondence.items():
            if label == label_vs:
                continue
            mask = (labels == label) | (labels == label_vs)
            roc_auc_matrices[label, label_vs] = roc_auc_score(labels[mask] == label, predictions_dict[label][mask],
                                                              sample_weight=weights if weights is None else weights[mask])
        
    matrix = pandas.DataFrame(roc_auc_matrices, columns=names_labels_correspondence.keys(),
                              index=names_labels_correspondence.keys())

    fig=plot_matrix(matrix)
    return fig, matrix


def plot_matrix(matrix, vmin=0.8, vmax=1., title='Particle vs particle ROC AUCs', fmt='.5f'):
    # Plot roc_auc_matrices
    inline_rc = dict(matplotlib.rcParams)
    
    import seaborn as sns
    fig = plt.figure(figsize=(10,7))
    sns.set()
    ax = plt.axes()
    sns.heatmap(matrix, vmin=vmin, vmax=vmax, annot=True, fmt=fmt, ax=ax, cmap=cm.coolwarm)
    plt.title(title, size=15)
    plt.xticks(size=15)
    plt.yticks(size=15)
    
    plt.show()
    plt.clf()
    plt.close()
    
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    matplotlib.rcParams.update(inline_rc)
    return fig


def plot_flatness_by_particle(labels, predictions_dict, spectator, spectator_name, predictions_dict_comparison=None,
                              names_algorithms=['MVA', 'Baseline'],
                              weights=None, bins_number=30, ignored_sideband=0.1, 
                              thresholds=None, cuts_values=False, ncol=1):
    plt.figure(figsize=(22, 20))
    for n, (name, label) in enumerate(names_labels_correspondence.items()):
        plt.subplot(3, 2, n + 1)
        mask =labels == label
        legends = []
        for preds, name_algo in zip([predictions_dict, predictions_dict_comparison], names_algorithms):
            if preds is None:
                continue
            probs = preds[label][mask]
            if cuts_values:
                thresholds_values = cut_values
            else:
                thresholds_values = [weighted_quantile(probs, quantiles=1 - eff / 100., 
                                                       sample_weight=None if weights is None else weights[mask])
                                     for eff in thresholds]
            eff = get_efficiencies(probs, spectator[mask], 
                                   sample_weight=None if weights is None else weights[mask], 
                                   bins_number=bins_number, errors=True, ignored_sideband=ignored_sideband,
                                   thresholds=thresholds_values)
            for thr in thresholds_values:
                eff[thr] = (eff[thr][0], 100*numpy.array(eff[thr][1]), 100*numpy.array(eff[thr][2]), eff[thr][3])
            plot_fig = ErrorPlot(eff)
            plot_fig.xlabel = '{} {}'.format(name, spectator_name)
            plot_fig.ylabel = 'Efficiency'
            plot_fig.title = name
            plot_fig.ylim = (0, 100)
            plot_fig.plot(fontsize=22)
            plt.xticks(fontsize=12), plt.yticks(fontsize=12)
            legends.append(['{} Eff {}%'.format(thr, name_algo) for thr in thresholds])
        plt.legend(numpy.concatenate(legends), loc='best', fontsize=12, framealpha=0.5, ncol=ncol)

            
def plot_flatness_particle(labels, predictions_dict, spectator, spectator_name, particle_name, 
                           weights=None, bins_number=30, ignored_sideband=0.1, 
                           thresholds=None, cuts_values=False):
    plt.figure(figsize=(18, 22))
    for n, (name, label) in enumerate(names_labels_correspondence.items()):
        plt.subplot(3, 2, n + 1)
        mask = labels == names_labels_correspondence[particle_name]
        probs = predictions_dict[label][mask]
        mask_signal = labels == label
        probs_signal = predictions_dict[label][mask_signal]
        if cuts_values:
            thresholds_values = cut_values
        else:
            thresholds_values = [weighted_quantile(probs_signal, quantiles=1 - eff / 100., 
                                                   sample_weight=None if weights is None else weights[mask_signal])
                                 for eff in thresholds]
        eff = get_efficiencies(probs, spectator[mask], 
                               sample_weight=None if weights is None else weights[mask], 
                               bins_number=bins_number, errors=True, ignored_sideband=ignored_sideband,
                               thresholds=thresholds_values)
        for thr in thresholds_values:
            eff[thr] = (eff[thr][0], 100*numpy.array(eff[thr][1]), 100*numpy.array(eff[thr][2]), eff[thr][3])
        plot_fig = ErrorPlot(eff)
        plot_fig.xlabel = '{} {}'.format(particle_name, spectator_name)
        plot_fig.ylabel = 'Efficiency'
        plot_fig.title = 'MVA {}'.format(name)
        plot_fig.ylim = (0, 100)
        plot_fig.plot(fontsize=22)
        plt.xticks(fontsize=12), plt.yticks(fontsize=12)
        if not cuts_values:
            plt.legend(['Signal Eff {}%'.format(thr) for thr in thresholds], loc='best', fontsize=18, framealpha=0.5)

    
def compute_cvm_by_particle(labels, predictions_dict, spectators):
    cvm_values = defaultdict(list)
    for spectator_name, spectator in spectators.items():
        for n, (name, label) in enumerate(names_labels_correspondence.items()):
            mask =labels == label
            probs = predictions_dict[label][mask]
            cvm_values[spectator_name].append(compute_cvm(probs, spectator[mask]))
    return pandas.DataFrame(cvm_values, index=names_labels_correspondence.keys())


import os


def generate_plots(preds, labels, weights, data, path=''):
    matrix_auc_one_vs_rest = roc_auc_score_one_vs_all_for_separate_algorithms(labels, preds, weights)
    print (matrix_auc_one_vs_rest)

    plot_roc_one_vs_rest(labels, preds, weights)
    plt.savefig(os.path.join(path, 'overall_roc_auc.png'), format='png')

    f, matrix_auc_one_vs_one = compute_roc_auc_matrix(labels, preds, weights)
    f.savefig(os.path.join(path, 'class_vs_class_roc_auc_matrix.png'), format='png')

    matrix_auc_one_vs_rest.to_csv(os.path.join(path, 'class_vs_rest_roc_auc_matrix.csv'))
    matrix_auc_one_vs_one.to_csv(os.path.join(path, 'class_vs_class_roc_auc_matrix.csv'))

    plot_roc_one_vs_one(labels, preds, weights)
    plt.savefig(os.path.join(path, 'one_vs_one_roc_auc.png'), format='png')

    # plot_flatness_by_particle(labels, preds, 1 / data.TrackP.values,
    #                           '1/(Momentum MeV/$c$)', thresholds=[5, 20, 40, 60, 80], weights=weights,
    #                           ignored_sideband=0.02)
    # plt.savefig(os.path.join(path, 'p_flatness.png'), format='png')

    plot_flatness_by_particle(labels, preds, 1 / data.JetPT.values,
                              '1/(Transverse Momentum MeV/$c$)', thresholds=[5, 20, 40, 60, 80], weights=weights,
                              ignored_sideband=0.02)
    plt.savefig(os.path.join(path, 'pt_flatness.png'), format='png')


    plot_flatness_by_particle(labels, preds, data.JetEta.values,
                              'Pseudo Rapidity', thresholds=[5, 20, 40, 60, 80], weights=weights,
                              ignored_sideband=0.02)
    plt.savefig(os.path.join(path, 'eta_flatness.png'), format='png')


    cvm_values = compute_cvm_by_particle(labels, preds,
                                         {'Pt': data.JetPT.values,
                                          'Eta':data.JetEta.values,})

    print (cvm_values)
    cvm_values.to_csv(os.path.join(path, 'flatness.csv'))


#    return matrix