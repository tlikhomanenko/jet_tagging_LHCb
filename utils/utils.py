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
from rep.metaml.utils import map_on_cluster
from rep.metaml.factory import train_estimator
import time
from rep.metaml import ClassifiersFactory



# names_pdg_correspondence = {"Ghost": 0, "Electron": 11, "Muon": 13, "Pion": 211, "Kaon": 321, "Proton": 2212}
# names_labels_correspondence = {"Ghost": 0, "Electron": 1, "Muon": 2, "Pion": 3, "Kaon": 4, "Proton": 5}
pdg_names_correspondence = {0: "Ghost", 11: "Electron", 13: "Muon", 211: "Pion", 321: "Kaon", 2212: "Proton"}
labels_names_correspondence = {0: "Ghost", 1:"Electron", 2: "Muon", 3: "Pion", 4: "Kaon", 5: "Proton"}

labels_names_correspondence = OrderedDict(sorted(labels_names_correspondence.items()))
pdg_names_correspondence    = OrderedDict(sorted(pdg_names_correspondence.items()))

names_pdg_correspondence = OrderedDict(map(lambda (x, y): (y, x), pdg_names_correspondence.items()))
names_labels_correspondence = OrderedDict(map(lambda (x, y): (y, x), labels_names_correspondence.items()))


def shrink_floats(data):
    for column in data.columns:
        if data[column].dtype == 'float64':
            data[column] = data[column].astype('float32')
        
        
def compute_labels_and_weights(pdg_column):
    """
    Compute labels column (from zero to five) and weight (sum of weights for each class are the same - balanced data).
    
    :param array pdg_column: pdg value for each sample
    :return: labels, weights
    """
    labels = numpy.abs(pdg_column).astype(int)
    mask = numpy.zeros(len(labels), dtype=bool)
    for key, val in names_pdg_correspondence.items():
        if key == 'Ghost':
            continue
        mask = mask | (labels == val)
    labels[~(mask)] = 0 # all other particles are not tracks, so they are GHOST also
    
    for key, value in names_labels_correspondence.items():
        labels[labels == names_pdg_correspondence[key]] = value
    weights = numpy.ones(len(labels))
    for label in names_labels_correspondence.values():
        weights[labels == label] = 1. / sum(labels == label)
    weights /= numpy.mean(weights) + 1e-10
    return labels, weights


def compute_charges(pdg_column):
    """
    Compute charge for each track to check charges assymetry for the algorithm.
    Charge can be -1, +1 and 0 (zero corresponds to GHOST tracks)
    
    :param array pdg_column: pdg value for each sample, it has the sign
    :return: charges
    """
    charges = numpy.zeros(len(pdg_column))
    charges[pdg_column == 11] = -1
    charges[pdg_column == 13] = -1
    charges[(pdg_column == 321) | (pdg_column == 211) | (pdg_column == 2212)] = 1
    charges[pdg_column == -11] = 1
    charges[pdg_column == -13] = 1
    charges[(pdg_column == -321) | (pdg_column == -211) | (pdg_column == -2212)] = -1
    return charges


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


def compute_eta(track_p, track_pt):

    """
    Calculate pseudo rapidity values
    
    :param track_p: array, shape = [n_samples], TrackP values.
    :param track_pt: array, shape = [n_samples], TrackPt values.
    :return: array, shape = [n_samples], Pseudo Rapdity values.
    """

    sinz = 1. * track_pt / track_p
    z = numpy.arcsin(sinz)
    eta = - numpy.log(numpy.tan(0.5 * z))

    return eta


class ClassifiersFactoryByClass(ClassifiersFactory):
    def fit(self, X, y, sample_weight=None, parallel_profile=None, features=None):
        """
        Train all estimators on the same data.
        :param X: pandas.DataFrame of shape [n_samples, n_features] with features
        :param y: array-like of shape [n_samples] with labels of samples
        :param sample_weight: weights of events,
        array-like of shape [n_samples] or None if all weights are equal
        :param features: features to train estimators
        If None, estimators will be trained on `estimator.features`
        :type features: None or list[str]
        :param parallel_profile: profile of parallel execution system or None
        :type parallel_profile: None or str
        :return: self
        """
        if features is not None:
            for name, estimator in self.items():
                if estimator.features is not None:
                    print('Overwriting features of estimator ' + name)
                self[name].set_params(features=features)

        start_time = time.time()
        labels = []
        for key in self.keys():
            labels.append((y == names_labels_correspondence[key]) * 1)
        result = map_on_cluster(parallel_profile, train_estimator, list(self.keys()), list(self.values()),
                                [X] * len(self), labels, [sample_weight] * len(self))
        for status, data in result:
            if status == 'success':
                name, estimator, spent_time = data
                self[name] = estimator
                print('model {:12} was trained in {:.2f} seconds'.format(name, spent_time))
            else:
                print('Problem while training on the node, report:\n', data)

        print("Totally spent {:.2f} seconds on training".format(time.time() - start_time))
        return self

import os

def get_eta(track_p, track_pt):

    """
    Calculate pseudo rapidity values.

    Parameters
    ----------
    track_p : array_like
        TrackP values with array shape = [n_samples].
    track_pt : array_like
        TrackPt values with array shape = [n_samples].

    Return
    ------
    eta : array_like
        Pseudo rapidity values with array shape = [n_samples].
    """

    sinz = 1. * track_pt / track_p
    z = numpy.arcsin(sinz)

    eta = - numpy.log(numpy.tan(0.5 * z))

    return eta

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

    plot_flatness_by_particle(labels, preds, 1 / data.TrackP.values,
                              '1/(Momentum MeV/$c$)', thresholds=[5, 20, 40, 60, 80], weights=weights,
                              ignored_sideband=0.02)
    plt.savefig(os.path.join(path, 'p_flatness.png'), format='png')

    plot_flatness_by_particle(labels, preds, 1 / data.TrackPt.values,
                              '1/(Transverse Momentum MeV/$c$)', thresholds=[5, 20, 40, 60, 80], weights=weights,
                              ignored_sideband=0.02)
    plt.savefig(os.path.join(path, 'pt_flatness.png'), format='png')


    plot_flatness_by_particle(labels, preds, get_eta(data.TrackP.values, data.TrackPt.values),
                              'Pseudo Rapidity', thresholds=[5, 20, 40, 60, 80], weights=weights,
                              ignored_sideband=0.02)
    plt.savefig(os.path.join(path, 'eta_flatness.png'), format='png')


    plot_flatness_by_particle(labels, preds, data.NumProtoParticles.values,
                              'NumProtoParticles', thresholds=[5, 20, 40, 60, 80], weights=weights,
                              ignored_sideband=0.02)
    plt.savefig(os.path.join(path, 'num_proto_particles_flatness.png'), format='png')

    cvm_values = compute_cvm_by_particle(labels, preds,
                                         {'TrackP': data.TrackP.values,
                                          'TrackPt': data.TrackPt.values,
                                          'PseudoRapidity':get_eta(data.TrackP.values, data.TrackPt.values),
                                          'NumProtoParticles':data.NumProtoParticles.values})

    print (cvm_values)
    cvm_values.to_csv(os.path.join(path, 'flatness.csv'))

    plot_flatness_particle(labels, preds, 1 / data.TrackPt.values,
                          '1/(Transverse Momentum MeV/$c$)', particle_name='Pion', thresholds=[80, 85, 90, 95],
                           weights=weights,
                           ignored_sideband=0.02)
    plt.savefig(os.path.join(path, 'pt_pion_flatness.png'), format='png')

    plot_flatness_particle(labels, preds, 1 / data.TrackP.values,
                          '1/(Momentum MeV/$c$)', particle_name='Pion', thresholds=[80, 85, 90, 95],
                           weights=weights,
                           ignored_sideband=0.02)
    plt.savefig(os.path.join(path, 'p_pion_flatness.png'), format='png')


########################## Mikhail Hushchyn Version is below ##########################################

import numpy
import pandas
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from rep.utils import get_efficiencies
from rep.plotting import ErrorPlot


def __rolling_window(data, window_size):
    """
    Rolling window: take window with definite size through the array

    Parameters
    ----------
    data : array-like
    window_size : int
        Size.

    Return
    ------
    sequence_windows : array_like
        The sequence of windows.

    Example
    -------
    >>> __rolling_window(array(1, 2, 3, 4, 5, 6), window_size = 4)
    array(array(1, 2, 3, 4), array(2, 3, 4, 5), array(3, 4, 5, 6))
    """
    shape = data.shape[:-1] + (data.shape[-1] - window_size + 1, window_size)
    strides = data.strides + (data.strides[-1],)
    return numpy.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def __cvm(subindices, total_events):
    """
    Compute Cramer-von Mises metric.
    Compared two distributions, where first is subset of second one.
    Assuming that second is ordered by ascending

    Parameters
    ----------
    subindices : array_like
        Indices of events which will be associated with the first distribution.
    total_events : int
        Count of events in the second distribution.

    Return
    ------
    metric_value : float
        Cvm metric value.
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

    Parameters
    ----------
    predictions : array_like
        Predictions.
    masses : array_like
        In case of Kaggle tau23mu this is reconstructed mass.
    n_neighbours : int
        Count of neighbours for event to define mass bin.
    step : int
        Step through sorted mass-array to define next center of bin.

    Return
    ------
    avg_cvm_value : float
        Average cvm value.
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



def labels_transform(labels):

    """
    Transform labels from shape = [n_samples] to shape = [n_samples, n_classes].

    Parameters
    ----------
    labels : array_like
        Labels with ineger values.

    Return
    ------
    labels : ndarray
        Transformed labels with {0, 1} values.
    """

    classes = numpy.unique(labels)

    new_labels = numpy.zeros((len(labels), len(classes)))
    for cl in classes:
        new_labels[:, cl] = (labels == cl) * 1.

    return new_labels


def get_roc_curves(labels, probas, curve_labels, save_path=None, show=True):
    """
    Creates roc curve for each class vs rest.

    Parameters
    ----------
    labels : array_like
        Labels for the each class 0, 1, ..., n_classes - 1.
    probas : ndarray
        Predicted probabilities with ndarray shape = [n_samples, n_classes].
    curve_labels : array_like
        Labels of the curves with array shape = [n_classes].
    save_path : string
        Path to a directory where the figure will saved. If None the figure will not be saved.
    show : boolean
        If true the figure will be displayed.
    """

    labels = labels_transform(labels)



    weights = numpy.zeros(len(labels))

    for num in range(labels.shape[1]):

        weights += 0.1 * (labels[:, num] == 1) * len(labels) / ((labels[:, num] == 1).sum())





    plt.figure(figsize=(10,7))

    for num in range(probas.shape[1]):

        roc_auc = roc_auc_score(labels[:, num], probas[:, num], sample_weight=weights)
        fpr, tpr, _ = roc_curve(labels[:, num], probas[:, num], sample_weight=weights)

        plt.plot(tpr, 1.-fpr, label=curve_labels[num] + ', %.4f' % roc_auc, linewidth=2)

    plt.title("ROC Curves", size=15)
    plt.xlabel("Signal efficiency", size=15)
    plt.ylabel("Background rejection", size=15)
    plt.legend(loc='best',prop={'size':15})
    plt.xticks(numpy.arange(0, 1.01, 0.1), size=15)
    plt.yticks(numpy.arange(0, 1.01, 0.1), size=15)


    if save_path != None:
        plt.savefig(save_path + "/overall_roc_auc.png")

    if show == True:
        plt.show()

    plt.clf()
    plt.close()

def get_roc_auc_matrix(labels, probas, axis_labels, save_path=None, show=True):

    """
    Calculate class vs class roc aucs matrix.

    Parameters
    ----------
    labels : array_like
        Labels for the each class 0, 1, ..., n_classes - 1 with array shape = [n_samples].
    probas : ndarray
        Predicted probabilities with array shape = [n_samples, n_classes].
    axis_labels : array_like
        Labels of the curves with array shape = [n_classes].
    save_path : string
        Path to a directory where the figure will saved. If None the figure will not be saved.
    show : boolean
        If true the figure will be displayed.

    Return
    ------
    roc_auc_matrix : pandas.DataFrame
        A table with the roc aucs values.
    """

    labels = labels_transform(labels)

    # Calculate roc_auc_matrices
    roc_auc_matrices = numpy.ones((probas.shape[1],probas.shape[1]))

    for first in range(probas.shape[1]):
        for second in range(probas.shape[1]):

            if first == second:
                continue

            weights = ((labels[:, first] != 0) + (labels[:, second] != 0)) * 1.

            roc_auc = roc_auc_score(labels[:, first], probas[:, first]/probas[:, second], sample_weight=weights)

            roc_auc_matrices[first, second] = roc_auc


    # Save roc_auc_matrices
    matrix = pandas.DataFrame(columns=axis_labels, index=axis_labels)

    for num in range(len(axis_labels)):

        matrix[axis_labels[num]] = roc_auc_matrices[num, :]

    if save_path != None:
        matrix.to_csv(save_path + "/class_vs_class_roc_auc_matrix.csv")


    # Plot roc_auc_matrices
    inline_rc = dict(mpl.rcParams)
    import seaborn as sns
    plt.figure(figsize=(10,7))
    sns.set()
    ax = plt.axes()
    sns.heatmap(matrix, vmin=0.8, vmax=1., annot=True, fmt='.4f', ax=ax, cmap=cm.coolwarm)
    plt.title('Particle vs particle roc aucs', size=15)
    plt.xticks(size=15)
    plt.yticks(size=15)

    if save_path != None:
        plt.savefig(save_path + "/class_vs_class_roc_auc_matrix.png")

    if show == True:
        plt.show()

    plt.clf()
    plt.close()

    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams.update(inline_rc)

    return matrix

def get_roc_auc_ratio_matrix(matrix_one, matrix_two, save_path=None, show=True):

    """
    Divide matrix_one to matrix_two.

    Parameters
    ----------
    matrix_one : pandas.DataFrame
        A matrix with roc auc values and with column 'Class' which contains class names.
    matrix_two : pandas.DataFrame
        A matrix with roc auc values and with column 'Class' which contains class names.
    save_path : string
        Path to a directory where the figure will saved. If None the figure will not be saved.
    show : boolean
        If true the figure will be displayed.


    Return
    ------
    roc_auc_ratio_matrix : pandas.DataFrame
        A table with ratios of the roc auc values from the two matrix.
    """

    # Calculate roc_auc_matrices
    classes = list(matrix_one.index)
    roc_auc_matrices = numpy.ones((len(classes), len(classes)))

    for first in range(len(classes)):
        for second in range(len(classes)):

            roc_auc_one = matrix_one.loc[classes[first], classes[second]]
            roc_auc_two = matrix_two.loc[classes[first], classes[second]]
            roc_auc_matrices[first, second] = roc_auc_one / roc_auc_two

    # Save roc_auc_matrices
    matrix = pandas.DataFrame(columns=classes, index=classes)

    for num in range(len(classes)):

        matrix[classes[num]] = roc_auc_matrices[num, :]

    if save_path != None:
        matrix.to_csv(save_path + "/class_vs_class_roc_auc_rel_matrix.csv")

    # Plot roc_auc_matrices
    from matplotlib import cm
    inline_rc = dict(mpl.rcParams)
    import seaborn as sns
    plt.figure(figsize=(10,7))
    sns.set()
    ax = plt.axes()
    sns.heatmap(matrix, vmin=0.9, vmax=1.1, annot=True, fmt='.4f', ax=ax, cmap=cm.seismic)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.title('Particle vs particle roc aucs ratio', size=15)

    if save_path != None:
        plt.savefig(save_path + "/class_vs_class_roc_auc_rel_matrix.png")

    if show == True:
        plt.show()

    plt.clf()
    plt.close()

    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams.update(inline_rc)

    return matrix

def get_flatness_threshold(n_simulations, q, track):

    """
    Compute percentile of CvM test for flatness.

    Parameters
    ----------
    n_simulations : int
        Number of simulations.
    q : float
        Percentile.
    track : array_like
        Variable values along which the CvM test computes

    Return
    ------
    threshold : float
        A cvm pdf value which corresponds to the percentile.
    """

    cvm_pdf = []

    for step in range(n_simulations):

        proba_rand = numpy.random.random(len(track))
        cvm_pdf.append(compute_cvm(proba_rand, track))

    cvm_pdf = numpy.array(cvm_pdf)
    threshold = numpy.percentile(cvm_pdf, q)

    return threshold

def get_flatness_table(data, labels, probas, class_names, save_path=None):

    """
    Compute CvM tests for TrackP and TrackPt for each classes.

    Parameters
    ----------
    data : pandas.DataFrame
        Data which contains TrackP and TrackPt columns.
    labels : array_like
        Labels for the each class 0, 1, ..., n_classes - 1 with array shape = [n_samples].
    probas : ndarray
        Predicted probabilities with array shape = [n_samples, n_classes].
    class_names : string
        Path to a directory where the results will be saved. If None the results will not be saved.

    Return
    ------
    flatness : pandas.DataFrame
        A table with the flatness values.
    """

    labels = labels_transform(labels)

    GeV = 1000
    limits = {"TrackP": [100*GeV, 0],
              "TrackPt": [10*GeV, 0] }

    track_p = data.TrackP.values
    sel_p = (track_p >= limits["TrackP"][1]) * (track_p < limits["TrackP"][0])

    track_pt = data.TrackPt.values
    sel_pt = (track_pt >= limits["TrackPt"][1]) * (track_pt < limits["TrackPt"][0])



    cvm_track_p = []
    cvm_track_pt = []

    threshold_track_p = []
    threshold_track_pt = []

    for num in range(probas.shape[1]):

        sel_class_p = sel_p * (labels[:, num] == 1)
        sel_class_pt = sel_pt * (labels[:, num] == 1)

        cvm_p = compute_cvm(probas[sel_class_p, num], track_p[sel_class_p])
        cvm_track_p.append(cvm_p)

        threshold_p = get_flatness_threshold(100, 95, track_p[sel_class_p])
        threshold_track_p.append(threshold_p)


        cvm_pt = compute_cvm(probas[sel_class_pt, num], track_pt[sel_class_pt])
        cvm_track_pt.append(cvm_pt)

        threshold_pt = get_flatness_threshold(100, 95, track_pt[sel_class_pt])
        threshold_track_pt.append(threshold_pt)

    flatness = pandas.DataFrame(columns=['TrackP', 'TrackPt', 'P_Conf_level', 'Pt_Conf_level'], index=class_names)
    flatness['TrackP'] = cvm_track_p
    flatness['TrackPt'] = cvm_track_pt
    flatness['P_Conf_level'] = threshold_track_p
    flatness['Pt_Conf_level'] = threshold_track_pt

    if save_path != None:
        flatness.to_csv(save_path + "/flatness.csv")

    return flatness

def get_flatness_ratio(flatness_one, flatness_two, save_path=None):

    """
    Get ratio of flatness_one and flatness_two

    Parameters
    ----------
    flatness_one : pandas.DataFrame
        A table with the flatness values and with column 'Class' which contain class names.
    flatness_two : pandas.DataFrame
        A table with the flatness values and with column 'Class' which contain class names.
    save_path : string
        Path to a directory where the figure will saved. If None the figure will not be saved.

    Return
    ------
    flatness_ratios : pandas.DataFrame
        A table with ratios of the flatness from the tables.
    """

    classes = flatness_one.index

    flatness_arr = numpy.zeros((len(classes), 2))

    for num in range(len(classes)):

        flat_one = flatness_one.loc[classes[num]][[u'TrackP', u'TrackPt']].values
        flat_two = flatness_two.loc[classes[num]][[u'TrackP', u'TrackPt']].values

        flatness_arr[num, :] = flat_one / flat_two


    flatness = pandas.DataFrame(columns=['TrackP', 'TrackPt'], index=classes)
    flatness['TrackP'] = flatness_arr[:, 0]
    flatness['TrackPt'] = flatness_arr[:, 1]

    if save_path != None:
        flatness.to_csv(save_path + "/rel_flatness.csv")

    return flatness

from collections import OrderedDict

def flatness_p_figure(proba, proba_baseline, track_p, track_name, particle_name, save_path=None, show=False):

    """
    Plot signal efficiency vs TrackP figure.

    Parameters
    ----------
    proba : array_like
        Predicted probabilities with array shape = [n_samples].
    probas_baseline : array_like
        Baseline predicted probabilities with array shape = [n_samples].
    track_p : array_like
        TrackP values with array shape = [n_samples].
    track_name : string
        A track name.
    particle_name : string
        A particle name.
    save_path : string
        Path to a directory where the figure will saved. If None the figure will not be saved.
    show : boolean
        If true the figure will be displayed.
    """

    thresholds = numpy.percentile(proba, 100 - numpy.array([20, 50, 80]))
    thresholds_baseline = numpy.percentile(proba_baseline, 100 - numpy.array([20, 50, 80]))

    eff = get_efficiencies(proba,
                           track_p,
                           bins_number=30,
                           errors=True,
                           ignored_sideband=0,
                           thresholds=thresholds)

    eff_baseline = get_efficiencies(proba_baseline,
                                    track_p,
                                    bins_number=30,
                                    errors=True,
                                    ignored_sideband=0,
                                    thresholds=thresholds_baseline)

    for i in thresholds:
        eff[i] = (eff[i][0], 100. * eff[i][1], 100. * eff[i][2], eff[i][3])

    for i in thresholds_baseline:
        eff_baseline[i] = (eff_baseline[i][0], 100. * eff_baseline[i][1], 100. * eff_baseline[i][2], eff_baseline[i][3])


    eff_total = OrderedDict()
    num = len(eff) + len(eff_baseline)

    for i in range(len(eff)):

        v = eff[eff.keys()[i]]
        v_baseline = eff_baseline[eff_baseline.keys()[i]]

        eff_total[num] = v
        eff_total[num - 1] = v_baseline
        num += -2


    plot_fig = ErrorPlot(eff_total)
    plot_fig.ylim = (0, 100)

    plot_fig.plot(new_plot=True, figsize=(10,7))
    labels = ['Eff model = 20 %', 'Eff baseline = 20 %',
              'Eff model = 50 %', 'Eff baseline = 50 %',
              'Eff model = 80 %', 'Eff baseline = 80 %']
    plt.legend(labels, loc='best',prop={'size':12}, framealpha=0.5, ncol=1)
    plt.xlabel(track_name + ' ' + particle_name + ' Momentum / MeV/c', size=15)
    plt.xticks(size=15)
    plt.ylabel('Efficiency / %', size=15)
    plt.yticks(size=15)
    plt.title('Flatness_SignalMVAEffVTrackP_' + track_name + ' ' + particle_name, size=15)

    if save_path != None:
        plt.savefig(save_path + "/" + 'Flatness_SignalMVAEffVTrackP_' + track_name + '_' + particle_name + ".png")

    if show == True:
        plt.show()

    plt.clf()
    plt.close()


def flatness_pt_figure(proba, proba_baseline, track_pt, track_name, particle_name, save_path=None, show=False):

    """
    Plot signal efficiency vs TrackPt figures.

    Parameters
    ----------
    proba : array_like
        Predicted probabilities with array shape = [n_samples].
    probas_baseline : array_like
        Baseline predicted probabilities with array shape = [n_samples].
    track_pt : array_like
        TrackPt values with array shape = [n_samples].
    track_name : string
        A track name.
    particle_name : string
        A particle name.
    save_path : string
        Path to a directory where the figure will saved. If None the figure will not be saved.
    show : boolean
        If true the figure will be displayed.
    """

    thresholds = numpy.percentile(proba, 100 - numpy.array([20, 50, 80]))
    thresholds_baseline = numpy.percentile(proba_baseline, 100 - numpy.array([20, 50, 80]))

    eff = get_efficiencies(proba,
                           track_pt,
                           bins_number=30,
                           errors=True,
                           ignored_sideband=0,
                           thresholds=thresholds)

    eff_baseline = get_efficiencies(proba_baseline,
                                    track_pt,
                                    bins_number=30,
                                    errors=True,
                                    ignored_sideband=0,
                                    thresholds=thresholds_baseline)

    for i in thresholds:
        eff[i] = (eff[i][0], 100. * eff[i][1], 100. * eff[i][2], eff[i][3])

    for i in thresholds_baseline:
        eff_baseline[i] = (eff_baseline[i][0], 100. * eff_baseline[i][1], 100. * eff_baseline[i][2], eff_baseline[i][3])


    eff_total = OrderedDict()
    num = len(eff) + len(eff_baseline)

    for i in range(len(eff)):

        v = eff[eff.keys()[i]]
        v_baseline = eff_baseline[eff_baseline.keys()[i]]

        eff_total[num] = v
        eff_total[num - 1] = v_baseline
        num += -2

    plot_fig = ErrorPlot(eff_total)
    plot_fig.ylim = (0, 100)

    plot_fig.plot(new_plot=True, figsize=(10,7))
    labels = ['Eff model = 20 %', 'Eff baseline = 20 %',
              'Eff model = 50 %', 'Eff baseline = 50 %',
              'Eff model = 80 %', 'Eff baseline = 80 %']
    plt.legend(labels, loc='best',prop={'size':12}, framealpha=0.5, ncol=1)
    plt.xlabel(track_name + ' ' + particle_name + ' Transverse Momentum / MeV/c', size=15)
    plt.xticks(size=15)
    plt.ylabel('Efficiency / %', size=12)
    plt.yticks(size=15)
    plt.title('Flatness_SignalMVAEffVTrackPt_' + track_name + ' ' + particle_name, size=15)

    if save_path != None:
        plt.savefig(save_path + "/" + 'Flatness_SignalMVAEffVTrackPt_' + track_name + '_' + particle_name + ".png")

    if show == True:
        plt.show()

    plt.clf()
    plt.close()




def get_all_p_pt_flatness_figures(data, probas, probas_baseline, labels, track_name, particle_names, save_path=None, show=False):

    """
    Plot signal efficiency vs TrackP and TrackPt figure.

    Parameters
    ----------
    probas : ndarray
        Predicted probabilities with array shape = [n_samples, n_classes].
    probas_baseline : ndarray
        Baseline predicted probabilities with array shape = [n_samples, n_classes].
    labels : array_like
        Class labels 0, 1, ..., n_classes - 1 with array shape = [n_samples].
    track_p : array_like
        TrackP values with array shape = [n_samples].
    track_name : string
        The track name.
    particle_names : array)like
        The particle names.
    save_path : string
        Path to a directory where the figure will saved. If None the figure will not be saved.
    show : boolean
        If true the figure will be displayed.
    """

    labels = labels_transform(labels)

    GeV = 1000
    limits = {"TrackP": [100*GeV, 0],
              "TrackPt": [10*GeV, 0] }

    track_p = data.TrackP.values
    sel_p = (track_p >= limits["TrackP"][1]) * (track_p < limits["TrackP"][0])

    track_pt = data.TrackPt.values
    sel_pt = (track_pt >= limits["TrackPt"][1]) * (track_pt < limits["TrackPt"][0])


    for num in range(len(particle_names)):

        sel_class_p = sel_p * (labels[:, num] == 1)
        sel_class_pt = sel_pt * (labels[:, num] == 1)

        probas[sel_class_p, num], track_p[sel_class_p]

        flatness_p_figure(probas[sel_class_p, num], probas_baseline[sel_class_p, num],
                          track_p[sel_class_p],
                          track_name,
                          particle_names[num],
                          save_path=save_path,
                          show=show)

        flatness_pt_figure(probas[sel_class_pt, num], probas_baseline[sel_class_pt, num],
                          track_pt[sel_class_pt],
                          track_name,
                          particle_names[num],
                          save_path=save_path,
                          show=show)


from collections import OrderedDict

def flatness_ntracks_figure(proba, proba_baseline, ntracks, track_name, particle_name, save_path=None, show=False):

    """
    Plot signal efficiency vs number of protoparticles.

    Parameters
    ----------
    probas : ndarray
        Predicted probabilities with array shape = [n_samples, n_classes].
    probas_baseline : ndarray
        Baseline predicted probabilities with array shape = [n_samples, n_classes].
    labels : array_like
        Labels of the particles with array shape = [n_sample].
    ntracks : array_like
        NumProtoParticles values with array shape = [n_samples].
    track_name : string
        The track name.
    particle_names : array_like
        The particle names.
    save_path : string
        Path to a directory where the figure will saved. If None the figure will not be saved.
    show : boolean
        If true the figure will be displayed.
    """

    thresholds = numpy.percentile(proba, 100 - numpy.array([20, 50, 80]))
    thresholds_baseline = numpy.percentile(proba_baseline, 100 - numpy.array([20, 50, 80]))

    eff = get_efficiencies(proba,
                           ntracks,
                           bins_number=30,
                           errors=True,
                           ignored_sideband=0,
                           thresholds=thresholds)

    eff_baseline = get_efficiencies(proba_baseline,
                                    ntracks,
                                    bins_number=30,
                                    errors=True,
                                    ignored_sideband=0,
                                    thresholds=thresholds_baseline)

    for i in thresholds:
        eff[i] = (eff[i][0], 100. * eff[i][1], 100. * eff[i][2], eff[i][3])

    for i in thresholds_baseline:
        eff_baseline[i] = (eff_baseline[i][0], 100. * eff_baseline[i][1], 100. * eff_baseline[i][2], eff_baseline[i][3])


    eff_total = OrderedDict()
    num = len(eff) + len(eff_baseline)

    for i in range(len(eff)):

        v = eff[eff.keys()[i]]
        v_baseline = eff_baseline[eff_baseline.keys()[i]]

        eff_total[num] = v
        eff_total[num - 1] = v_baseline
        num += -2


    plot_fig = ErrorPlot(eff_total)
    plot_fig.ylim = (0, 100)

    plot_fig.plot(new_plot=True, figsize=(10,7))
    labels = ['Eff model = 20 %', 'Eff baseline = 20 %',
              'Eff model = 50 %', 'Eff baseline = 50 %',
              'Eff model = 80 %', 'Eff baseline = 80 %']
    plt.legend(labels, loc='best',prop={'size':12}, framealpha=0.5, ncol=3)
    plt.xlabel(track_name + ' ' + particle_name + ' NumProtoParticles / units', size=15)
    plt.xticks(size=15)
    plt.ylabel('Efficiency / %', size=15)
    plt.yticks(size=15)
    plt.title('Flatness_SignalMVAEffVNumProtoParticles_' + track_name + ' ' + particle_name, size=15)

    if save_path != None:
        plt.savefig(save_path + "/" + 'Flatness_SignalMVAEffVNumProtoParticles_' + track_name + '_' + particle_name + ".png")

    if show == True:
        plt.show()

    plt.clf()
    plt.close()



def get_all_ntracks_flatness_figures(data, probas, probas_baseline, labels, track_name, particle_names, save_path=None, show=False):

    """
    Plot all signal efficiency vs number of protoparticles figures.

    Parameters
    ----------
    data : pandas.DataFrame
        Data.
    probas : ndarray
        Predicted probabilities with array shape = [n_samples, n_classes].
    probas_baseline : ndarray
        Baseline predicted probabilities with array shape = [n_samples, n_classes].
    labels : array_like
        Labels of the particles with array shape = [n_sample].
    track_name : string
        The track name.
    particle_names : array_like
        The particle names.
    save_path : string
        Path to a directory where the figure will saved. If None the figure will not be saved.
    show : boolean
        If true the figure will be displayed.
    """

    labels = labels_transform(labels)

    GeV = 1000
    limits = {"TrackP": [100*GeV, 0],
              "TrackPt": [10*GeV, 0] }

    ntracks = data.NumProtoParticles.values


    for num in range(len(particle_names)):


        probas[:, num], ntracks

        flatness_ntracks_figure(probas[:, num], probas_baseline[:, num],
                                ntracks,
                                track_name,
                                particle_names[num],
                                save_path=save_path,
                                show=show)


from collections import OrderedDict

def get_eta(track_p, track_pt):

    """
    Calculate pseudo rapidity values.

    Parameters
    ----------
    track_p : array_like
        TrackP values with array shape = [n_samples].
    track_pt : array_like
        TrackPt values with array shape = [n_samples].

    Return
    ------
    eta : array_like
        Pseudo rapidity values with array shape = [n_samples].
    """

    sinz = 1. * track_pt / track_p
    z = numpy.arcsin(sinz)

    eta = - numpy.log(numpy.tan(0.5 * z))

    return eta

def flatness_eta_figure(proba, proba_baseline, eta, track_name, particle_name, save_path=None, show=False):

    """
    Plot signal efficiency vs pseudo rapidity figure.

    Parameters
    ----------
    proba : array_like
        Predicted probabilities with array shape = [n_samples].
    probas_baseline : array_like
        Baseline predicted probabilities with array shape = [n_samples].
    eta : array_like
        Pseudo rapidity values with array shape = [n_samples].
    track_name : string
        The track name.
    particle_name : string
        The particle name.
    save_path : string
        Path to a directory where the figure will saved. If None the figure will not be saved.
    show : boolean
        If true the figure will be displayed.
    """

    thresholds = numpy.percentile(proba, 100 - numpy.array([20, 50, 80]))
    thresholds_baseline = numpy.percentile(proba_baseline, 100 - numpy.array([20, 50, 80]))

    eff = get_efficiencies(proba,
                           eta,
                           bins_number=30,
                           errors=True,
                           ignored_sideband=0.005,
                           thresholds=thresholds)

    eff_baseline = get_efficiencies(proba_baseline,
                                    eta,
                                    bins_number=30,
                                    errors=True,
                                    ignored_sideband=0.005,
                                    thresholds=thresholds_baseline)

    for i in thresholds:
        eff[i] = (eff[i][0], 100. * eff[i][1], 100. * eff[i][2], eff[i][3])

    for i in thresholds_baseline:
        eff_baseline[i] = (eff_baseline[i][0], 100. * eff_baseline[i][1], 100. * eff_baseline[i][2], eff_baseline[i][3])


    eff_total = OrderedDict()
    num = len(eff) + len(eff_baseline)

    for i in range(len(eff)):

        v = eff[eff.keys()[i]]
        v_baseline = eff_baseline[eff_baseline.keys()[i]]

        eff_total[num] = v
        eff_total[num - 1] = v_baseline
        num += -2


    plot_fig = ErrorPlot(eff_total)
    plot_fig.ylim = (0, 100)

    plot_fig.plot(new_plot=True, figsize=(10,7))
    labels = ['Eff model = 20 %', 'Eff baseline = 20 %',
              'Eff model = 50 %', 'Eff baseline = 50 %',
              'Eff model = 80 %', 'Eff baseline = 80 %']
    plt.legend(labels, loc='best',prop={'size':10}, framealpha=0.5, ncol=3)
    plt.xlabel(track_name + ' ' + particle_name + ' Pseudo Rapidity', size=15)
    plt.xticks(size=15)
    plt.ylabel('Efficiency / %', size=15)
    plt.yticks(size=15)
    plt.title('Flatness_SignalMVAEffVPseudoRapidity_' + track_name + ' ' + particle_name, size=15)

    if save_path != None:
        plt.savefig(save_path + "/" + 'Flatness_SignalMVAEffVPseudoRapidity_' + track_name + '_' + particle_name + ".png")

    if show == True:
        plt.show()

    plt.clf()
    plt.close()



def get_all_eta_flatness_figures(data, probas, probas_baseline, labels, track_name, particle_names, save_path=None, show=False):

    """
    Plot signal efficiency vs pseudo rapidity figure.

    Parameters
    ----------
    data : pandas.dataFrame
        Data.
    probas : ndarray
        Predicted probabilities with array shape = [n_samples, n_classes].
    probas_baseline : ndarray
        Baseline predicted probabilities with array shape = [n_samples, n_classes].
    labels : array_like
        The class labels 0, 1, ..., n_classes - 1 with array shape = [n_samples].
    track_p : array_like
        TrackP values with array shape = [n_samples].
    track_name : string
        The track name.
    particle_names : array_like
        The particle names.
    save_path : string
        Path to a directory where the figure will saved. If None the figure will not be saved.
    show : boolean
        If true the figure will be displayed.
    """

    labels = labels_transform(labels)

    GeV = 1000
    limits = {"TrackP": [100*GeV, 0],
              "TrackPt": [10*GeV, 0] }

    track_p = data.TrackP.values
    track_pt = data.TrackPt.values
    eta = get_eta(track_p, track_pt)


    for num in range(len(particle_names)):


        flatness_eta_figure(probas[:, num], probas_baseline[:, num],
                                eta,
                                track_name,
                                particle_names[num],
                                save_path=save_path,
                                show=show)



def get_one_vs_one_roc_curves(labels, probas, curve_labels, save_path=None, show=True):
    """
    Creates one vs one roc curves.

    Parameters
    ----------
    labels : array_like
        Labels for the each class 0, 1, ..., n_classes - 1 with array shape = [n_samples].
    probas : ndarray
        Predicted probabilities with array shape = [n_samples, n_classes].
    curve_labels : array_like
        Labels of the curves with array shape = [n_classes].
    save_path : string
        Path to a directory where the figure will saved. If None the figure will not be saved.
    show : boolean
        If true the figure will be displayed.
    """

    classes = numpy.unique(labels)

    for one_class, one_name in zip(classes, curve_labels):

        plt.figure(figsize=(10,7))

        for two_class, two_name in zip(classes, curve_labels):

            if one_class == two_class:
                continue

            weights = (labels == one_class) * 1. + (labels == two_class) * 1.
            one_labels = (labels == one_class) * 1.
            roc_auc = roc_auc_score(one_labels, probas[:, one_class] / probas[:, two_class], sample_weight=weights)
            fpr, tpr, _ = roc_curve(one_labels, probas[:, one_class] / probas[:, two_class], sample_weight=weights)

            plt.plot(tpr, 1.-fpr, label=one_name + ' vs ' + two_name + ', %.4f' % roc_auc, linewidth=2)

        plt.title("ROC Curves", size=15)
        plt.xlabel("Signal efficiency", size=15)
        plt.ylabel("Background rejection", size=15)
        plt.legend(loc='best',prop={'size':15})
        plt.xticks(numpy.arange(0, 1.01, 0.1), size=15)
        plt.yticks(numpy.arange(0, 1.01, 0.1), size=15)


        if save_path != None:
            plt.savefig(save_path + "/" + one_name + "_vs_one_roc_auc.png")

        if show == True:
            plt.show()

        plt.clf()
        plt.close()


def get_roc_aoc_ratio_matrix(matrix_one, matrix_two, save_path=None, show=True):

    """
    Divide matrix_one to matrix_two.

    Parameters
    ----------
    matrix_one : pandas.DataFrame
        A table with roc aoc values and with column 'Class' which contain class names.
    matrix_two : pandas.DataFrame
        A table with roc aoc values and with column 'Class' which contain class names.
    save_path : string
        Path to a directory where the figure will saved. If None the figure will not be saved.
    show : boolean
        If true the figure will be displayed.

    Return
    ------
    matrix : pandas.DataFrame
        Roc aoc ratios matrix: (1 - matrix_one / matrix_two) * 100%.
    """

    # Calculate roc_auc_matrices
    classes = list(matrix_one.index)
    roc_auc_matrices = numpy.ones((len(classes), len(classes)))

    for first in range(len(classes)):
        for second in range(len(classes)):

            roc_auc_one = matrix_one.loc[classes[first], classes[second]]
            roc_auc_two = matrix_two.loc[classes[first], classes[second]]
            roc_auc_matrices[first, second] = (1. - (1. - roc_auc_one) / (1. - roc_auc_two)) * 100.

    # Save roc_auc_matrices
    matrix = pandas.DataFrame(columns=classes, index=classes)

    for num in range(len(classes)):

        matrix[classes[num]] = roc_auc_matrices[num, :]

    if save_path != None:
        matrix.to_csv(save_path + "/class_vs_class_roc_aoc_rel_matrix.csv")

    # Plot roc_auc_matrices
    from matplotlib import cm
    inline_rc = dict(mpl.rcParams)
    import seaborn as sns
    plt.figure(figsize=(10,7))
    sns.set()
    ax = plt.axes()
    sns.heatmap(matrix, vmin=-100., vmax=100.0, annot=True, fmt='.1f', ax=ax, cmap=cm.seismic)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.title('Particle vs particle roc aucs ratio', size=15)

    if save_path != None:
        plt.savefig(save_path + "/class_vs_class_roc_aoc_rel_matrix.png")

    if show == True:
        plt.show()

    plt.clf()
    plt.close()

    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams.update(inline_rc)

    return matrix