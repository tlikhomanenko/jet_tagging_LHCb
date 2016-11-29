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


def add_features(*arrays):
    new_data = []
    for data in arrays:
        data['SV_M_PT'] = numpy.log1p(data['SVM'] / data['SVPT'])
        data['SV_MC_PT'] = numpy.log1p(data['SVMCor'] / data['SVPT'])
        data['SVM_diff'] = numpy.sqrt(numpy.clip(data['SVMCor'] ** 2 - data['SVM']**2, 0, 1e10))
        data['SV_theta'] = numpy.log1p(numpy.sqrt(numpy.clip(data['SVMCor'] ** 2 - data['SVM']**2, 0, 1e10)) / data['SVPT'])
        data['SVM_rel'] = numpy.log1p(data['SVM'] / data['SVMCor'])
        data['SV_Q_N_rel'] = 1. * data['SVQ'] / data['SVN']
        data['SV_Q_abs'] = abs(data['SVQ'])
        dot_prod = lambda x, y: x[0]*y[0] + x[1]*y[1] + x[2]*y[2]
        sv_pos = (data['SVX'], data['SVY'], data['SVZ'])
        sv_p = (data['SVPx'], data['SVPy'], data['SVPz'])
        data['SV_cos_angle'] = dot_prod(sv_pos, sv_p) / numpy.sqrt(dot_prod(sv_pos, sv_pos) * dot_prod(sv_p, sv_p))
        data['JetSigma1toJetSigma2'] = data['JetSigma1'] / data['JetSigma2']
        data.loc[~numpy.isfinite(data['JetSigma1toJetSigma2']), 'JetSigma1toJetSigma2'] = 0
        data['JetSigma1multJetSigma2'] = data['JetSigma1'] * data['JetSigma2']
        data['SVPTtoJetPT'] = data.SVPT.values / data.JetPT.values
        data['MuPTtoJetPT'] = data.MuPT.values / data.JetPT.values
        data['HardPTtoJetPT'] = data.HardPT.values / data.JetPT.values
        new_data.append(data)
    return new_data


def names_labels_correspondence_update(new_labels_names_correspondence):
    labels_names_correspondence = new_labels_names_correspondence
    labels_names_correspondence = OrderedDict(sorted(labels_names_correspondence.items()))
    names_labels_correspondence = OrderedDict(map(lambda (x, y): (y, x), labels_names_correspondence.items()))


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
        plt.figure(figsize=(6, 4))
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
    plt.figure(figsize=(22, 5))
    for label, name in labels_names_correspondence.items():
        plt.subplot(1, 3, label + 1)
        for label_vs, name_vs in labels_names_correspondence.items():
            if label == label_vs:
                continue
            mask = (labels == label) | (labels == label_vs)
            fpr, tpr, _ = roc_curve(labels[mask] == label, 
                                    predictions_dict[label][mask] / predictions_dict[label_vs][mask], 
                                    sample_weight=weights if weights is None else weights[mask])
            auc = roc_auc_score(labels[mask] == label, predictions_dict[label][mask] / predictions_dict[label_vs][mask],
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
            roc_auc_matrices[label, label_vs] = roc_auc_score(labels[mask] == label,
                                                              predictions_dict[label][mask] / predictions_dict[label_vs][mask],
                                                              sample_weight=weights if weights is None else weights[mask])
        
    matrix = pandas.DataFrame(roc_auc_matrices, columns=names_labels_correspondence.keys(),
                              index=names_labels_correspondence.keys())

    fig=plot_matrix(matrix)
    return fig, matrix


def plot_matrix(matrix, vmin=0.8, vmax=1., title='Particle vs particle ROC AUCs', fmt='.5f'):
    # Plot roc_auc_matrices
    inline_rc = dict(matplotlib.rcParams)
    
    import seaborn as sns
    fig = plt.figure(figsize=(4, 3))
    sns.set()
    ax = plt.axes()
    sns.heatmap(matrix, vmin=vmin, vmax=vmax, annot=True, fmt=fmt, ax=ax, cmap=cm.coolwarm)
    plt.title(title, size=12)
    plt.xticks(size=12)
    plt.yticks(size=12)
    
    plt.show()
    plt.clf()
    plt.close()
    
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    matplotlib.rcParams.update(inline_rc)
    return fig


def generate_plots(preds, labels, weights, data, path=''):
    matrix_auc_one_vs_rest = roc_auc_score_one_vs_all_for_separate_algorithms(labels, preds, weights)
    print (matrix_auc_one_vs_rest)

    plot_roc_one_vs_rest(labels, preds, weights)
    # plt.savefig(os.path.join(path, 'overall_roc_auc.png'), format='png')

    f, matrix_auc_one_vs_one = compute_roc_auc_matrix(labels, preds, weights)
    # f.savefig(os.path.join(path, 'class_vs_class_roc_auc_matrix.png'), format='png')

    #matrix_auc_one_vs_rest.to_csv(os.path.join(path, 'class_vs_rest_roc_auc_matrix.csv'))
    #matrix_auc_one_vs_one.to_csv(os.path.join(path, 'class_vs_class_roc_auc_matrix.csv'))

    plot_roc_one_vs_one(labels, preds, weights)
    # plt.savefig(os.path.join(path, 'one_vs_one_roc_auc.png'), format='png')

    
def plot_feature_importances(feature_importances, features):
    imp = numpy.array(feature_importances)
    names = numpy.array(features)
    sort = imp.argsort()

    plt.figure(figsize=(12, numpy.ceil(8 * len(features) / 30.) ))
    plt.barh(range(len(imp)), imp[sort], align='center', color='b')
    plt.yticks(range(len(names)), names[sort], rotation=0)
    plt.title("Feature Importances", fontsize=15)
    plt.xlabel('Importance', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=12)
    plt.ylim(-0.5, len(names))
    plt.grid(linewidth=1)
    plt.show()