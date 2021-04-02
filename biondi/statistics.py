import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import sklearn
from sklearn.neighbors import KDTree
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import random
import pandas as pd


def agreement_table(a, b, c):
    """
    Returns an agreement table as a 2D array.

    Meant for use in cohens_kappa() function.

    :param a: count values for observer #1
    :type a: ndarray of ints [Yes, No]
    :param b: count values for observer #2
    :type b: ndarray of ints [Yes, No]
    :param c: agreement count values between the 2 observers
    :type c: ndarray of ints [Yes, No]
    :return: agreement table
    :rtype: ndarray
    """
    if (len(a[0]) + len(a[1])) != (len(b[0]) + len(b[1])):
        print('total annotations do not match each other!')
        return
    table = np.empty([3, 3])
    table[0, 0] = c[0]
    table[0, 1] = a[1] - c[1]
    table[1, 0] = a[0] - c[0]
    table[1, 1] = c[1]
    table[0, 2] = b[0]
    table[1, 2] = b[1]
    table[2, 0] = a[0]
    table[2, 1] = a[1]
    table[2, 2] = len(a[0]) + len(a[1])
    print(table)
    return table


def binary_cohens_kappa(a, b, return_agreement_table=True):
    """
    Performs Cohen's kappa statistic on binary data from 2 observers.

    :param a: annotations/labels for observer 1
    :type a: ndarray
    :param b: annotations.labels for observer 2
    :type b: ndarray
    :param return_agreement_table: if True, returns an agreement table along with Kappa
    :type return_agreement_table: bool (default=True)
    :return: kappa value or kappa value and agreement table (if return_agreement_table is True)
    :rtype: List[float, Optional[ndarray]]
    """
    n_obs1 = len(a)
    n_obs2 = len(b)
    if n_obs1 != n_obs2:
        print('total annotations do not match each other!')
        return
    a.shape = (len(a),)
    b.shape = (len(b),)
    a = a.tolist()
    b = b.tolist()
    yes_obs1 = a.count(1)
    no_obs1 = a.count(0)
    yes_obs2 = b.count(1)
    no_obs2 = b.count(0)
    agreement = [i for i, j in zip(a, b) if i == j]
    yy = agreement.count(1)
    nn = agreement.count(0)
    expected_agreement = ((yes_obs1 / n_obs2) * (yes_obs2 / n_obs2)) + ((no_obs1 / n_obs2) * (no_obs2 / n_obs2))
    observed_agreement = (yy + nn) / n_obs2
    k = (observed_agreement - expected_agreement) / (1 - expected_agreement)
    print('Pe = ' + str(expected_agreement))
    print('Po = ' + str(observed_agreement))
    print('Cohens Kappa = ' + str(k))

    if return_agreement_table:
        table = agreement_table([yes_obs1, no_obs1], [yes_obs2, no_obs2], [yy, nn])
        return k, table
    else:
        return k


def binary_tpfptnfn(predictions, label):
    """
    Returns the number of true positives, false positives,
    true negatives,and false negatives for binary data.

    Keyword arguments:
    predictions -- an array or list of binary (0 or 1) predictions
    labels -- an array or list of binary (0 or 1) ground truth labels
    """
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(predictions)):
        if predictions[i] == 1 and label[i] == 1:
            tp += 1
        elif predictions[i] == 1 and label[i] == 0:
            fp += 1
        elif predictions[i] == 0 and label[i] == 0:
            tn += 1
        elif predictions[i] == 0 and label[i] == 1:
            fn += 1
        else:
            print('error', 'wrong label?')
            return
    print('TP:', tp)
    print('FP:', fp)
    print('TN:', tn)
    print('FN:', fn)
    return tp, fp, tn, fn


def sensitivity_specificity_ppv_npv(tp, fp, tn, fn):
    """
    Returns sensitivity, specificity, positive predictive value,
    and negative predictive value.

    Keyword arguments:
    tp -- # of true positives
    fp -- # of false positives
    tn -- # of true negatives
    fn -- # of false negatives
    """
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    print('Sensitivity:', sensitivity)
    print('Specificity:', specificity)
    print('PPV:', ppv)
    print('NPV:', npv)
    return sensitivity, specificity, ppv, npv


def ai_binary_statistics(predictions, labels):
    """
    Returns TP, FP, TN, FN, sensitivity, specificity, PPV, and NPV.

    Keyword arguments:
    predictions -- list or array of binary (0 or 1) predictions
    labels -- binary (0 or 1) ground truth
    """
    if len(predictions) == len(labels):
        a, b, c, d = binary_tpfptnfn(predictions, labels)
        e, f, g, h = sensitivity_specificity_ppv_npv(a, b, c, d)
    else:
        print('prediction-label mismatch!')
        return
    return a, b, c, d, e, f, g, h


def convert_probabilities_to_predictions(probs, from_logits=True):
    """
    Converts a 2D array of logits (default) or probabilities to
    binary predictions.

    Keyword arguments:
    probs -- a 2D array of binary class probabilities or logits
    from_logits -- boolean whether probs values are logits (default True)
    """
    if from_logits:
        return np.argmax(softmax(probs), axis=1)
    else:
        return np.argmax(probs, axis=1)


def get_pred_and_binary_statistics(probs, labels):
    """
    Returns binary comparison statistics (TP, FP, TN, FN, sensitivity,
    specificity, PPV, NPV) given logits and ground truth.

    Keyword arguments:
    probs -- a 2D array of binary class logits
    labels -- an array or list of binary ground truth labels
    """
    pred = convert_probabilities_to_predictions(probs)
    a, b, c, d, e, f, g, h = ai_binary_statistics(pred, labels)
    counter = 0
    for i in range(len(pred)):
        if pred[i] == labels[i]:
            counter += 1
    print('accuracy:', counter / len(pred))
    return a, b, c, d, e, f, g, h


def softmax(logits, axis=-1):
    """
    Returns probabilities from logits.

    Keyword arguments:
    logits -- a 2D array of logit values
    axis -- axis to calculate probabilities along (default)
    """
    y = np.exp(logits - logits.max(axis=axis, keepdims=True))
    return y / y.sum(axis=axis, keepdims=True)


# code way to optionally save the figures
#
#
#
def roc_pr_curves(probabilities, labels, roc=True, pr=True,
                  from_logits=True):
    """
    Returns Receiver operating characteristic (ROC) and precision-recall
    (PR) curves.

    Keyword arguments:
    probabilities -- a 2D array of binary class probabilities or logits
    labels -- an array or list of binary ground truth labels
    ROC -- Boolean; whether to return ROC graph (default True)
    PR -- Boolean; whether to return PR graph (default True)
    from_logits -- Boolean; whether probabilities are logits (default True)
    """
    roc_list = []
    auroc_list = []
    pr_list = []
    avgpr_list = []
    for i in probabilities:
        if from_logits:
            probs = softmax(i)
        else:
            probs = i
        if roc:
            # calculates FPR and TPR for a ROC curve
            roc = sklearn.metrics.roc_curve(labels, probs[:, 1])
            roc_list.append(roc)
            # calculates area under the ROC curve
            auc_roc = sklearn.metrics.roc_auc_score(labels, probs[:, 1])
            auroc_list.append(auc_roc)
        if pr:
            # calculates precision and recall for a PR curve
            pr = sklearn.metrics.precision_recall_curve(labels, probs[:, 1])
            pr_list.append(pr)
            # calculates average precision
            avg_pr = sklearn.metrics.average_precision_score(labels, probs[:, 1])
            avgpr_list.append(avg_pr)
    if roc:
        # plot ROC curve and AUROC
        plt.figure()
        lw = 2
        for i in range(len(roc_list)):
            plt.plot(roc_list[i][0], roc_list[i][1], lw=lw,
                     label='AUROC = %0.4f' % auroc_list[i])
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic curve')
        plt.legend(loc="lower right")
        plt.show()
    if pr:
        # plot PR curve and avg precision
        plt.figure()
        lw = 2
        for i in range(len(pr_list)):
            plt.plot(pr_list[i][1], pr_list[i][0], lw=lw,
                     label='Avg_precision = %0.2f' % avgpr_list[i])
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall (TP/(TP+FN))')
        plt.ylabel('Precision (TP/(TP+FP))')
        plt.title('Precision-Recall curve')
        plt.legend(loc="lower right")
        plt.show()


# Need to add code to add save option for figures
#
#
#
def bland_altman(x1, x2, percent_difference=False, save=False):
    """
    Returns Bland-Altman plots along with mean, min/max 95% CI of the
    mean, standard deviation, and min/max 95% CI of the standard deviation.

    Keyword arguments:
    x1 -- an array or list of values
    x2 -- an array or list of values of equal length as x1
    percent_difference -- Boolean; whether you want the percent_difference
    plotted instead of only the difference
    save -- Boolean; whether to save plot as figure
    """
    difference = [i - j for (i, j) in zip(x1, x2)]
    average = [sum([i, j]) / 2 for (i, j) in zip(x1, x2)]
    if percent_difference:
        difference = [i / j for (i, j) in zip(difference, average)]
    avg, var, std = scipy.stats.bayes_mvs(difference, 0.95)
    plt.figure()
    # points
    plt.scatter(average, difference)
    # mean difference
    plt.plot([0, max(average)], [avg[0], avg[0]], color='orange', label='Mean difference')
    # x origin axis
    plt.plot([0, max(average)], [0, 0], lw=1, color='k')
    # upper 95% CI for mean
    plt.plot([0, max(average)], [avg[1][1], avg[1][1]], color='orange', ls='--')
    # lower 95% CI for mean
    plt.plot([0, max(average)], [avg[1][0], avg[1][0]], color='orange', ls='--')
    # upper 1.96*std
    plt.plot([0, max(average)], [avg[0] + 1.96 * std[0], avg[0] + 1.96 * std[0]], color='lightblue',
             label='+/- 1.96*Stdev')
    # upper 95% CI for upper std
    plt.plot([0, max(average)], [avg[0] + 1.96 * std[1][1], avg[0] + 1.96 * std[1][1]], color='lightblue', ls='--')
    # lower 95% CI for upper std
    plt.plot([0, max(average)], [avg[0] + 1.96 * std[1][0], avg[0] + 1.96 * std[1][0]], color='lightblue', ls='--')
    # lower 1.96*std
    plt.plot([0, max(average)], [avg[0] - 1.96 * std[0], avg[0] - 1.96 * std[0]], color='lightblue')
    # upper 95% CI for lower std
    plt.plot([0, max(average)], [avg[0] - 1.96 * std[1][0], avg[0] - 1.96 * std[1][0]], color='lightblue', ls='--')
    # lower 95% CI for lower std
    plt.plot([0, max(average)], [avg[0] - 1.96 * std[1][1], avg[0] - 1.96 * std[1][1]], color='lightblue', ls='--')

    plt.xlim([0.0, max(average)])
    plt.legend(loc="upper right")
    # TODO: need to add code to add save option
    plt.show()
    return avg[0], avg[1][0], avg[1][1], std[0], std[1][0], std[1][1]


def coordinate_separation(total_cpec_coords, affected_cpec_coords, bin_num=50,
                          plot_range=(5, 20)):
    """
    Returns a histogram showing pixel distances within total CPEC coords
    and distances between total CPEC and affected CPEC coords.

    Histogram can be interpreted to determine a cutoff distance that
    identifies coordinates that are likely the same cell.

    Keyword arguments:
    total_cpec_coords -- a 2D array of all CPEC coordinates
    affected_cpec_coords -- a 2D array of affected CPEC coordinates
    bin_num -- # of bins to plot in the histogram (default 50)
    plot_range -- a two number tuple range of the histogram (default (5,20))
    """
    tc = KDTree(total_cpec_coords)
    self_separation = tc.query(total_cpec_coords, k=2)
    affected_separation = tc.query(affected_cpec_coords, k=1)
    plt.hist([self_separation[0][..., 1], affected_separation[0]],
             bins=bin_num, range=plot_range)


def indices_for_deleting_affected_coords_self(affected_self_separation, threshold):
    del_affected = []
    for i in range(len(affected_self_separation[0][..., 1])):
        if affected_self_separation[0][..., 1][i] <= threshold:
            del_affected.append(affected_self_separation[1][..., 1][i])
    return del_affected


def affected_self_total_separation_v1(total_coords, affected_coords):
    """
    Calculates nearest-neighbor (nn) distances.
    :param total_coords: total CPEC coordinates
    :param affected_coords: affected CPEC coordinates
    :return: nn distances within affected coords and between affected and total coords
    """
    ac = KDTree(affected_coords)
    tc = KDTree(total_coords)
    affected_self_separation = ac.query(affected_coords, k=2)
    affected_total_separation = tc.query(affected_coords, k=1)
    return affected_self_separation, affected_total_separation


def remove_redundant_affected_coords(total_cpec_coords, affected_cpec_coords,
                                     threshold):
    """
    Returns a 2D array of affected coordinates with redundant coordinates removed.

    Only redundant coordinates within affected coordinates are removed.

    Keyword arguments:
    total_cpec_coords -- a 2D array of all CPEC coordinates
    affected_cpec_coords -- a 2D array of affected CPEC coordinates
    threshold -- distance cutoff to determine redundant coordinates
    """
    ac = KDTree(affected_cpec_coords)
    affected_self_separation = ac.query(affected_cpec_coords, k=2)
    if total_cpec_coords:
        tc = KDTree(total_cpec_coords)
        affected_total_separation = tc.query(affected_cpec_coords, k=1)

    del_affected = indices_for_deleting_affected_coords_self(affected_self_separation, threshold)
    # May be able to remove this part of the code in the future.
    # That way, total CPEC coordinates do not need to be provided.
    if total_cpec_coords:
        for i in range(len(affected_total_separation[0])):
            if affected_total_separation[0][i] > threshold:
                del_affected.append(i)
    print(len(del_affected))
    trimmed_affected_coords = np.delete(affected_cpec_coords, del_affected, 0)
    return trimmed_affected_coords


def get_unaffected_from_total_coords(total_cpec_coords, trimmed_affected_cpec_coords, threshold):
    """
    Returns an array of unaffected (binary label:0) coordinates from total
    coordinates.

    Keyword arguments:
    total_cpec_coords -- an array of total CPEC coordinates
    trimmed_affected_cpec_coords -- an array of affected CPEC coordinates
    threshold -- distance threshold to determine matching total and affected
    coordinates
    """
    affected_self_separation, affected_total_separation = affected_self_total_separation_v1(total_cpec_coords,
                                                                                            trimmed_affected_cpec_coords
                                                                                            )
    del_total = []
    del_affected = indices_for_deleting_affected_coords_self(affected_self_separation, threshold)
    for i in range(len(affected_total_separation[0])):
        if affected_total_separation[0][i] > threshold:
            del_affected.append(i)
        else:
            del_total.append(affected_total_separation[1][i])
    print(len(del_affected), len(del_total))
    unaffected_coords = np.delete(total_cpec_coords, del_total, 0)
    return unaffected_coords


# CODE not finished; Need to finish
# def remove_redundant_coords(main_list, check_listk, radius):
#    """
#    Returns affected and total CPEC coordinates with redundant coordinates
#    removed.
#
#    Keyword arguments:
#    main_list --
#    check_;ost
#    """
#    main = []
#    check = []
#    for i in range(len(nn_dist_affected[0])):
#        if nn_dist_affected[0][i] > 15:
#            del_affected.append(i)
#        else:
#            del_total.append(nn_dist_affected[1][i])
#    print(len(del_total), len(del_affected))


def cpec_density(total_cpec_coords, radius=310.559):
    """

    :param total_cpec_coords:
    :type total_cpec_coords:
    :param radius:
    :type radius:
    :return:
    :rtype:
    """
    tc = KDTree(total_cpec_coords)
    total = tc.query_radius(total_cpec_coords, r=radius, count_only=True)
    return


def percent_affected(total_cpec_coords, affected_cpec_coords, density=False, radius=310.559):
    """
    Returns a list of percent affected values (within a given radius) for
    use in applying a heatmap to a scatterplot.

    Keyword arguments:
    total_cpec_coords -- a 2D array of all CPEC coordinates
    affected_cpec_coords -- a 2D array of affected CPEC coordinates
    radius -- radius to calculate percent of affected CPEC (default 310.559)
    """
    ac = KDTree(affected_cpec_coords)
    tc = KDTree(total_cpec_coords)

    total = tc.query_radius(total_cpec_coords, r=radius, count_only=True)
    affected = ac.query_radius(total_cpec_coords, r=radius, count_only=True)
    percentage_affected = affected / total
    for i in range(len(percentage_affected)):
        if percentage_affected[i] > 1:
            percentage_affected[i] = 1
    if density:
        return percentage_affected, total
    else:
        return percentage_affected


def plot_xy_hist(total_cpec_coords, percentage_affected, xy_savefile,
                 hist_savefile, xy=True, hist=True, axes=True):
    """
    Returns and saves a xy plot of CPEC coordinates with local percent
    affected heatmap and histogram.

    Keyword arguments:
    total_cpec_coords -- a 2D array of all CPEC coordinates
    percent_affected -- a list of percent affected values within a radius
    xy_savefile -- string; save filename for the xy plot
    hist_savefile -- string; save filename for the histogram
    XY -- Boolean; whether to plot a xy plot (default True)
    HIST -- Boolean; whether to plot a histogram (default True)
    axes -- Boolean; whether to plot axes for the xy plot
    """
    z = percentage_affected
    if xy:
        # Plot XY plot

        # Plot CPEC coords
        x = total_cpec_coords[:, 0]
        y = total_cpec_coords[:, 1]

        # local percent affected heatmap
        fig, ax = plt.subplots()
        ax.scatter(x, y, c=z, vmin=0, vmax=1,cmap='inferno', s=100, marker=".", edgecolor='')
        plt.axis('scaled')
        if not axes:
            plt.axis('off')
        plt.gca().invert_yaxis()
        plt.autoscale(False)
        default_size = fig.get_size_inches()
        fig.set_size_inches((default_size[0] * 5, default_size[1] * 5))
        plt.savefig(xy_savefile, transparent=True, bbox_inches='tight', pad_inches=0)
    if hist:
        # Plot Histogram
        font = {'family': 'normal',
                'weight': 'bold',
                'size': 22}
        matplotlib.rc('font', **font)
        plt.figure(figsize=(15, 15))
        plt.hist(z, bins=23, range=(0, 1), ec='black')
        plt.savefig(hist_savefile, transparent=True, bbox_inches='tight', pad_inches=0)


def montecarlo_histogram_overlay(x, y, filename, n_sims=1):
    """
    Generates a overlay histogram.
    :param filename: save filename
    :type filename:
    :param x: observed percent affected values
    :type x:
    :param y: montecarlo percent affected values
    :type y:
    :return:
    :rtype:
    """
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 22}
    matplotlib.rc('font', **font)
    fig, ax1 = plt.subplots(figsize=(15, 15))

    #ax2 = ax1.twinx()
    sim = np.histogram(y.flatten(), bins=23, range=(0,1))
    ax1.hist(sim[1][:-1], bins=sim[1], weights=sim[0]/n_sims, color='tab:grey', label='Monte Carlo', alpha=0.7, ec='black')
    ax1.hist(x, bins=23, range=(0, 1), label='Observed', alpha=0.6, ec='black')
    fig.legend(loc=(0.7, 0.85))

    plt.savefig(filename, transparent=True, bbox_inches='tight', pad_inches=0)


def homogeneity_monte_carlo(affected, total, iterations, radius=310.559):
    """
    Returns monte_carlo simulations of local percent affected values.

    Keyword arguments:
    affected -- a 2D array of affected CPEC coordinates
    total -- a 2D array of all CPEC coordinates
    iterations -- # of simulations to perform
    radius -- radius to calculate percent of affected CPEC (default 310.559)
    """
    simulated_densities = []
    k = len(affected)
    totallist = total.tolist()
    counter = 0
    for i in range(iterations):
        sim_affected = random.sample(totallist, k)
        sim_affected = np.stack(sim_affected)
        affected_tree = KDTree(sim_affected)
        total_tree = KDTree(total)
        nearby_affected = affected_tree.query_radius(total, r=radius, count_only=True)
        nearby_total = total_tree.query_radius(total, r=radius, count_only=True)
        density = nearby_affected / nearby_total
        simulated_densities.append(density)
        counter += 1
        print(counter)
    simulated_densities = np.stack(simulated_densities, axis=0)
    return simulated_densities


def sort_affected_coords_from_aipredictions(aipredictions, coords):
    """
    Sorts out affected coordinates from total coordinates based on
    ai predictions

    Keyword arguments:
    aipredictions -- a list or array of binary predictions (1=affected)
    coords -- an array of coordinates
    """
    affected = []
    for i in range(len(aipredictions)):
        if aipredictions[i] == 1:
            affected.append(i)
    affected_coords = coords[affected]
    return affected_coords


def kolmogorov_smirnov_statistic_for_wsi_montecarlo_simulations(obs_pa, simulated_pa, start=None, stop=None,
                                                                observed_only=False, test_observed=False):
    # Todo: add option to provide variable instead of filename
    sim_dvalues = []
    sim_pvalues = []
    observed = np.load(obs_pa)
    obs_reshape = observed.copy()
    obs_reshape.shape = (1, observed.shape[0])
    simulations = np.load(simulated_pa)
    obs_and_sims = np.concatenate([obs_reshape, simulations]).flatten()
    counter = 0
    if test_observed:
        obs_dvalue, obs_pvalue = scipy.stats.ks_2samp(observed, obs_and_sims)
        print('Observed data D-value:', obs_dvalue, 'Observed data P-value:', obs_pvalue)
        if observed_only:
            return obs_dvalue, obs_pvalue
    if not start:
        start = 0
    if not stop:
        stop = len(simulations)
    for i in range(start, stop):
        d, p = scipy.stats.ks_2samp(simulations[i], obs_and_sims)
        sim_dvalues.append(d)
        sim_pvalues.append(p)
        counter += 1
        print(counter, 'D-value:', sim_dvalues[counter - 1], 'P-value:', sim_pvalues[counter - 1])
    if test_observed:
        return obs_dvalue, obs_pvalue, sim_dvalues, sim_pvalues
    else:
        return sim_dvalues, sim_pvalues


def plot_training(history):
    """
    Diplays a training history graph based on accuracy. Good for classifiers.

    Keyword arguments:
    history: history object that keras returns during training.
    """
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])

    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def retinanet_ious(validation_data_dict, model, boundingbox):
    ious = {
        'med': [],
        'p25': [],
        'p75': []}

    box = model.predict(validation_data_dict)
    if type(box) is list:
        box = {k: l for k, l in zip(model.output_names, box)}

    pred_anchors, _ = boundingbox.convert_box_to_anc(box)
    true_anchors, _ = boundingbox.convert_box_to_anc(validation_data_dict)

    curr = []
    for pred, true in zip(pred_anchors, true_anchors):
        for p in pred:
            iou = boundingbox.calculate_ious(box=p, anchors=true)
            if iou.size > 0:
                curr.append(np.max(iou))
            else:
                curr.append(0)
    if len(curr) == 0:
        curr = [0]

    ious['med'].append(np.median(curr))
    ious['p25'].append(np.percentile(curr, 25))
    ious['p75'].append(np.percentile(curr, 75))

    return {k: np.array(v) for k, v in ious.items()}


def pca_analysis(data, n_comp=3, graph=True, pd_database=False):
    # TODO: add capability to provide a numpy array for data
    # assumes 0 index column is sample_name/Target
    if pd_database:
        df = data
    else:
        df = pd.read_excel(data)
    morphologies = list(df.columns)[1:]
    x = df.loc[:, morphologies].values
    y = df.loc[:, [df.columns[0]]].values
    pca = sklearn.decomposition.PCA(n_components=n_comp)
    principalcomponents = pca.fit_transform(x)
    component_names = [f'principal component {i+1}' for i in range(n_comp)]
    principal_df = pd.DataFrame(data=principalcomponents, columns=component_names)
    final_df = pd.concat([principal_df, df[[df.columns[0]]]], axis=1)
    if graph:
        targets = list(np.unique(df[df.columns[0]]))
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('principal component 1', fontsize=15)
        ax.set_ylabel('principal component 2', fontsize=15)
        ax.set_title('2 component PCA', fontsize=20)
        for target in targets:
            indices_to_keep = final_df[df.columns[0]] == target
            ax.scatter(final_df.loc[indices_to_keep, component_names[0]],
                       final_df.loc[indices_to_keep, component_names[1]],
                       s=50)
        ax.legend(targets, bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
        ax.grid()
        return final_df, fig
    else:
        return final_df


def plot_pca(data, pc, _3d=False, elev=None, azim=None, labels=False, legend=True):
    # data must be a pandas dataframe with principal components and targets
    # pc is a tuple of which principal components to plot (1 index)
    targets = list(np.unique(data['Target']))
    fig = plt.figure(figsize=(8, 8))
    if _3d:
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.set_xlabel(f'principal component {pc[0]}', fontsize=15)
        ax.set_ylabel(f'principal component {pc[1]}', fontsize=15)
        ax.set_zlabel(f'principal component {pc[2]}', fontsize=15)
        ax.set_title('3 component PCA', fontsize=20)
        for target in targets:
            indices_to_keep = data['Target'] == target
            ax.scatter(data.loc[indices_to_keep, f'principal component {pc[0]}'],
                       data.loc[indices_to_keep, f'principal component {pc[1]}'],
                       data.loc[indices_to_keep, f'principal component {pc[2]}'],
                       s=50)
            if labels:
                label = str(data.loc[indices_to_keep, ['Target']].values[0][0])
                ax.annotate(label, (data.loc[indices_to_keep, f'principal component {pc[0]}'],
                                    data.loc[indices_to_keep, f'principal component {pc[1]}']),
                            textcoords='offset points',
                            xytext=(0, 10),
                            ha='center')
        if legend:
            ax.legend(targets, bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
        ax.grid()
        ax.view_init(elev=elev, azim=azim)
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel(f'principal component {pc[0]}', fontsize=15)
        ax.set_ylabel(f'principal component {pc[1]}', fontsize=15)
        ax.set_title('2 component PCA', fontsize=20)
        for target in targets:
            indices_to_keep = data['Target'] == target
            ax.scatter(data.loc[indices_to_keep, f'principal component {pc[0]}'],
                       data.loc[indices_to_keep, f'principal component {pc[1]}'],
                       s=50)
            if labels:
                label = str(data.loc[indices_to_keep, ['Target']].values[0][0])
                ax.annotate(label, (data.loc[indices_to_keep, f'principal component {pc[0]}'],
                                    data.loc[indices_to_keep, f'principal component {pc[1]}']),
                            textcoords='offset points',
                            xytext=(0, 10),
                            ha='center')
        if legend:
            ax.legend(targets, bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
        ax.grid()
    return fig
