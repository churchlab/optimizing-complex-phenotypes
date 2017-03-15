"""
Utility functions for fitting models to data.

Currently just linear regression.
"""

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import sklearn
from sklearn import linear_model


def build_model_df_given_snp_position_list(
        model, snp_position_list, feature_key='POSITION'):
    assert len(snp_position_list) == len(model.coef_), (
            len(snp_position_list), len(model.coef_))

    model_data_dict_list = []
    for snp_idx in range(len(snp_position_list)):
        position = snp_position_list[snp_idx]
        coef = model.coef_[snp_idx]
        model_data_dict_list.append({
            'model_coef': coef,
            feature_key: position
        })
    return pd.DataFrame(model_data_dict_list)


def build_model_df(model, allele_df, feature_key='POSITION'):
    """Builds a DataFrame mapping from coefficient to SNP position.

    Assumes model coefficients in same order as occurrence matrix.

    Args:
        model: Model fit by sklearn.
        allele_df: DataFrame with allele values. NOTE: First n elements of
            DataFrame index must correspond to snp positions.
    """
    assert len(model.coef_) == len(allele_df.columns)
    snp_position_list = allele_df.columns
    return build_model_df_given_snp_position_list(
            model, snp_position_list, feature_key=feature_key)


def fit_model(
        model_maker, matrix, doubling_time_array,
        snp_position_list):
    """DEPRECATED Use fit_model_return_metadata.
    """
    metadata = fit_model_return_metadata(
        model_maker, matrix, doubling_time_array,
        snp_position_list)
    return metadata['df']


def fit_model_return_metadata(
        model_maker, matrix, doubling_time_array,
        snp_position_list):
    """Fits model and returns dictionary with keys:
        * model: sklearn model object.
        * df: DataFrame with fit data.
    """
    lm = model_maker.fit(matrix, doubling_time_array)
    model_df = build_model_df_given_snp_position_list(lm, snp_position_list)
    return {
        'model': lm,
        'df': model_df

    }


def generate_sample_to_signal_pivot_table(input_df, feature_key='POSITION'):
    """Generates a DataFrame indicating occurrence of each model feature (SNP)
    for each sample.

    0 = same as starting strain at that position
    1 = reverted or de novo relative to the starting strain at that position

    Rows are sorted by BARCODE.

    Args:
        input_df: Contains at least columns"
            * BARCODE
            * POSITION
            * signal_relative_to_C321
        feature_key: Name for the key/col in the table that represents the
            the features in our model.

    Returns a DataFrame indexed by BARCODE (rows) and each column represents
    a feature. Values are all 0's or 1's.
    """
    assert not (
            set(['BARCODE', feature_key, 'signal_relative_to_C321']) -
            set(input_df))

    pivot_table_df = (
            input_df.pivot_table(
                    index='BARCODE',
                    columns=[feature_key],
                    values='signal_relative_to_C321'))

    # Get rid of columns that are entirely 0.
    pivot_table_df = pivot_table_df[
            pivot_table_df.columns[pivot_table_df.apply(sum) != 0]]
    return pivot_table_df


def get_doubling_times_array(
        source_data_df, barcode_filter=[], doubling_time_key='doubling_time'):
    """Returns an array of doubling times ordered by barcode.

    Args:
        source_data_df: DataFrame containing, among other stuff,
            BARCODE to doubling_time mapping.
    """
    barcode_to_doubling_time_df = source_data_df[[
            'BARCODE', doubling_time_key]].drop_duplicates()

    # Keep only specified barcodes.
    if len(barcode_filter):
        barcode_filter_set = set(barcode_filter)
        barcode_to_doubling_time_df = barcode_to_doubling_time_df[
                barcode_to_doubling_time_df['BARCODE'].apply(
                        lambda b: b in barcode_filter_set)]

    barcode_to_doubling_time_df.sort_values('BARCODE', inplace=True)
    return np.array(barcode_to_doubling_time_df[doubling_time_key])


def apply_elastic_net_cv(
        snp_occurrence_matrix_df,
        doubling_times,
        add_annotation_metadata_fn=None,
        l1_ratio_list=[.1, .5, .7, .9, .95, .99, 1],
        test_size=20,
        feature_key='POSITION',
        max_iter=1000):
    """Applies Elastic Net CV, returning model.

    Leaves out test_size samples for scoring the fit.
    """
    snp_occurrence_matrix = snp_occurrence_matrix_df.as_matrix()
    X_train, X_test, y_train, y_test = (
            sklearn.model_selection.train_test_split(
                    snp_occurrence_matrix, doubling_times, test_size=test_size))
    elastic_net_cv_maker = sklearn.linear_model.ElasticNetCV(
            l1_ratio=l1_ratio_list, cv=5, n_jobs=3, max_iter=max_iter)
    elastic_net_cv_model = elastic_net_cv_maker.fit(X_train, y_train)
    elastic_net_cv_model_df = build_model_df(
            elastic_net_cv_model, snp_occurrence_matrix_df,
            feature_key=feature_key)
    if add_annotation_metadata_fn:
        elastic_net_cv_model_df = add_annotation_metadata_fn(
                elastic_net_cv_model_df)
    return {
            'model': elastic_net_cv_model,
            'df': elastic_net_cv_model_df,
            'score': elastic_net_cv_model.score(X_test, y_test),
            'l1_ratio': elastic_net_cv_model.l1_ratio_,
            'alpha': elastic_net_cv_model.alpha_,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
    }


def multiple_apply_elastic_net_cv(
        sample_to_snp_occurrence_df,
        doubling_times,
        add_annotation_metadata_fn=None,
        l1_ratio_list=[.1, .5, .7, .9, .95, .99, 1],
        repeats=1,
        test_size=20,
        feature_key='POSITION',
        max_iter=1000):
    """Performs multiple rounds of elastic net regression.

    NOTE: Different runs of ElasticNetCV yield slightly different ranking,
    depending on the train/test split for cross-validation.

    For each SNP, having the list is useful in itself. The coefficient ranks
    give us additional information for being able to get a better overall
    ranking.

    Args:
        TODO: Complete.
        feature_key: Name of the key/col in the DataFrame that stores the
            the features for the model.

    Returns dictionary with keys:
        * snp_to_ranking_list_dict: Map from SNP position to list of rankings
            over multiple repeats of linear regression.
        * snp_to_coef_list_dict: Map from SNP position to list of coefficients
            over multiple repeats of linear regression.
        * alpha_list: List of alpha values corresponding to each repeat.
        * l1_ratio_list: List of l1_ratios chosen over multiple repeats.
        * score_list: List of scores, one per repeat.
    """
    result = {
        'snp_to_ranking_list_dict': defaultdict(list),
        'snp_to_coef_list_dict': defaultdict(list),
        'alpha_list': [],
        'l1_ratio_list': [],
        'score_list': [],
        'model_list': [],
        'intercept_list': [],
        'X_train_list': [],
        'X_test_list': [],
        'y_train_list': [],
        'y_test_list': [],
    }
    for i in range(repeats):
        fit_data = apply_elastic_net_cv(
                sample_to_snp_occurrence_df,
                doubling_times,
                add_annotation_metadata_fn=add_annotation_metadata_fn,
                l1_ratio_list=l1_ratio_list,
                test_size=test_size,
                feature_key=feature_key,
                max_iter=max_iter)

        # Sort coefficients for ranking.
        # NOTE: We used to filter out non-zero values here.
        sorted_filtered_df = fit_data['df'].sort_values(by='model_coef')

        # Record SNP model coefficients.
        for idx, row in sorted_filtered_df.iterrows():
            result['snp_to_coef_list_dict'][row[feature_key]].append(
                    row['model_coef'])

        # Record rank. For non-negative values, we record None, to allow
        # ignoring these when calculating mean.
        order_counter = 1
        for idx, row in sorted_filtered_df.iterrows():
            if row['model_coef'] < 0:
                result['snp_to_ranking_list_dict'][row[feature_key]].append(
                        order_counter)
            else:
                result['snp_to_ranking_list_dict'][row[feature_key]].append(None)
            order_counter += 1

        # Other useful data.
        result['alpha_list'].append(fit_data['alpha'])
        result['l1_ratio_list'].append(fit_data['l1_ratio'])
        result['score_list'].append(fit_data['score'])
        result['model_list'].append(fit_data['model'])
        result['intercept_list'].append(fit_data['model'].intercept_)
        result['X_train_list'].append(fit_data['X_train'])
        result['X_test_list'].append(fit_data['X_test'])
        result['y_train_list'].append(fit_data['y_train'])
        result['y_test_list'].append(fit_data['y_test'])

    return result


def plot_predicted_vs_observed_given_coefficients(
        lm_result_df, allele_occur_matrix_df, actual_doubling_times,
        starting_strain_dt, color_sample_points=[],
        color_sample_points_label=None,
        window_override=None, units='min', plot_fit_line=True):
    """Plots prediction vs observation for model and a fit line.

    NOTE: Code adopted from Experiment 1 Repeat Analysis.
    """
    print 'Num positions in model', len(lm_result_df)

    sample_to_snp_occurrence_matrix_gt_type = allele_occur_matrix_df.as_matrix().transpose()

    # Get the positions and weights, ordered by position.
    sorted_coef_list = list(lm_result_df['model_coef'])

    # Predict doubling times by multiplying weighted coefficients by observations.
    predicted_doubling_times = (
            np.dot(sample_to_snp_occurrence_matrix_gt_type, sorted_coef_list)
            + starting_strain_dt)

    # Add the data to a DataFrame to return.
    report_df = pd.DataFrame({
        'sample_well': allele_occur_matrix_df.columns,
        'predicted': predicted_doubling_times,
        'observed': actual_doubling_times
    })

    # Figure out line fit.
    slope, intercept, r_value, p_value, std_err = stats.linregress(
            predicted_doubling_times, actual_doubling_times)

    plt.figure()

    # Plot measurements.
    plt.plot(
            report_df['predicted'],
            report_df['observed'],
            'r.')

    # Calculate window so that everything fits.
    if window_override:
        assert len(window_override) == 2
        min_dt, max_dt = window_override
    else:
        min_dt = min(report_df['predicted'].min(), report_df['observed'].min()) - 5
        max_dt = max(report_df['predicted'].max(), report_df['observed'].max()) + 5

    if plot_fit_line:
        fit_x = np.linspace(min_dt, max_dt, 100)
        fit_y = slope * fit_x + intercept
        plt.plot(fit_x, fit_y)

    # Plot call-out points colored differently.
    ordered_samples = list(allele_occur_matrix_df.columns)
    sample_indeces_to_color = [
            ordered_samples.index(s) for s in color_sample_points]
    colored_x = [report_df['predicted'][i] for i in sample_indeces_to_color]
    colored_y = [report_df['observed'][i] for i in sample_indeces_to_color]
    plt.plot(colored_x, colored_y, 'bo', label=color_sample_points_label)


    plt.xlabel('predicted doubling time (%s)' % units)
    plt.ylabel('observed doubling time (%s)' % units)
    plt.xlim([min_dt, max_dt])
    plt.ylim([min_dt, max_dt])

    plt.legend(numpoints=1, loc=2, frameon=True)

    ax = plt.axes()

    # Style: Get rid of top and right borders.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.show()

    return report_df


def plot_predicted_vs_observed_given_modeling_result(
        lm_result_df, allele_occur_matrix_df, actual_doubling_times,
        color_sample_points=[], color_sample_points_label=None,
        window_override=None, units='min', plot_fit_line=True,
        log_space=False):
    """Plots prediction vs observation for model and a fit line.

    NOTE: Copied/modified from plot_predicted_vs_observed_given_coefficients.
    """
    print 'Num positions in model', len(lm_result_df['model'].coef_)

    sample_to_snp_occurrence_matrix_gt_type = (
            allele_occur_matrix_df.as_matrix().transpose())

    predicted_doubling_times = lm_result_df['model'].predict(
            sample_to_snp_occurrence_matrix_gt_type)

    score = lm_result_df['model'].score(
            sample_to_snp_occurrence_matrix_gt_type, actual_doubling_times)

    if log_space:
        predicted_doubling_times = np.exp(predicted_doubling_times)
        actual_doubling_times = np.exp(actual_doubling_times)

    # Add the data to a DataFrame to return.
    report_df = pd.DataFrame({
        'sample_well': allele_occur_matrix_df.columns,
        'predicted': predicted_doubling_times,
        'observed': actual_doubling_times
    })

    # Figure out line fit.
    slope, intercept, r_value, p_value, std_err = stats.linregress(
            predicted_doubling_times, actual_doubling_times)

    plt.figure()

    # Plot measurements.
    plt.plot(
            report_df['predicted'],
            report_df['observed'],
            'r.')

    # Calculate window so that everything fits.
    if window_override:
        assert len(window_override) == 2
        min_dt, max_dt = window_override
    else:
        min_dt = min(report_df['predicted'].min(), report_df['observed'].min()) - 5
        max_dt = max(report_df['predicted'].max(), report_df['observed'].max()) + 5

    if plot_fit_line:
        fit_x = np.linspace(min_dt, max_dt, 100)
        fit_y = slope * fit_x + intercept
        plt.plot(fit_x, fit_y)

    # Plot call-out points colored differently.
    if len(color_sample_points):
        ordered_samples = list(allele_occur_matrix_df.columns)
        sample_indeces_to_color = [
                ordered_samples.index(s) for s in color_sample_points]
        colored_x = [report_df['predicted'][i] for i in sample_indeces_to_color]
        colored_y = [report_df['observed'][i] for i in sample_indeces_to_color]
        plt.plot(colored_x, colored_y, 'bo', label=color_sample_points_label)

    plt.xlabel('predicted doubling time (%s)' % units)
    plt.ylabel('observed doubling time (%s)' % units)
    plt.xlim([min_dt, max_dt])
    plt.ylim([min_dt, max_dt])

    plt.title('R_squared: %f' % score)

    plt.legend(numpoints=1, loc=2, frameon=True)

    ax = plt.axes()

    # Style: Get rid of top and right borders.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.show()

    return report_df


def plot_train_test_given_iteration_index(
        multiple_lin_reg_run_result, idx, units='min', savefig_filename=None,
        log_space=False):
    """Plots train-test data for given idx in the multiple results.

    This is useful for plotting a single result and distinguishing test points
    from training points.

    Returns:
        Dataframe with paired predicted and observed data, labeled as TRAIN
        or TEST.
    """
    # Aggregate result objects. Convert to DataFrame at the end and return.
    result_obj_list = []

    plt.figure()

    model = multiple_lin_reg_run_result['model_list'][idx]

    # Training data.
    X_train = multiple_lin_reg_run_result['X_train_list'][idx]
    y_train_observed = multiple_lin_reg_run_result['y_train_list'][idx]
    y_train_predicted = model.predict(X_train)

    if log_space:
        y_train_observed = np.exp(y_train_observed)
        y_train_predicted = np.exp(y_train_predicted)

    for y_train_el_pred, y_train_el_obs in zip(y_train_predicted, y_train_observed):
        result_obj_list.append({
            'predicted': y_train_el_pred,
            'observed': y_train_el_obs,
            'type': 'TRAIN'
        })
    plt.plot(y_train_predicted, y_train_observed, 'y.', label='training')

    # Test data.
    X_test = multiple_lin_reg_run_result['X_test_list'][idx]
    y_test_observed = multiple_lin_reg_run_result['y_test_list'][idx]
    y_test_predicted = model.predict(X_test)
    score = model.score(X_test, y_test_observed)

    if log_space:
        y_test_observed = np.exp(y_test_observed)
        y_test_predicted = np.exp(y_test_predicted)

    for y_test_el_pred, y_test_el_obs in zip(y_test_predicted, y_test_observed):
        result_obj_list.append({
            'predicted': y_test_el_pred,
            'observed': y_test_el_obs,
            'type': 'TEST'
        })
    plt.plot(
            y_test_predicted, y_test_observed, 'b.',
            label='test ($R^2 = %.2f$)' % score)

    plt.xlabel('predicted doubling time (%s)' % units)
    plt.ylabel('observed doubling time (%s)' % units)
    plt.legend(numpoints=1, loc=0, frameon=False)
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    if savefig_filename:
        plt.savefig(savefig_filename, dpi=1000)

    plt.show()

    result_df = pd.DataFrame(result_obj_list)
    result_df['r_squared'] = score
    return result_df[['type', 'predicted', 'observed', 'r_squared']]


def single_snp_linear_modeling(
        snp_occurrence_matrix_df,
        doubling_times):
    """
    Fit a simple linear model to each snp individually, ala
    quantitative trait GWAS.

    Perform a Bonferroni correction on the p-value.

    Returns a DataFrame with results of applying linear modeling with respect
    to occurrence of a single SNP at a time.
    """

    # perform linear regressions per snp
    linregress_array = np.apply_along_axis(
            stats.linregress,
            axis=0,
            arr=snp_occurrence_matrix_df,
            y=doubling_times)

    # pandas dataframe of linear regressions per snp
    linregress_df = pd.DataFrame(
            np.transpose(linregress_array),
            columns=['gradient', 'intercept', 'r', 'gwas_p', 'stderr'])

    # add a predicted effect column that is 1 if no effect
    linregress_df['gwas_coef'] = linregress_df['gradient'] + 1

    linregress_df['gwas_coef'][
            linregress_df['gwas_coef'].isnull()] = 1

    # Bonferroni correction
    linregress_df['gwas_p'] = linregress_df['gwas_p'] * doubling_times.shape[0]
    linregress_df['gwas_p'][linregress_df['gwas_p'] > 1] = 1

    return linregress_df
