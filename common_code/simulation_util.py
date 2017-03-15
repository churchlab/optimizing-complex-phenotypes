"""
Utility functions for experiment design simulations.
"""

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from . import model_fitting


# Number of samples from cycling, selected at regular
# intervals throughout cycling.
DEFAULT_NUM_SAMPLES = 100

MAGE_CYCLES = 50

MAGE_EFFICIENCY = 0.05

DEFAULT_SNPS_CONSIDERED = 120

DEFAULT_SNPS_WITH_EFFECT = 5

MAX_ADDITIIVE_FITNESS_EFFECT = 0.5

# Internal parameter for population size of bacteria to maintain.
# We want this as big as possible without being computationally
# intractable.
DEFAULT_POPULATION_SIZE = 10000

# Determines how many "doublings" (growth to mid-log) are allowed
# between MAGE cycles.
POPULATION_GROWTH_FACTOR_BEFORE_SELECTION = 100

METRIC_KEYS = [
    'pearson_r',
    'pearson_p',
    'true_positives',
    'false_positives',
    'true_negatives',
    'false_negatives',
    'precision',
    'recall',
    'specificity',
    'false_positive_rate',
    'smallest_effect_detected',
    'largest_effect_detected',
    'total_effect_detected',
    'percent_of_total_effect_detected',
    'weighted_precision',
    'weighted_recall',
]

SIM_RESULT_KEY_ORDER = [
    'num_snps_considered',
    'num_samples',
    'num_snps_with_effect',
    'replicate',
    'mage_cycles',
    'population_size',
    'total_fitness_effect',
]

# These are not calculated.
SKIP_KEYS = [
    'enrichment_pearson_r',
    'enrichment_pearson_p',
    'enrichment_weighted_precision',
    'enrichment_weighted_recall',
]

for prefix in ['lm_', 'gwas_', 'enrichment_']:
    for key in METRIC_KEYS:
        combo_key = prefix + key
        if combo_key in SKIP_KEYS:
            continue
        SIM_RESULT_KEY_ORDER.append(combo_key)


class SimulationParams(object):
    """Container for params passed to simulation.

    Initializes all attributes to default. Clients can override by setting
    attributes directly.
    """
    def __init__(self):
        self.num_samples = DEFAULT_NUM_SAMPLES
        self.num_snps_considered = DEFAULT_SNPS_CONSIDERED
        self.num_snps_with_effect = DEFAULT_SNPS_WITH_EFFECT
        self.population_size = DEFAULT_POPULATION_SIZE
        self.mage_cycles = MAGE_CYCLES

    def __str__(self):
        return 'self.num_samples: %d, self.num_snps_considered: %d, self.num_snps_with_effect: %d, self.population_size: %d, self.mage_cycles: %d' % (
                self.num_samples, self.num_snps_considered, self.num_snps_with_effect, self.population_size, self.mage_cycles)


def run_simulation(
        simulation_params=SimulationParams(),
        snp_effects=None,
        should_apply_selection_pressure=True):
    """Runs simulation of MAGE over many cycles.

    Returns dictionary containing final population, samples taken regularly
    according to parameters, and corresponding doubling times.
    """
    num_samples = simulation_params.num_samples
    num_snps_considered = simulation_params.num_snps_considered
    num_snps_with_effect = simulation_params.num_snps_with_effect
    population_size = simulation_params.population_size
    mage_cycles = simulation_params.mage_cycles

    # Initial population.
    population = np.zeros((population_size, num_snps_considered), dtype=np.bool)

    # Generate SNP effects.
    if snp_effects is None:
        snp_effects = generate_snp_effects(
                num_snps_considered, num_snps_with_effect)
    assert len(snp_effects) == num_snps_considered

    # Determine MAGE cycles at whch we sample, distributing samples
    # regularly throughout mage cycling.
    # We compute 0-indexed cycles, and then subtract from mage_cycles
    # so we're partitioning the latter half.
    samples_per_mage_cycle = float(num_samples) / mage_cycles
    wgs_samples_mage_cycle_list = []
    next_sample_idx = 0
    for cycle in range(mage_cycles):
        # Maybe sample for "whole genome sequencing".
        while cycle * samples_per_mage_cycle >= next_sample_idx:
            if next_sample_idx >= num_samples:
                break
            wgs_samples_mage_cycle_list.append(cycle)
            next_sample_idx += 1
    # Sample from the the latter half.
    wgs_samples_mage_cycle_list = list(reversed(
            mage_cycles - np.array(wgs_samples_mage_cycle_list)))
    while len(wgs_samples_mage_cycle_list) < num_samples:
        wgs_samples_mage_cycle_list.append(mage_cycles)
    assert len(wgs_samples_mage_cycle_list) == num_samples

    # Store the samples for "whole genome sequencing" (read: linear modeling).
    wgs_samples = np.zeros((num_samples, num_snps_considered))

    # Index into wgs_samples rows, the data structure used to store samples
    # that are sampled for "WGS" (linear modeling). Incremented in loop below.
    next_wgs_sample_idx = 0
    next_wgs_sample_mage_cycle = wgs_samples_mage_cycle_list[next_wgs_sample_idx]

    # Run MAGE cycling with sampling for WGS.
    for cycle in range(1, mage_cycles + 1):
        # Perform 1 cycle of MAGE
        population = update_population_with_mage_mutations(population)

        # Apply selection (grow to 100x cells, then dilute 1:100).
        if should_apply_selection_pressure:
            selection_result = apply_selection_pressure(population, snp_effects)
            population = selection_result['updated_population']

        # Maybe sample for "whole genome sequencing".
        while next_wgs_sample_mage_cycle == cycle:
            population_sample_idx = np.random.choice(range(population.shape[0]))
            wgs_samples[next_wgs_sample_idx, :] = population[
                    population_sample_idx, :]
            next_wgs_sample_idx += 1
            if next_wgs_sample_idx >= num_samples:
                break
            next_wgs_sample_mage_cycle = wgs_samples_mage_cycle_list[
                    next_wgs_sample_idx]
    assert next_wgs_sample_idx == num_samples, (
            'next_wgs_sample_idx: %d, next_wgs_sample_mage_cycle: %d, num_samples: %d' % (
                    next_wgs_sample_idx, next_wgs_sample_mage_cycle, num_samples))
    assert len(wgs_samples_mage_cycle_list) == wgs_samples.shape[0]

    # Compute doubling times for final population.
    final_doubling_times = generate_doubling_times(
            population, snp_effects)

    # Compute doubling times for WGS samples.
    wgs_samples_doubling_times = generate_doubling_times(
            wgs_samples, snp_effects)

    return {
        'sim_params': simulation_params,
        'snp_effect': snp_effects,
        'final_population': population,
        'final_doubling_times': final_doubling_times,
        'wgs_samples': wgs_samples,
        'wgs_samples_mage_cycle_list': wgs_samples_mage_cycle_list,
        'wgs_samples_doubling_times': wgs_samples_doubling_times,
    }


def generate_snp_effects(num_snps_considered, num_snps_with_effect):
    """Returns array of snp effects, in order to be used for
    remainder of simulation.
    """
    snp_effects = np.ones(num_snps_considered)
    non_zero_effects = sample_effects_by_power_law(
            num_snps_with_effect)
    snp_effects[:num_snps_with_effect] = non_zero_effects
    np.random.shuffle(snp_effects)
    return snp_effects


def sample_effects_by_power_law(
        num_snps_with_effect,
        total_fitness_effect_mean=0.5, total_fitness_effect_std=0.05,
        rate=0.5,  # empiricaly chosen
        debug_plot=False):
    """Samples SNP effects according to power law and normalizes them to
    given total effect size.
    """
    total_effect_size = np.random.normal(
            loc=total_fitness_effect_mean,
            scale=total_fitness_effect_std)
    log_effects = sorted(
            np.random.power(rate, size=num_snps_with_effect), reverse=True)
    normalized_log_effects = (log_effects / sum(log_effects)) * np.log(total_effect_size)
    effects = sorted(np.exp(normalized_log_effects), reverse=True)

    if debug_plot:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.title('Log Effects (sorted)')
        plt.bar(range(len(log_effects)), log_effects)

        plt.subplot(1, 2, 2)
        plt.title('Effects (sorted)')
        plt.bar(range(len(effects)), effects)
        plt.show()

    return effects


def update_population_with_mage_mutations(
        population, mage_efficiency=MAGE_EFFICIENCY):
    """Returns population updated with mutations at MAGE frequency.

    Each clone in the population can be randomly updated according to MAGE
    efficiency.
    """
    # Generate an update matrix where MAGE efficiency fraction of population
    # gets a new mutation.
    mage_matrix = np.zeros(population.shape, dtype=np.bool)

    num_samples, num_snps_considered = population.shape
    expected_samples_with_new_mutation = int(mage_efficiency * num_samples)

    # Unique samples to update, considering MAGE efficiency.
    update_sample_indeces = random.sample(
            range(num_samples), expected_samples_with_new_mutation)

    # Possibly repeating SNPs to update (we'll do 1 per sample).
    update_snp_indeces = np.random.choice(
            range(num_snps_considered), expected_samples_with_new_mutation)

    for sample_idx, snp_idx in zip(update_sample_indeces, update_snp_indeces):
        mage_matrix[sample_idx][snp_idx] = 1

    # Apply MAGE using bitwise OR.
    return mage_matrix | population


def apply_selection_pressure(population, snp_effects):
    """Simulates competition among cells based on fitness in between
    MAGE cycles.

    Returns a dictionary with keys:
        * updated_population: The new population.
        * metadata_df: DataFrame for debug.
        * subsampled_clone_ids: Ids of clones chosen for next round.

    Returns updated population of same size, with selection applied.

    Allows cells to double until there are 100x more cells.
    """
    doubling_times = generate_doubling_times(
            population, snp_effects)

    # Growth rates inversely proportional to doubling times.
    growth_rates = 1 / doubling_times

    # Increment time periods until there would be 1000x more cells. Then use that
    # time in an exponential increase for each genotype.

    def _compute_population_size_after_t_periods(t):
        new_population_size = sum(np.exp(growth_rates * t))
        # print new_population_size
        return new_population_size

    time_periods = 1
    confluent_population_size = (
            POPULATION_GROWTH_FACTOR_BEFORE_SELECTION * population.shape[0])
    while (_compute_population_size_after_t_periods(time_periods) <
            confluent_population_size):
        assert time_periods < 100  # Sanity check.
        time_periods += 1

    descendent_counts = np.exp(growth_rates * time_periods)

    # Finally keep only POPULATION_SIZE cells moving forward.
    updated_population = np.zeros(population.shape, dtype=np.bool)
    descendent_probabilities = descendent_counts / sum(descendent_counts)
    population_size = population.shape[0]
    subsampled_clone_ids = np.random.choice(
            population_size, size=population_size, p=descendent_probabilities)
    for i, clone_id in enumerate(subsampled_clone_ids):
        updated_population[i, :] = population[clone_id, :]

    # Metadata for debug.
    metadata_df = pd.DataFrame({
        'doubling_times': doubling_times,
        'growth_rates': growth_rates,
        'descendents': descendent_counts
    })

    return {
        'updated_population': updated_population,
        'metadata_df': metadata_df,
        'subsampled_clone_ids': subsampled_clone_ids
    }


def generate_doubling_times(population, snp_effects):
    genotype_effects = population * snp_effects
    genotype_effects[genotype_effects == 0] = 1
    doubling_times = np.prod(genotype_effects, axis=1)
    return doubling_times


def run_linear_modeling(genotype_matrix, doubling_times, repeats=1,
        max_iter=1000):
    genotype_df = pd.DataFrame(genotype_matrix)
    current_elastic_cv_result = model_fitting.multiple_apply_elastic_net_cv(
            genotype_df,
            doubling_times,
            test_size=0.2,
            repeats=repeats,
            max_iter=max_iter)
    return current_elastic_cv_result


def evaluate_modeling_result(simulation_data, lm_result):
    """Evalutes the model.

    Returns dictionary with keys:
        * results_df
        * pearson_r
        * p_value
    """
    results_df = pd.DataFrame({
            'snp_effect': simulation_data['snp_effect']})
    avg_coef_list = []
    for snp in results_df.index:
        coef_list = lm_result['snp_to_coef_list_dict'][snp]
        avg_coef_list.append(np.mean(coef_list))
    results_df['linear_model_coef'] = np.array(avg_coef_list) + 1

    # One metric is Pearson correlation.
    pearson_r, p_value = pearsonr(
            results_df['snp_effect'],
            results_df['linear_model_coef'])

    return {
        'results_df': results_df,
        'pearson_r': pearson_r,
        'p_value': p_value
    }


def run_gwas(genotype_matrix, doubling_times):
    """Runs GWAS.
    """
    linregress_df = model_fitting.single_snp_linear_modeling(
            genotype_matrix, doubling_times)
    return linregress_df


def evaluate_gwas_result(gwas_results_df, lm_results_df, show_plot=False):
    """Evaluates GWAS result and compares to linear modeling result.
    """
    # Compare the results.
    # NOTE: The following code assumes the SNPs from the
    # two analyses are in the same order.
    gwas_vs_lm_comparison_df = pd.concat([
            gwas_results_df[['gwas_p', 'gwas_coef']],
            lm_results_df['snp_effect']], axis=1)

    pearson_r, p_value = pearsonr(
            gwas_vs_lm_comparison_df['snp_effect'],
            gwas_vs_lm_comparison_df['gwas_coef'])

    if show_plot:
        plt.figure()
        plt.scatter(
            x=gwas_vs_lm_comparison_df['gwas_coef'],
            y=gwas_vs_lm_comparison_df['snp_effect'],
            c=1 - (
                    -np.log10(gwas_vs_lm_comparison_df['gwas_p']) /
                    np.max(-np.log10(gwas_vs_lm_comparison_df['gwas_p']))),
            s=200)
        plt.xlabel('GWAS-predicted coefficient')
        plt.ylabel('Simulated SNP effect')
        plt.show()

    return {
        'results_df': gwas_vs_lm_comparison_df,
        'pearson_r': pearson_r,
        'p_value': p_value
    }


def calc_common_metrics(d, calc_weighted=True):
    """Metrics common to modeling and enrichment.

    Args:
        d: Basic data as sets of indeces refering to a DataFrame, including
            sim count, observed counts, TP, FP, etc.
        calc_weighted: If True, calculate weighted metrics.

    Returns a dictionary with common metrics.
    """
    # Validation.
    assert 'snp_effects' in d
    if calc_weighted:
        assert 'model_effects' in d

    d['true_positives'] = d['observed_true'] & d['sim_true']
    d['false_positives'] = d['observed_true'] & d['sim_false']
    d['true_negatives'] = d['observed_false'] & d['sim_false']
    d['false_negatives'] = d['observed_false'] & d['sim_true']

    # Sanity checks.
    assert (len(d['snp_effects']) ==
        len(d['true_positives']) + len(d['false_positives']) +
        len(d['true_negatives']) + len(d['false_negatives']))
    assert (len(d['sim_false']) ==
            len(d['true_negatives']) + len(d['false_positives']))

    assert len(d['observed_true']) == len(d['true_positives']) + len(d['false_positives'])
    if len(d['observed_true']) == 0:
        precision = 1.0
    else:
        precision = float(len(d['true_positives'])) / len(d['observed_true'])

    assert len(d['sim_true']) == len(d['true_positives']) + len(d['false_negatives'])
    recall = float(len(d['true_positives'])) / len(d['sim_true'])


    assert len(d['sim_false']) == len(d['true_negatives']) + len(d['false_positives'])
    specificity = float(len(d['true_negatives'])) / len(d['sim_false'])

    # Additional interesting metrics for analysis.
    snp_effects_detected = d['snp_effects'][d['true_positives']]
    if not len(snp_effects_detected):
        smallest_effect_detected = 1.0
        largest_effect_detected = 1.0
        total_effect_detected = 1.0
    else:
        smallest_effect_detected = sorted(snp_effects_detected, reverse=True)[0]
        largest_effect_detected = sorted(snp_effects_detected)[0]
        total_effect_detected = np.prod(snp_effects_detected)
    percent_of_total_effect_detected = (
            (1 - total_effect_detected) /
            (1 - np.prod(d['snp_effects'])))

    common_results_dict = {
        'sim_true': len(d['sim_true']),
        'sim_false': len(d['sim_false']),
        'observed_true': len(d['observed_true']),
        'observed_false': len(d['observed_false']),
        'true_positives': len(d['true_positives']),
        'false_positives': len(d['false_positives']),
        'true_negatives': len(d['true_negatives']),
        'false_negatives': len(d['false_negatives']),
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'false_positive_rate': 1.0 - specificity,
        'smallest_effect_detected': smallest_effect_detected,
        'largest_effect_detected': largest_effect_detected,
        'total_effect_detected': total_effect_detected,
        'percent_of_total_effect_detected': percent_of_total_effect_detected,
    }

    # Metrics weighted by SNP effect.
    # Precision weighted by modeled coefficients.
    # Recall weighted by simulated values.
    if calc_weighted:
        # Precision.
        if len(d['observed_true']) == 0:
            common_results_dict['weighted_precision'] = 1.0
        elif len(d['true_positives']) == 0:
            common_results_dict['weighted_precision'] = 0.0
        else:
            model_effect_weighted_true_positives = 1 - np.prod(
                    np.array(d['model_effects'][d['true_positives']]))
            model_effect_weighted_observed_true = 1 - np.prod(
                    np.array(d['model_effects'][d['observed_true']]))
            common_results_dict['weighted_precision'] = (
                    model_effect_weighted_true_positives /
                    model_effect_weighted_observed_true)

        # Recall.
        if len(d['true_positives']) == 0:
            common_results_dict['weighted_recall'] = 0.0
        else:
            snp_effect_weighted_true_positives = 1 - np.prod(
                    np.array(d['snp_effects'][d['true_positives']]))
            snp_effect_weighted_sim_true = 1 - np.prod(
                    np.array(d['snp_effects'][d['sim_true']]))
            common_results_dict['weighted_recall'] = (
                    snp_effect_weighted_true_positives /
                    snp_effect_weighted_sim_true)

    return common_results_dict


def calculate_modeling_metrics(
        modeling_result, coef_key, results_prefix='',
        min_bonferroni_corrected_p_value=0.05):
    """Computes various metrics of interest for a given modeling result.
    """
    assert coef_key in ['linear_model_coef', 'gwas_coef']

    sim_true = set(modeling_result[modeling_result['snp_effect'] < 1].index)
    sim_false = set(modeling_result.index) - sim_true

    if coef_key == 'gwas_coef':
        observed_true = set(modeling_result[
                (modeling_result[coef_key] < 1) &
                (modeling_result['gwas_p'] < 0.05)
        ].index)
        observed_false = set(modeling_result.index) - observed_true
    else:
        observed_true = set(
                modeling_result[modeling_result[coef_key] < 1].index)
        observed_false = set(modeling_result.index) - observed_true


    results_dict = {
        'snp_effects': modeling_result['snp_effect'],
        'model_effects': modeling_result[coef_key],
        'sim_true': sim_true,
        'sim_false': sim_false,
        'observed_true': observed_true,
        'observed_false': observed_false,
    }
    results_dict.update(calc_common_metrics(results_dict, calc_weighted=True))

    # Don't need this anymore. Removing to allow saner debug print.
    del results_dict['snp_effects']
    del results_dict['model_effects']

    results_with_prefixed_keys = {}
    for key, value in results_dict.iteritems():
        results_with_prefixed_keys[results_prefix + key] = value
    return results_with_prefixed_keys


def run_enrichment_analysis(simulation_result):
    """Samples from final timepoint and returns DataFrame containing
    enrichment counts and corresponding SNP effect.
    """
    sim_params = simulation_result['sim_params']

    # Sub-sample final population.
    final_timepoint_subsample = np.zeros(
            (sim_params.num_samples, sim_params.num_snps_considered), dtype=np.bool)
    random_indeces_from_final_population = np.random.choice(
            range(len(simulation_result['final_population'])),
            size=sim_params.num_samples)
    final_timepoint_doubling_times = []
    for i, random_idx in enumerate(random_indeces_from_final_population):
        final_timepoint_subsample[i, :] = (
                simulation_result['final_population'][random_idx, :])
        final_timepoint_doubling_times.append(
                simulation_result['final_doubling_times'][random_idx])
    assert len(final_timepoint_doubling_times) == sim_params.num_samples

    final_timepoint_enrichment_df = pd.DataFrame({
        'snp_effect': simulation_result['snp_effect'],
        'enrichment_count': final_timepoint_subsample.sum(axis=0)
    })
    return final_timepoint_enrichment_df


def calculate_enrichment_metrics(enrichment_df, results_prefix='enrichment_'):
    """Calculates metrics for enrichment (e.g. TP, FP, recall, etc.)
    """
    mean_enrichment = enrichment_df['enrichment_count'].mean()

    sim_true = set(enrichment_df[enrichment_df['snp_effect'] < 1].index)
    sim_false = set(enrichment_df.index) - sim_true
    observed_true = set(enrichment_df[
            enrichment_df['enrichment_count'] >= mean_enrichment].index)
    observed_false = set(enrichment_df.index) - observed_true

    results_dict = {
        'snp_effects': enrichment_df['snp_effect'],
        'sim_true': sim_true,
        'sim_false': sim_false,
        'observed_true': observed_true,
        'observed_false': observed_false,
    }
    results_dict.update(calc_common_metrics(results_dict, calc_weighted=False))

    # Don't need this anymore. Removing to allow saner debug print.
    del results_dict['snp_effects']

    results_with_prefixed_keys = {}
    for key, value in results_dict.iteritems():
        results_with_prefixed_keys[results_prefix + key] = value
    return results_with_prefixed_keys


def run_simulation_with_params(
        sim_params, replicate, repeats=10, should_perform_gwas=True):
    """Runs simulation with given params and returns result object.
    """
    try:
        simulation_result = run_simulation(
                simulation_params=sim_params)
    except Exception as e:
        print sim_params
        raise e

    result = {
        'num_snps_considered': sim_params.num_snps_considered,
        'num_samples': sim_params.num_samples,
        'num_snps_with_effect': sim_params.num_snps_with_effect,
        'replicate': replicate,
        'total_fitness_effect': np.prod(simulation_result['snp_effect']),
        'mage_cycles': sim_params.mage_cycles,
        'population_size': sim_params.population_size
    }

    # Apply linear modeling.
    lm_result = run_linear_modeling(
        simulation_result['wgs_samples'],
        simulation_result['wgs_samples_doubling_times'],
        repeats=repeats)
    lm_eval_results = evaluate_modeling_result(
            simulation_result, lm_result)
    lm_eval_results_df = lm_eval_results['results_df']
    result.update({
        'lm_pearson_r': lm_eval_results['pearson_r'],
        'lm_pearson_p': lm_eval_results['p_value'],
    })
    result.update(
            calculate_modeling_metrics(
                    lm_eval_results_df, 'linear_model_coef',
                    results_prefix='lm_'))

    # Maybe perform GWAS.
    if should_perform_gwas:
        gwas_results_df = run_gwas(
            simulation_result['wgs_samples'],
            simulation_result['wgs_samples_doubling_times'])
        gwas_eval_results = evaluate_gwas_result(
                gwas_results_df, lm_eval_results_df)
        gwas_eval_results_df = gwas_eval_results['results_df']
        result.update({
            'gwas_pearson_r': gwas_eval_results['pearson_r'],
            'gwas_pearson_p': gwas_eval_results['p_value'],
        })
        result.update(
                calculate_modeling_metrics(
                        gwas_eval_results_df, 'gwas_coef', results_prefix='gwas_'))

    # Perform enrichment analysis on final timepoint.
    enrichment_result_df = run_enrichment_analysis(simulation_result)
    result.update(
            calculate_enrichment_metrics(
                    enrichment_result_df))

    return result


def visualize_simulation_result(simulation_result):
    """Plots for visualizing results of simulation.
    """
    plt.figure(figsize=(20, 8))

    # SNP effects
    snp_effects = simulation_result['snp_effect']
    non_trivial_snp_effects = sorted([
            e for e in snp_effects if e != 1], reverse=True)
    print 'Total Fitness Defect', np.prod(non_trivial_snp_effects)
    plt.subplot(2, 3, 1)
    plt.title('Sanity Check | Fitness effects %d/%d SNPs' % (
            DEFAULT_SNPS_WITH_EFFECT, DEFAULT_SNPS_CONSIDERED))
    plt.bar(range(len(non_trivial_snp_effects)), non_trivial_snp_effects)

    # Look at distribution of mutations in final population.
    plt.subplot(2, 3, 2)
    plt.title('Final population mutation distribution')
    plt.hist(simulation_result['final_population'].sum(axis=1))

    # Final population doubling times.
    doubling_times = simulation_result['final_doubling_times']
    plt.subplot(2, 3, 3)
    plt.title('Final population doubling times')
    plt.bar(range(len(doubling_times)), sorted(doubling_times, reverse=True))

    # WGS mutation distribution
    plt.subplot(2, 3, 4)
    plt.title('WGS samples mutation distribution')
    plt.hist(simulation_result['wgs_samples'].sum(axis=1))

    # WGS doubling times
    plt.subplot(2, 3, 5)
    plt.title('WGS samples doubling times')
    plt.plot(
            simulation_result['wgs_samples_mage_cycle_list'],
            simulation_result['wgs_samples_doubling_times'],
            '.'
    )

    # Mutations vs MAGE cycle
    plt.subplot(2, 3, 6)
    plt.title('Figure 2b')
    plt.plot(
            simulation_result['wgs_samples_mage_cycle_list'],
            np.sum(simulation_result['wgs_samples'], axis=1),
            '.')
    plt.xlabel('MAGE Cycle')
    plt.ylabel('Reverted Mutations')
    plt.show()
