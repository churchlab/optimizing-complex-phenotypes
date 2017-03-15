"""Utility functions for cleaning up data.
"""

import re

import numpy as np
import pandas as pd


def normalize_well_name(raw_well_name):
    """Returns a normalized version of the well name.

    For normalized, we'll require letter-digit-digit so that
    A6 gets normalized as A06.
    """
    match = re.match(r'([A-Za-z]{1})(\d+)', raw_well_name)
    letter_part = match.group(1)
    number_part = int(match.group(2))
    return letter_part + '{0:02d}'.format(number_part)

# Tests
assert normalize_well_name('A6') == 'A06'
assert normalize_well_name('A06') == 'A06'
assert normalize_well_name('H11') == 'H11'


def get_filtered_alleles_df_that_have_allele(allele_name, alleles_df):
    """Returns reduced alleles DF containing only entries
    that contain the given allele.
    """
    alleles_df_t = alleles_df.transpose()
    alleles_df_t_filtered_has_allele = alleles_df_t[
            alleles_df_t[allele_name] == 1]
    return alleles_df_t_filtered_has_allele.transpose()


def get_doubling_times_for_clones_in_alleles_df(alleles_df, doubling_times_df):
    """Returns filtered doubling times for clones in alleles."""
    return doubling_times_df[
        doubling_times_df.well.apply(
                lambda w: w in list(
                        alleles_df.columns))]


def get_stats_for_allele(allele_name, alleles_df, doubling_times_df):
    """Gets count and mean/stdev doubling time for strains w/ that allele."""
    df_filtered_has_allele = get_filtered_alleles_df_that_have_allele(
            allele_name, alleles_df)
    doubling_times_for_clones_with_allele_df = (
            get_doubling_times_for_clones_in_alleles_df(
                    df_filtered_has_allele, doubling_times_df))
    return {
        'allele': allele_name,
        'count': len(doubling_times_for_clones_with_allele_df),
        'mean': np.mean(doubling_times_for_clones_with_allele_df.doubling_time),
        'stdev': np.std(doubling_times_for_clones_with_allele_df.doubling_time),
    }


def get_per_allele_stats(alleles_df, growth_data_df):
    """Returns DataFrame with growth stats per allele."""
    allele_stats_obj_list = []
    for allele in alleles_df.index:
        allele_stats_obj_list.append(get_stats_for_allele(
                allele,
                alleles_df,
                growth_data_df))
    return pd.DataFrame(allele_stats_obj_list)


def get_doubling_time_for_well(well, growth_data_df):
    """Gets doubling time for a well.

    Assume growth data has columns 'well' and 'doubling_time'.
    """
    assert 'well' in growth_data_df.columns
    assert 'doubling_time' in growth_data_df.columns
    well_data_df = growth_data_df['doubling_time'][
            growth_data_df['well'] == well]
    return well_data_df.values[0]
