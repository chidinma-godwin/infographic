#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 00:35:43 2023

@author: Chidex
"""

import os
import shutil

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

import seaborn as sns


def process_data(regions, ncd_mortality, population):
    """
    Cleans the loaded data, process and merge them to a single dataframe

    Parameters
    ----------
    regions : DataFrame
        The dataframe containing countries and the region they fall under.
    ncd_mortality : DataFrame
        The dataframe containg non-communicable disesases deaths by their
        cause and sex from year 2000 to 2019 (20 years).
    population : DataFrame
        The dataframe containing the total number of people in each countries
        from year 2000 to 2019.

    Returns
    -------
    merged_df : DataFrame
        The merged dataframe containing ncd deaths by cause and sex for each
        countries, the regions and population for the respective countries.

    """
    # Clean the non-communicable diseases death data
    ncd_mortality.rename(
        columns={"Countries, territories and areas": "Country",
                 "Both sexes": "Total"}, inplace=True)
    # The number of death is given along with the lower and upper limit, we
    # extract just the number of deaths e.g 15565 [7609-28280] becomes 15565
    gender_columns = ["Total", "Male", "Female"]
    ncd_mortality[gender_columns] = ncd_mortality[gender_columns].apply(
        lambda x: pd.to_numeric(x.str.split("[").str[0]))

    # Clean the population data
    population.drop(
        columns=["Series Code", "Country Name"], inplace=True)
    # Change column names like "2000 [YR2000]" to "2000"
    columns = ["Sex" if name == "Series Name"
               else name.split(" [")[0] for name in list(population.columns)]
    population.columns = columns
    population.loc[population["Sex"] == "Population, female", "Sex"] = "Male"
    population.loc[population["Sex"] == "Population, male", "Sex"] = "Female"
    population.loc[population["Sex"] == "Population, total", "Sex"] = "Total"
    # Reshape the population data
    population = population.melt(id_vars=["Country Code", "Sex"],
                                 var_name='Year', value_name='Population')
    population["Year"] = pd.to_datetime(population["Year"], format="%Y")

    # Merge the ncd mortality and regions data, so that the merged data
    # includes the region for each country
    ncd_regions = pd.merge(ncd_mortality, regions, on="Country")
    # Join Male, Female, and Total columns to a single column
    ncd_regions = ncd_regions.melt(
        id_vars=["Country", "Country Code", "Year", "Causes", "Region"],
        var_name='Sex', value_vars=['Total', 'Male', 'Female'],
        value_name='Number of Deaths')
    ncd_regions["Year"] = pd.to_datetime(ncd_regions["Year"], format="%Y")

    # Merge the population data to the ncd mortality and regions data
    merged_df = pd.merge(ncd_regions, population, on=[
        "Country Code", "Year", "Sex"], sort=True)
    # Create new Deaths per 100,000 population column
    merged_df["Deaths per 100,000"] = (
        merged_df["Number of Deaths"] / merged_df["Population"]) * 100000

    return merged_df


# Read the data with pandas
regions = pd.read_csv("regions.csv")
ncd_mortality = pd.read_csv("NCD-deaths.csv", skiprows=1)
population = pd.read_csv("population.csv")

data = process_data(regions, ncd_mortality, population)
