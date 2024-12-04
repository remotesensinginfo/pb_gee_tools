#!/usr/bin/env python

# This file is part of 'pb_gee_tools'
#
# Copyright 2024 Pete Bunting
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# Purpose:  Get access to datasets.
#
# Author: Pete Bunting
# Email: pfb@aber.ac.uk
# Date: 17/07/2025
# Version: 1.0
#
# History:
# Version 1.0 - Created.

from typing import List, Tuple
import datetime


def do_dates_overlap(
    s1_date: datetime.datetime,
    e1_date: datetime.datetime,
    s2_date: datetime.datetime,
    e2_date: datetime.datetime,
):
    """
    A function which checks if two dated periods overlap.

    :param s1_date: start of period 1
    :param e1_date: end of period 1
    :param s2_date: start of period 2
    :param e2_date: end of period 2

    :return: boolean - True the two periods overlap, False otherwise
    """
    latest_start = max(s1_date, s2_date)
    earliest_end = min(e1_date, e2_date)
    delta = (earliest_end - latest_start).days + 1
    overlap = max(0, delta)
    return overlap > 0


def create_year_month_start_end_lst(
    start_year: int, start_month: int, end_year: int, end_month: int
) -> List[Tuple[int, int]]:
    """
    A function which creates a list of year and month tuples from a start
    and end month and year.

    :param start_year: int with the start year
    :param start_month: int with the start month
    :param end_year: int with the end year
    :param end_month: int with the end month
    :return: List of tuples (year, month)

    """
    import numpy

    out_year_mnt_lst = list()
    years = numpy.arange(start_year, end_year + 1, 1)
    for year in years:
        if (year == start_year) and (year == end_year):
            months = numpy.arange(start_month, end_month + 1, 1)
        elif year == start_year:
            months = numpy.arange(start_month, 13, 1)
        elif year == end_year:
            months = numpy.arange(1, end_month + 1, 1)
        else:
            months = numpy.arange(1, 13, 1)
        for month in months:
            out_year_mnt_lst.append((year, month))

    return out_year_mnt_lst

def create_year_month_n_months_lst(
    start_year: int, start_month: int, n_months: int
) -> List[Tuple[int, int]]:
    """
    A function which creates a list of year and month tuples from a start
    and end month and year.

    :param start_year: int with the start year
    :param start_month: int with the start month
    :param n_months: int with the number of months ahead
    :return: List of tuples (year, month)

    """
    import numpy

    if start_year < 0:
        raise Exception("Year must be positive")
    if (start_month < 1) or (start_month > 12):
        raise Exception("Month must be between 1-12")

    out_year_mnt_lst = list()
    out_year_mnt_lst.append((start_year, start_month))
    months = numpy.arange(0, n_months, 1)
    months = months + start_month

    month_vals = months % 12
    year = start_year
    first = True
    for month in month_vals:
        if first:
            out_year_mnt_lst.append((year, month + 1))
            first = False
        else:
            if month == 0:
                year += 1
            out_year_mnt_lst.append((year, month + 1))

    return out_year_mnt_lst

def find_month_end_date(year: int, month: int) -> int:
    """
    A function which returns the date of the last day of the month.

    :param year: int for the year (e.g., 2019)
    :param month: month (e.g., 9)
    :return: last day of the month date

    """
    import calendar
    import numpy

    cal = calendar.Calendar()
    month_days = cal.monthdayscalendar(year, month)
    max_day_month = numpy.array(month_days).flatten().max()
    return max_day_month
