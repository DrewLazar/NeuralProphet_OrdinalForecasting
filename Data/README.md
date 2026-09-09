# Data

Two hourly ordinal series. In both, categories are formed by binning an underlying
continuous quantity, and the intervals are left-open and right-closed, so a value
equal to a cut-point falls in the lower category.

## `BeijingAQI_data.csv`

Hourly Beijing air quality, 12 January 2014 06:00 to 9 June 2014 01:00, *n* = 3548,
no gaps. Derived from the UCI Beijing PM2.5 data set.

| column | description |
|---|---|
| `datetime` | timestamp, hourly |
| `pm25` | PM2.5 concentration (µg/m³) |
| `DEWP` | dew point |
| `TEMP` | temperature |
| `PRES` | pressure |
| `cbwd` | combined wind direction |
| `Iws` | cumulated wind speed |
| `Is` | cumulated hours of snow |
| `Ir` | cumulated hours of rain |
| `aqi_category` | ordinal category, 1–4 |

`aqi_category` is derived from `pm25` using the PM2.5 cut-points of HJ 633-2012 at
35, 75 and 150 µg/m³: category 1 is ≤ 35, then (35, 75], (75, 150], and > 150.
Category shares are 23.5, 20.8, 31.8 and 23.9 percent.

The file retains the original UCI columns. The model uses `TEMP`, `DEWP`, `PRES` and
`Iws` as covariates; `cbwd`, `Is` and `Ir` are not used.

## `energy_data_excerpt.csv`

Hourly electricity load at Ball State University's Foundational Sciences Building,
1 September to 30 November 2024, *n* = 2184. Used with the permission of Ball State
University Facilities Planning and Management.

| column | description |
|---|---|
| `datetime` | timestamp, hourly |
| `temp` | outdoor temperature (°F) |
| `is_not_weekend` | 1 on weekdays |
| `in_session` | 1 when classes are in session |
| `is_not_holiday` | 1 on non-holidays |
| `is_not_summerbreak` | 1 outside summer break |
| `load_category` | ordinal category, 1–4 |

**The kilowatt readings are not included and are not redistributed.** `load_category`
was formed from building load at the mean and mean ± 0.7 standard deviations, giving
cut-points at 370, 415 and 460 kW: category 1 is ≤ 370, then (370, 415], (415, 460],
and > 460. Category shares in this excerpt are 31.7, 16.9, 22.8 and 28.5 percent.

The series is the combined reading of the two meters serving the building. No
campus-wide or consolidated metering is involved.

This is a three-month excerpt of the record analysed in the paper, which runs from
1 April 2024 to 28 July 2025. It is enough to run the method end to end on the load
data, but it does not reproduce the load results reported in the paper. Requests for
the underlying meter data should be directed to Ball State University Facilities
Planning and Management.
