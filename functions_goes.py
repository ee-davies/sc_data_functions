import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from spacepy import pycdf
import glob
import os
import urllib.request
from urllib.request import urlopen
import os.path
import pickle
import xarray as xr



import data_frame_transforms as data_transform
import position_frame_transforms as pos_transform
import functions_general as fgen

from functions_general import load_path

"""
GOES DATA PATH
"""

goes_path = load_path(path_name='goes_path')
print(f"GOES data path loaded: {goes_path}")


"""
GOES SGPS DATA
"""
#data from https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes18/l2/data/sgps-l2-avg5m/

def open_netcdf(filepath, show=True): #open file to see variables
    ds = xr.open_dataset(filepath)

    if show:
        print("Dimensions:")
        for name, size in ds.sizes.items():
            print(f"  {name}: {size}")

        print("\nVariables:")
        for name, var in ds.variables.items():
            units = var.attrs.get("units", "")
            desc = var.attrs.get("long_name", "")
            if len(desc) > 80:
                desc = desc[:77] + "..."
            print(f"  {name} {var.dims} {var.shape} [{units}]\n      {desc}")

        print("\nGlobal attributes:")
        for key, val in ds.attrs.items():
            val = str(val)
            print(f"  {key}: {val[:100] + '...' if len(val) > 100 else val}")

    return ds


# Differential proton channel names, in the order given in the file's
# DiffProtonLowerEnergy / UpperEnergy descriptions
PROTON_CHANNELS = ["P1", "P2A", "P2B", "P3", "P4", "P5", "P6",
                   "P7", "P8A", "P8B", "P8C", "P9", "P10"]


def get_sgps_proton_flux(fp):
    """
    Load AvgDiffProtonFlux for sensor unit 0 (-X, looks westward when the
    spacecraft is upright) from a GOES SEISS/SGPS L2 5-minute NetCDF file.

    Returns a DataFrame indexed by 'time' (start of each 5-minute average),
    with one column per energy channel (P1 ... P10), in
    protons/(cm^2 sr keV s). Channel energy bounds (keV) are stored in
    df.attrs["energy_range_keV"].
    """
    # Differential proton channel names, in the order given in the file's
    # DiffProtonLowerEnergy / UpperEnergy descriptions
    PROTON_CHANNELS = ["P1", "P2A", "P2B", "P3", "P4", "P5", "P6",
                    "P7", "P8A", "P8B", "P8C", "P9", "P10"]

    with xr.open_dataset(fp) as ds:
        flux = ds["AvgDiffProtonFlux"].isel(sensor_units=0).load()
        lower = ds["DiffProtonLowerEnergy"].isel(sensor_units=0).values
        upper = ds["DiffProtonUpperEnergy"].isel(sensor_units=0).values

    df = pd.DataFrame(
        flux.values,
        index=pd.Index(flux["time"].values, name="time"),
        columns=PROTON_CHANNELS,
    )
    df.attrs["energy_range_keV"] = {
        ch: (float(lo), float(hi))
        for ch, lo, hi in zip(PROTON_CHANNELS, lower, upper)
    }
    df.attrs["units"] = "protons/(cm^2 sr keV s)"
    return df


def get_sgps_proton_flux_range(start_time: datetime, end_time: datetime,
                               path=f'{goes_path}'+'18/sgps/'):
    """
    Load and stitch together daily GOES SEISS/SGPS L2 5-minute files
    covering start_time to end_time.
 
    Parameters
    ----------
    start_time, end_time : datetime.datetime
        Start and end of the range, e.g. datetime(2025, 11, 10, 12, 0).
        Both ends are inclusive. Timezone-aware datetimes are converted to
        UTC (the data are in UT); naive datetimes are assumed to be UTC.
    path : str or Path
        Folder containing the daily .nc files, named like
        sci_sgps-l2-avg5m_g18_d20251110_v3-0-2.nc
    satellite : str
        Satellite tag in the filename (default "g18").
 
    Returns
    -------
    pandas.DataFrame
        Indexed by 'time', trimmed to [start_time, end_time], with the same
        columns as load_sgps_proton_flux. Returns None if no files were found.
    """
    def _to_naive_utc(t):
        t = pd.Timestamp(t)
        return t.tz_convert("UTC").tz_localize(None) if t.tzinfo else t
 
    start_timestamp = _to_naive_utc(start_time)
    end_timestamp = _to_naive_utc(end_time)
 
    frames = []
    attrs = None
    day = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while day < end:
        date_str = f"{day.year}{day.month:02}{day.day:02}"
        fn = sorted(glob.glob(
            os.path.join(path, f"sci_sgps-l2-avg5m_g18_d{date_str}_v*.nc")
        ))
        if not fn:
            print(f"WARNING: no file found for {date_str}")
        else:
            try:
                # if several versions exist for a day, use the latest (last sorted)
                _df = get_sgps_proton_flux(fn[-1])
                if attrs is None:
                    attrs = _df.attrs
                frames.append(_df)
            except Exception as e:
                print(f"ERROR loading {date_str}: {e}")
        day += timedelta(days=1)
 
    if not frames:
        return None
 
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.loc[start_timestamp:end_timestamp]
    df.attrs = attrs
    df = df.reset_index()
    return df

