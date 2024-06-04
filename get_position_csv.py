from outlier import exclude_interpolate_outlier
from calculate_score import calc_score, vincenty_distance
from calculate_wls import *
from kalman_filter import kalman_smoothing, kalman_smoothing_origin
from tqdm.auto import tqdm
from time import time 
import numpy as np
import pandas as pd
import scipy.optimize
import pymap3d as pm
import glob as gl
import pymap3d.vincenty as pmv
import matplotlib.pyplot as plt

# Constants
CLIGHT = 299_792_458   # speed of light (m/s)
RE_WGS84 = 6_378_137   # earth semimajor axis (WGS84) (m)
OMGE = 7.2921151467E-5  # earth angular velocity (IS-GPS) (rad/s)

path = '2021-07-19-20-49-us-ca-mtv-a/pixel5'

def satellite_selection(df):
    """
    Args:
        df : DataFrame from device_gnss.csv
        column : Column name
    Returns:
        df: DataFrame with eliminated satellite signals
    """
    idx = df['RawPseudorangeMeters'].notnull()
    idx &= df['CarrierErrorHz'] < 2.0e6  # carrier frequency error (Hz)
    idx &= df['Cn0DbHz'] > 18.0  # C/N0 (dB-Hz)
    idx &= df['MultipathIndicator'] == 0 # Multipath flag
    idx &= df['ReceivedSvTimeUncertaintyNanos'] < 500

    return df[idx]


def main():
    gnss = pd.read_csv('data/%s/device_gnss.csv' % path, dtype={'SignalType': str})

    # Add standard Frequency column
    frequency_median = gnss.groupby('SignalType')['CarrierFrequencyHz'].median()
    gnss = gnss.merge(frequency_median, how='left', on='SignalType', suffixes=('', 'Ref'))
    carrier_error = abs(gnss['CarrierFrequencyHz'] - gnss['CarrierFrequencyHzRef'])
    gnss['CarrierErrorHz'] = carrier_error
    utcTimeMillis = gnss['utcTimeMillis'].unique()
    nepoch = len(utcTimeMillis)
    x0 = np.zeros(4)  # [x,y,z,tGPSL1]
    v0 = np.zeros(4)  # [vx,vy,vz,dtGPSL1]
    x_wls = np.full([nepoch, 3], np.nan)  # For saving position
    v_wls = np.full([nepoch, 3], np.nan)  # For saving velocity
    cov_x = np.full([nepoch, 3, 3], np.nan) # For saving position covariance
    cov_v = np.full([nepoch, 3, 3], np.nan) # For saving velocity covariance  


    for i, (t_utc, df) in enumerate(tqdm(gnss.groupby('utcTimeMillis'), total=nepoch)):
        df_pr = satellite_selection(df)
        df_prr = satellite_selection(df)

        # Corrected pseudorange/pseudorange rate
        pr = (df_pr['RawPseudorangeMeters'] + df_pr['SvClockBiasMeters'] - df_pr['IsrbMeters'] -
              df_pr['IonosphericDelayMeters'] - df_pr['TroposphericDelayMeters']).to_numpy()
        prr = (df_prr['PseudorangeRateMetersPerSecond'] +
               df_prr['SvClockDriftMetersPerSecond']).to_numpy()

        # Satellite position/velocity
        xsat_pr = df_pr[['SvPositionXEcefMeters', 'SvPositionYEcefMeters',
                         'SvPositionZEcefMeters']].to_numpy()
        xsat_prr = df_prr[['SvPositionXEcefMeters', 'SvPositionYEcefMeters',
                           'SvPositionZEcefMeters']].to_numpy()
        vsat = df_prr[['SvVelocityXEcefMetersPerSecond', 'SvVelocityYEcefMetersPerSecond',
                       'SvVelocityZEcefMetersPerSecond']].to_numpy()

        # Weight matrix for peseudorange/pseudorange rate
        Wx = np.diag(1 / df_pr['RawPseudorangeUncertaintyMeters'].to_numpy())
        Wv = np.diag(1 / df_prr['PseudorangeRateUncertaintyMetersPerSecond'].to_numpy())

        # Robust WLS requires accurate initial values for convergence,
        # so perform normal WLS for the first time
        if len(df_pr) >= 4:
            # Normal WLS
            if np.all(x0 == 0):
                opt = scipy.optimize.least_squares(
                    f_wls, x0, jac_pr_residuals, args=(xsat_pr, pr, Wx))
                x0 = opt.x 
            # Robust WLS for position estimation
            opt = scipy.optimize.least_squares(
                 f_wls, x0, jac_pr_residuals, args=(xsat_pr, pr, Wx), loss='soft_l1')
            if opt.status < 1 or opt.status == 2:
                print(f'i = {i} position lsq status = {opt.status}')
            else:
                # Covariance estimation
                cov = np.linalg.inv(opt.jac.T @ Wx @ opt.jac)
                cov_x[i, :, :] = cov[:3, :3]
                x_wls[i, :] = opt.x[:3]
                x0 = opt.x
                 
        # Velocity estimation
        if len(df_prr) >= 4:
            if np.all(v0 == 0): # Normal WLS
                opt = scipy.optimize.least_squares(
                    prr_residuals, v0, jac_prr_residuals, args=(vsat, prr, x0, xsat_prr, Wv))
                v0 = opt.x
            # Robust WLS for velocity estimation
            opt = scipy.optimize.least_squares(
                prr_residuals, v0, jac_prr_residuals, args=(vsat, prr, x0, xsat_prr, Wv), loss='soft_l1')
            if opt.status < 1:
                print(f'i = {i} velocity lsq status = {opt.status}')
            else:
                # Covariance estimation
                cov = np.linalg.inv(opt.jac.T @ Wv @ opt.jac)
                cov_v[i, :, :] = cov[:3, :3]
                v_wls[i, :] = opt.x[:3]
                v0 = opt.x

    # Exclude velocity outliers
    x_wls, v_wls, cov_x, cov_v = exclude_interpolate_outlier(x_wls, v_wls, cov_x, cov_v)
    
    #Kalman filter all epoch
    x_kf, _ = kalman_smoothing(x_wls, v_wls, cov_x, cov_v)
    llh_kf = np.array(pm.ecef2geodetic(x_kf[:,0], x_kf[:,1], x_kf[:,2])).T

    position_array = [(time, lat, lon) for time, (lat, lon, alt) in zip(utcTimeMillis, llh_kf)]
    df = pd.DataFrame(position_array, columns=["Time", "Latitude", "Longitude"])
    df.to_csv("position_data.csv", index=False)
if __name__ == "__main__":
    main()