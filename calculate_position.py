from outlier import exclude_interpolate_outlier
from calculate_score import calc_score
from calculate_wls import *
from kalman_filter import Kalman_smoothing
from tqdm.auto import tqdm
from time import time 
import numpy as np
import pandas as pd
import scipy.optimize
import pymap3d as pm
import glob as gl
import pymap3d.vincenty as pmv

# Constants
CLIGHT = 299_792_458   # speed of light (m/s)
RE_WGS84 = 6_378_137   # earth semimajor axis (WGS84) (m)
OMGE = 7.2921151467E-5  # earth angular velocity (IS-GPS) (rad/s)



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

    return df[idx]


def main1():
    path = '2020-06-25-00-34-us-ca-mtv-sb-101/pixel4xl'
    gnss_all = pd.read_csv('%s/device_gnss.csv' % path, dtype={'SignalType': str})
    gt = pd.read_csv('%s/ground_truth.csv' % path, dtype={'SignalType': str})

    #For app use this
    #current_time_seconds = time()

    # Convert to milliseconds
    #utcTimeMillis = int(current_time_seconds * 1000)
    start_time = time()
    gnss = gnss_all[gnss_all['utcTimeMillis'] == 1593045255447]
    # Add standard Frequency column
    frequency_median = gnss.groupby('SignalType')['CarrierFrequencyHz'].median()
    gnss = gnss.merge(frequency_median, how='left', on='SignalType', suffixes=('', 'Ref'))
    carrier_error = abs(gnss['CarrierFrequencyHz'] - gnss['CarrierFrequencyHzRef'])
    gnss['CarrierErrorHz'] = carrier_error
    gnss = satellite_selection(gnss)


    x0 = np.zeros(4)  # [x,y,z,tGPSL1]
    v0 = np.zeros(4)  # [vx,vy,vz,dtGPSL1]
    x_wls = np.full(3, np.nan)  # For saving position
    v_wls = np.full(3, np.nan)  # For saving velocity
    cov_x = np.full([3, 3], np.nan) # For saving position covariance
    cov_v = np.full([3, 3], np.nan) # For saving velocity covariance
    # Corrected pseudorange/pseudorange rate
    pr = (gnss['RawPseudorangeMeters'] + gnss['SvClockBiasMeters'] - gnss['IsrbMeters'] -
            gnss['IonosphericDelayMeters'] - gnss['TroposphericDelayMeters']).to_numpy()
    prr = (gnss['PseudorangeRateMetersPerSecond'] +
            gnss['SvClockDriftMetersPerSecond']).to_numpy()

    # Satellite position/velocity
    xsat_pr = gnss[['SvPositionXEcefMeters', 'SvPositionYEcefMeters',
                        'SvPositionZEcefMeters']].to_numpy()
    xsat_prr = gnss[['SvPositionXEcefMeters', 'SvPositionYEcefMeters',
                        'SvPositionZEcefMeters']].to_numpy()
    vsat = gnss[['SvVelocityXEcefMetersPerSecond', 'SvVelocityYEcefMetersPerSecond',
                    'SvVelocityZEcefMetersPerSecond']].to_numpy()

    # Weight matrix for peseudorange/pseudorange rate
    Wx = np.diag(1 / gnss['RawPseudorangeUncertaintyMeters'].to_numpy())
    Wv = np.diag(1 / gnss['PseudorangeRateUncertaintyMetersPerSecond'].to_numpy())

    # Robust WLS requires accurate initial values for convergence,
    # so perform normal WLS for the first time
    if len(gnss) >= 4:
        # Normal WLS
        if np.all(x0 == 0):
            opt = scipy.optimize.least_squares(
                f_wls, x0, jac_pr_residuals, args=(xsat_pr, pr, Wx))
            x0 = opt.x 
        # Robust WLS for position estimation
        opt = scipy.optimize.least_squares(
                f_wls, x0,  jac_pr_residuals, args=(xsat_pr, pr, Wx), loss='soft_l1')
        if opt.status < 1 or opt.status == 2:
            print(f'position lsq status = {opt.status}')
        else:
            # Covariance estimation
            cov = np.linalg.inv(opt.jac.T @ Wx @ opt.jac)
            cov_x[:, :] = cov[:3, :3]
            x_wls[:] = opt.x[:3]
            x0 = opt.x

    # Velocity estimation
    if len(gnss) >= 4:
        if np.all(v0 == 0): # Normal WLS
            opt = scipy.optimize.least_squares(
                prr_residuals, v0, jac_prr_residuals, args=(vsat, prr, x0, xsat_prr, Wv))
            v0 = opt.x
        # Robust WLS for velocity estimation
        opt = scipy.optimize.least_squares(
            prr_residuals, v0, jac_prr_residuals, args=(vsat, prr, x0, xsat_prr, Wv), loss='soft_l1')
        if opt.status < 1:
            print(f'velocity lsq status = {opt.status}')
        else:
            # Covariance estimation
            cov = np.linalg.inv(opt.jac.T @ Wv @ opt.jac)
            cov_v[:, :] = cov[:3, :3]
            v_wls[:] = opt.x[:3]
            v0 = opt.x

    # Baseline
    x_bl = gnss.groupby('TimeNanos')[
        ['WlsPositionXEcefMeters', 'WlsPositionYEcefMeters', 'WlsPositionZEcefMeters']].mean().to_numpy()
    llh_bl = np.array(pm.ecef2geodetic(x_bl[:, 0], x_bl[:, 1], x_bl[:, 2])).T
    llh_wls = np.array(pm.ecef2geodetic(x_wls[0], x_wls[1], x_wls[2])).T
    gt = gt[gt['UnixTimeMillis'] == 1593045255447]
    print(gt)
    llh_gt = gt[['LatitudeDegrees', 'LongitudeDegrees']].to_numpy()[0]
    print(llh_gt)
    exec_time = time() - start_time
    score_bl = calc_score(llh_bl, llh_gt)
    print("My score: ", calc_score(llh_gt, llh_wls))
    print("Baseline score: ", calc_score(llh_gt, llh_bl))
    print("Execute time: ", exec_time*1000, "ms")


def main():
    path = '2020-06-25-00-34-us-ca-mtv-sb-101/pixel4xl'
    gnss = pd.read_csv('%s/device_gnss.csv' % path, dtype={'SignalType': str})
    gt = pd.read_csv('%s/ground_truth.csv' % path, dtype={'SignalType': str})
    rtk = pd.read_csv('GSDC_2023/data/locations_train_05_27.csv', dtype={'SignalType': str})
    rtk_train = rtk[rtk['tripId'] == path]

    #For app use this
    #current_time_seconds = time()

    # Convert to milliseconds
    #utcTimeMillis = int(current_time_seconds * 1000)
    #gnss = gnss_all[gnss_all['utcTimeMillis'] == 1593045253440]
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
    score = []    

    for i, (t_utc, df) in enumerate(tqdm(gnss.groupby('utcTimeMillis'), total=nepoch)):
        if i ==0 : continue
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
        #RTK
        llh_rtk = rtk_train[['LatitudeDegrees', 'LongitudeDegrees', 'AltitudeMeters']].to_numpy()[i-1]
        x_rtk = np.array(pm.geodetic2ecef(llh_rtk[0], llh_rtk[1], llh_rtk[2])).T
        x_wls[i,:] = x_rtk
        # Baseline
        bl = gnss[gnss['utcTimeMillis'] == utcTimeMillis[i]]
        x_bl = bl[['WlsPositionXEcefMeters', 'WlsPositionYEcefMeters', 'WlsPositionZEcefMeters']].mean().to_numpy()
        llh_bl = np.array(pm.ecef2geodetic(x_bl[ 0], x_bl[1], x_bl[2])).T
        llh_wls = np.array(pm.ecef2geodetic(x_wls[i,0], x_wls[i,1], x_wls[i,2])).T
        gt_i = gt[gt['UnixTimeMillis'] == utcTimeMillis[i]]
        llh_gt = gt_i[['LatitudeDegrees', 'LongitudeDegrees']].to_numpy()[0]
        score_rtk = calc_score(llh_gt, llh_rtk)
        score.append(score_rtk)

    x_wls = x_wls[1:,:]
    v_wls = v_wls[1:,:]
    cov_x = cov_x[1:,:,:]
    cov_v = cov_v[1:,:,:]
    print(x_wls)
    n, dim_x = x_wls.shape
    print("n :", n)
    print("dim_x: " ,dim_x)
    x_kf, _,_ = Kalman_smoothing(x_wls, v_wls, cov_x, cov_v)
    llh_kf = np.array(pm.ecef2geodetic(x_kf[:,0], x_kf[:,1], x_kf[:,2])).T
    llh_wls = np.array(pm.ecef2geodetic(x_wls[:,0], x_wls[:,1], x_wls[:,2])).T
    llh_gt = gt[['LatitudeDegrees', 'LongitudeDegrees']].to_numpy()
    score_wls = np.mean([np.quantile(score, 0.50), np.quantile(score, 0.95)])
    score_kf = []
    for i in range(len(llh_gt)):
        score_kf_i = calc_score(llh_gt[i], llh_kf[i])
        score_kf.append(score_kf_i)
    score_kf = np.mean([np.quantile(score_kf, 0.50), np.quantile(score_kf, 0.95)])
    print("RTK score: ", score_kf)
    print("Total score: ", score_wls)
if __name__ == "__main__":
    main()