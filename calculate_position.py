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
    path = '2020-06-25-00-34-us-ca-mtv-sb-101/pixel4xl'
    gnss = pd.read_csv('data/%s/device_gnss.csv' % path, dtype={'SignalType': str})
    gt = pd.read_csv('data/%s/ground_truth.csv' % path, dtype={'SignalType': str})
    rtk = pd.read_csv('GSDC_2023/data/locations_train_06_03.csv', dtype={'SignalType': str})
    rtk_train = rtk[rtk['tripId'] == path]

    # Add standard Frequency column
    frequency_median = gnss.groupby('SignalType')['CarrierFrequencyHz'].median()
    gnss = gnss.merge(frequency_median, how='left', on='SignalType', suffixes=('', 'Ref'))
    carrier_error = abs(gnss['CarrierFrequencyHz'] - gnss['CarrierFrequencyHzRef'])
    gnss['CarrierErrorHz'] = carrier_error
    utcTimeMillis = gnss['utcTimeMillis'].unique()
    nepoch = len(utcTimeMillis)
    gt_len = len(gt)
    x0 = np.zeros(4)  # [x,y,z,tGPSL1]
    v0 = np.zeros(4)  # [vx,vy,vz,dtGPSL1]
    x_wls = np.full([nepoch, 3], np.nan)  # For saving position
    v_wls = np.full([nepoch, 3], np.nan)  # For saving velocity
    cov_x = np.full([nepoch, 3, 3], np.nan) # For saving position covariance
    cov_v = np.full([nepoch, 3, 3], np.nan) # For saving velocity covariance  
    score_wls = []
    score_bl = [] 
    score_rtk = []

    for i, (t_utc, df) in enumerate(tqdm(gnss.groupby('utcTimeMillis'), total=nepoch)):
        if (i ==0) and (nepoch != gt_len) : continue  #First position is not in ground truth in some phone
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
        if nepoch == gt_len :
            llh_rtk = rtk_train[['LatitudeDegrees', 'LongitudeDegrees']].to_numpy()[i]
        else: 
            llh_rtk = rtk_train[['LatitudeDegrees', 'LongitudeDegrees']].to_numpy()[i-1]
        # Baseline
        bl = gnss[gnss['utcTimeMillis'] == utcTimeMillis[i]]
        x_bl = bl[['WlsPositionXEcefMeters', 'WlsPositionYEcefMeters', 'WlsPositionZEcefMeters']].mean().to_numpy()
        llh_bl = np.array(pm.ecef2geodetic(x_bl[ 0], x_bl[1], x_bl[2])).T
        llh_wls = np.array(pm.ecef2geodetic(x_wls[i,0], x_wls[i,1], x_wls[i,2])).T
        gt_i = gt[gt['UnixTimeMillis'] == utcTimeMillis[i]]
        llh_gt = gt_i[['LatitudeDegrees', 'LongitudeDegrees']].to_numpy()[0]
        if not np.isnan(llh_wls).any():
            score_wls.append(vincenty_distance(llh_wls, llh_gt))
        score_bl.append(vincenty_distance(llh_bl, llh_gt))
        score_rtk.append(vincenty_distance(llh_gt, llh_rtk))

    #Remove 1st position to compare to ground truth
    if nepoch != gt_len :
        x_wls = x_wls[1:,:]
        v_wls = v_wls[1:,:]
        cov_x = cov_x[1:,:,:]
        cov_v = cov_v[1:,:,:]

    # Exclude velocity outliers
    x_wls, v_wls, cov_x, cov_v = exclude_interpolate_outlier(x_wls, v_wls, cov_x, cov_v)

    #Kalman filter all epoch
    x_kf, _ = kalman_smoothing(x_wls, v_wls, cov_x, cov_v)
    llh_kf = np.array(pm.ecef2geodetic(x_kf[:,0], x_kf[:,1], x_kf[:,2])).T
    llh_wls = np.array(pm.ecef2geodetic(x_wls[:,0], x_wls[:,1], x_wls[:,2])).T
    llh_gt = gt[['LatitudeDegrees', 'LongitudeDegrees']].to_numpy()
    score_all_wls = np.mean([np.quantile(score_wls, 0.50), np.quantile(score_wls, 0.95)])
    score_all_bl = np.mean([np.quantile(score_bl, 0.50), np.quantile(score_bl, 0.95)])
    #score_all_wls = calc_score(llh_wls, llh_gt)
    #score_all_bl = calc_score(llh_bl, llh_gt)
    score_kf = []
    for i in range(len(llh_gt)):
        score_kf_i = vincenty_distance(llh_gt[i], llh_kf[i])
        score_kf.append(score_kf_i)
    score_all_kf = np.mean([np.quantile(score_kf, 0.50), np.quantile(score_kf, 0.95)])
    score_all_rtk = np.mean([np.quantile(score_rtk, 0.50), np.quantile(score_rtk, 0.95)])

    print("KF score: ", score_all_kf)
    print("Baseline score: ", score_all_bl)
    print("WLS score: ", score_all_wls)
    print("RTK score: ", score_all_rtk)


    # Plot distance error
    plt.figure()
    plt.title('Distance error')
    plt.ylabel('Distance error [m]')
    plt.plot(score_bl, label=f'Baseline, Score: {score_all_bl:.4f} m')
    plt.plot(score_wls, label=f'WLS, Score: {score_all_wls:.4f} m')
    plt.plot(score_kf, label=f'KF, Score: {score_all_kf:.4f} m')
    plt.plot(score_rtk, label=f'RTK, Score: {score_all_rtk:.4f} m')
    plt.legend()
    plt.grid()
    plt.ylim([0, 30])
    plt.show()

if __name__ == "__main__":
    main()