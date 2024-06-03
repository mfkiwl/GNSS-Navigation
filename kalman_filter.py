import numpy as np
from scipy.spatial import distance

def kalman_filter(x_wls, v_wls, cov_x, cov_v):
    # Parameters
    sigma_mahalanobis = 30.0  # Mahalanobis distance for rejecting innovation

    n, dim_x = x_wls.shape
    F = np.eye(3)  # Transition matrix
    H = np.eye(3)  # Measurement function

    # Initial state and covariance
    x = x_wls[0, :3].T  # State
    P = 5.0**2 * np.eye(3)  # State covariance
    I = np.eye(3)

    x_kf = np.zeros([n, dim_x])
    P_kf = np.zeros([n, dim_x, dim_x])

    #Initial for first epoch
    x_kf[0] = x.T
    P_kf[0] = P

    # Kalman filtering
    for i, (u, z) in enumerate(zip(v_wls, x_wls)):
        if i == 0:
            continue

        # Prediction step
        Q = cov_v[i] # Estimated WLS velocity covariance
        x = F @ x + u.T
        P = (F @ P) @ F.T + Q

        # Check outliers for observation
        d = distance.mahalanobis(z, H @ x, np.linalg.inv(P))

        # Update step
        if d < sigma_mahalanobis:
            R = cov_x[i] # Estimated WLS position covariance
            y = z.T - H @ x
            S = (H @ P) @ H.T + R
            K = (P @ H.T) @ np.linalg.inv(S)
            x = x + K @ y
            P = (I - (K @ H)) @ P
        else:
            # If observation update is not available, increase covariance
            P += 10**2*Q

        x_kf[i] = x.T
        P_kf[i] = P

    return x_kf, P_kf



def kalman_smoothing(x_wls, v_wls, cov_x, cov_v):

    #Get the velocity between 2 epoch (median)
    v = np.vstack([np.zeros([1, 3]), (v_wls[:-1] + v_wls[1:])/2])
    x_f, P_f = kalman_filter(x_wls, v, cov_x, cov_v)    
    return x_f, P_f


def kalman_smoothing_origin(x_wls, v_wls, cov_x, cov_v):
    n, _ = x_wls.shape

    # Forward
    v = np.vstack([np.zeros([1, 3]), (v_wls[:-1] + v_wls[1:])/2])
    x_f, P_f = kalman_filter(x_wls, v, cov_x, cov_v)    

    # Backward
    v = -np.flipud(v_wls)
    v = np.vstack([np.zeros([1, 3]), (v[:-1] + v[1:])/2])
    cov_xf = np.flip(cov_x, axis=0)
    cov_vf = np.flip(cov_v, axis=0)
    x_b, P_b = kalman_filter(np.flipud(x_wls), v, cov_xf, cov_vf)

    # Smoothing
    x_fb = np.zeros_like(x_f)
    P_fb = np.zeros_like(P_f)
    for (f, b) in zip(range(n), range(n-1, -1, -1)):
        P_fi = np.linalg.inv(P_f[f])
        P_bi = np.linalg.inv(P_b[b])

        P_fb[f] = np.linalg.inv(P_fi + P_bi)
        x_fb[f] = P_fb[f] @ (P_fi @ x_f[f] + P_bi @ x_b[b])

    return x_f, x_f, np.flipud(x_b)