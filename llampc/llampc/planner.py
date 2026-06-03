"""	Generate reference for MPC using a trajectory generator.
"""

__author__ = 'Dvij Kalaria'
__email__ = 'dkalaria@andrew.cmu.edu'


import numpy as np
from llampc.utils import Spline2D


# def get_reference_trajectory_segment(x0, v0, track, N, Ts, projidx, scale=1., wrap = True, curr_mu = 1.):
#     """	generate a reference trajectory of size 2x(N+1)
#         first column is x0

#         x0 		: current position (2x1)
#         v0 		: current velocity (scaler)
#         track	: see llampc.tracks, example Rectangular
#         N 		: no of reference points, same as horizon in MPC
#         Ts 		: sampling time in MPC
#         projidx : hack required when raceline is longer than 1 lap

#     """
#     # project x0 onto raceline
#     raceline = track.raceline
#     xy, idx = track.project_fast(x=x0[0], y=x0[1], raceline=raceline[:,projidx:projidx+10])
#     projidx = idx+projidx

#     # start ahead of the current position
#     start = track.raceline[:,:projidx+2]

#     xref = np.zeros([6,N+1])
#     xref[:2,0] = x0

#     # use splines to sample points based on max acceleration
#     # note that you will need to pass in velocity profiles matching
#     # the points v to the track on creation, but not for mus (i.e. vs argument)
#     # when loading the raceline
#     dist0 = np.sum(np.linalg.norm(np.diff(start), 2, axis=0))
#     dist = dist0
#     v = max(v0,.2)
#     # xref[3,0] = v
#     # vr = 0.
#     eps=1e-4
#     for idh in range(1,N+1):
#         dist += scale*v*Ts


#         max_s = track.spline.s[-1] - eps
        
#         if dist >= max_s:
#             if wrap:
#                 # Wrap around to the beginning of the track
#                 dist = dist % max_s
#             else:
#                 # Clamp to the end and fill the rest of the horizon
#                 final_pos = track.spline.calc_position(max_s)
#                 xref[:2, idh:] = final_pos.reshape(2, 1)
#                 break
#         dist = dist % track.spline.s[-1] #WRAPPING


#         # print(dist)
#         xref[:2,idh] = track.spline.calc_position(dist)
#         # print(curr_mu,v)
#         v = track.spline_v.calc(dist)
#         # xref[3,idh] = v
#         # print(v)
#         # if curr_mu < track.mus[0] :
#         # 	v = track.spline_v[0].calc(dist)
#         # 	i=0
#         # elif curr_mu > track.mus[-1] :
#         # 	v = track.spline_v[-1].calc(dist)
#         # 	i=len(track.mus)-1
#         # else :
#         # 	i = 0
#         # 	for i in range(len(track.mus)) :
#         # 		if track.mus[i] >= curr_mu :
#         # 			break
#         # 	# print(i)
#         # 	vb = track.spline_v[i-1].calc(dist)
#         # 	va = track.spline_v[i].calc(dist)
#         # 	v = vb*(track.mus[i]-curr_mu)/(track.mus[i]-track.mus[i-1]) + va*(curr_mu-track.mus[i-1])/(track.mus[i]-track.mus[i-1])
#         # if idh==1 :
#         # 	vr = v*scale
#         # 	# print(v,va,vb)

#     return xref, projidx#, vr

def get_reference_trajectory_segment(x0, v0, track, N, Ts, projidx, scale=1., wrap=True, skip=2, max_accel = 9.51):
    raceline = track.raceline
    num_pts = raceline.shape[1]
    
    search_indices = np.arange(projidx, projidx + 50) % num_pts
    search_window = raceline[:, search_indices]

    if np.allclose(search_window[:, 0], search_window[:, 1]):
        search_indices = (search_indices + 1) % num_pts
        search_window = raceline[:, search_indices]
    
    xy, idx_rel = track.project_fast(x=x0[0], y=x0[1], raceline=search_window)
    new_projidx = search_indices[idx_rel]

    dist_start = track.spline.s[new_projidx]
    max_s = track.spline.s[-1]
    
    dist = dist_start
    v = max(v0, 0.2)

    # Generate N+1+skip points, then slice off the first `skip`
    total = N + 1 + skip
    xref_full = np.zeros([6, total])
    xref_full[:2, 0] = x0

    for idh in range(1, total):
        dist += scale * v * Ts
        
        if wrap:
            s_sample = dist % max_s
        else:
            if dist >= max_s:
                xref_full[:2, idh:] = track.spline.calc_position(max_s - 1e-4).reshape(2, 1)
                xref_full[2, idh:] = track.spline.calc_yaw(max_s - 1e-4)
                xref_full[3, idh:] = 0 
                break
            s_sample = dist

        xref_full[:2, idh] = track.spline.calc_position(s_sample)
        xref_full[2, idh] = track.spline.calc_yaw(s_sample)
        v_next = track.spline_v.calc(s_sample)
        
        v = min(v_next, max_accel * Ts + v) # velocity aware planner
        
        xref_full[3, idh] = v 

    xref = xref_full[:, skip:skip + N + 1]

    return xref, new_projidx

def get_lookahead_point(x0, track, projidx, lookahead_dist):
    """
    Walk forward along the raceline from projidx until a point is
    found at euclidean distance >= lookahead_dist from x0.
    """
    raceline = track.raceline
    num_pts  = raceline.shape[1]

    search_indices = np.arange(projidx, projidx + 50) % num_pts
    search_window  = raceline[:, search_indices]

    # Skip ahead if the first two points in the window are identical 
    # to avoid zero-division in projection
    if np.allclose(search_window[:, 0], search_window[:, 1]):
        search_indices = (search_indices + 1) % num_pts
        search_window = raceline[:, search_indices]

    _, idx_rel     = track.project_fast(x=x0[0], y=x0[1], raceline=search_window)
    new_projidx    = search_indices[idx_rel]

    for i in range(num_pts):
        idx = (new_projidx + i) % num_pts
        dp = raceline[:2, idx] - x0[:2]
        if np.linalg.norm(dp) >= lookahead_dist:
            s = track.spline.s[idx] % track.spline.s[-1]  # wrap to valid range
            xy  = track.spline.calc_position(s)
            yaw = track.spline.calc_yaw(s)
            speed = track.spline_v.calc(s)
            return np.array([xy[0], xy[1], yaw, speed, 0.0, 0.0]), new_projidx

    s = track.spline.s[idx] % track.spline.s[-1]  # wrap to valid range
    xy  = track.spline.calc_position(s)
    yaw = track.spline.calc_yaw(s)
    speed = track.spline_v.calc(s)
    return np.array([xy[0], xy[1], yaw, speed, 0.0, 0.0]), new_projidx
