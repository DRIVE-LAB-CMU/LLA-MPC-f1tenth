"""	Generate reference for MPC using a trajectory generator.
"""

__author__ = 'Dvij Kalaria'
__email__ = 'dkalaria@andrew.cmu.edu'


import numpy as np
from llampc.utils import Spline2D


def get_reference_trajectory_segment(x0, v0, track, N, Ts, projidx, scale=1., curr_mu = 1.):
	"""	generate a reference trajectory of size 2x(N+1)
		first column is x0

		x0 		: current position (2x1)
		v0 		: current velocity (scaler)
		track	: see llampc.tracks, example Rectangular
		N 		: no of reference points, same as horizon in MPC
		Ts 		: sampling time in MPC
		projidx : hack required when raceline is longer than 1 lap

	"""
	# project x0 onto raceline
	raceline = track.raceline
	xy, idx = track.project_fast(x=x0[0], y=x0[1], raceline=raceline[:,projidx:projidx+10])
	projidx = idx+projidx

	# start ahead of the current position
	start = track.raceline[:,:projidx+2]

	xref = np.zeros([2,N+1])
	xref[:2,0] = x0

	# use splines to sample points based on max acceleration
    # note that you will need to pass in velocity profiles matching
    # the points v to the track on creation, but not for mus (i.e. vs argument)
    # when loading the raceline
	dist0 = np.sum(np.linalg.norm(np.diff(start), 2, axis=0))
	dist = dist0
	v = max(v0,.3)
	# vr = 0.
	eps=1e-4
	for idh in range(1,N+1):
		dist += scale*v*Ts


		if dist >= track.spline.s[-1] - eps: # NOT WRAPPING
			for i in range (0, N+1-idh, -1):
				dist = track.spline.s[-1] - eps * (N+1-idh-i)
				xref[:2, idh+i:] = track.spline.calc_position(dist)
			break
		dist = dist % track.spline.s[-1] #WRAPPING


		# print(dist)
		xref[:2,idh] = track.spline.calc_position(dist)
		# print(curr_mu,v)
		v = track.spline_v.calc(dist)
		# print(v)
		# if curr_mu < track.mus[0] :
		# 	v = track.spline_v[0].calc(dist)
		# 	i=0
		# elif curr_mu > track.mus[-1] :
		# 	v = track.spline_v[-1].calc(dist)
		# 	i=len(track.mus)-1
		# else :
		# 	i = 0
		# 	for i in range(len(track.mus)) :
		# 		if track.mus[i] >= curr_mu :
		# 			break
		# 	# print(i)
		# 	vb = track.spline_v[i-1].calc(dist)
		# 	va = track.spline_v[i].calc(dist)
		# 	v = vb*(track.mus[i]-curr_mu)/(track.mus[i]-track.mus[i-1]) + va*(curr_mu-track.mus[i-1])/(track.mus[i]-track.mus[i-1])
		# if idh==1 :
		# 	vr = v*scale
		# 	# print(v,va,vb)

	return xref, projidx#, vr
