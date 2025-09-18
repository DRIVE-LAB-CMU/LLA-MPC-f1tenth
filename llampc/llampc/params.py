""" Params for F1/10 1:10 scale car from University of Pennsylvania
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'
import numpy as np
def F110_sim():
	return {
		'mu': 1.0489, 
		'C_Sf': 4.718, 
		'C_Sr': 5.4562, 
		'lf': 0.15875, 
		'lr': 0.17145, 
		'h': 0.074, 
		'm': 3.74, 
		'I': 0.04712, 
		's_min': -0.4189, 
		's_max': 0.4189, 
		'sv_min': -3.2, 
		'sv_max': 3.2, 
		'v_switch': 7.319,
		'a_max': 9.51,
		'v_min':-5.0, 
		'v_max': 20.0,
		'width': 0.31,
		'length': 0.58
	}

def F110():
	lf = 0.15875			# front tyres from center of gravity [m]
	lr = 0.17145			# read tyres from center of gravity [m]
	mass = 3.74 			# vehicle mass [kg]
	Iz = 0.04712 			# moment of inertia [kgm^2]
	Cf = 2.3 				# front cornering stiffness
	Cr = 2.3 				# rear cornering stiffness
	Csf = 4.718
	Csr = 5.4562
	hcog = 0.074
	mu = 0.523
	min_v = 0.
	max_v = 20.
	switch_v = 1.
	max_acc = 9.51 			# max acceleration [m/s^2]
	min_acc = -13.26 		# max deceleration [m/s^2]
	max_steer = 0.4189 		# max steering angle [rad]
	min_steer = -0.4189 	# min steering angle [rad]
	max_steer_vel = 3.2 	# max steering velocity [rad/s]

	max_inputs = [max_acc, max_steer]
	min_inputs = [min_acc, min_steer]

	max_rates = [None, max_steer_vel]
	min_rates = [None, -max_steer_vel]	

	params = {
		'lf': lf,
		'lr': lr,
		'mass': mass,
		'Iz': Iz,
		'Cf': Cf,
		'Cr': Cr,
		'Csf': Csf,
		'Csr': Csr,
		'hcog': hcog,
		'mu': mu,
		'min_v': min_v,
		'max_v': max_v,
		'switch_v': switch_v,
		'max_acc': max_acc,
		'min_acc': min_acc,
		'max_steer': max_steer,
		'min_steer': min_steer,
		'max_steer_vel': max_steer_vel,
		'max_inputs': max_inputs,
		'min_inputs': min_inputs,
		'max_rates': max_rates,
		'min_rates': min_rates,
		}
	return params

def get_param_dict(mean_dict, variation_dict, num_models, ground_truth = False):
	param_dict = {}
	for key in variation_dict.keys():
		param_dict[key] = mean_dict[key] * (np.random.uniform(-variation_dict[key], variation_dict[key], num_models) + 1)
	
	if ground_truth:
		for key in variation_dict.keys():
			param_array = param_dict[key]
			param_array[0] = mean_dict[key]
			
	return param_dict