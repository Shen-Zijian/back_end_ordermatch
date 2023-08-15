import numpy as np
import pandas as pd
from numpy import *
import pickle
import random
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from io import StringIO
import gdown
from werkzeug.utils import secure_filename
import os

driver_behaviour_model = pickle.load(open('./Driver_Behaviour.pickle', 'rb'))

app = Flask(__name__)
CORS(app)


@app.route('/process_data', methods=['POST'])
def order_driver_info():
	method = request.form['method']
	radius = float(request.form['radius'])
	driver_info = request.files['driver_info']
	order_info = request.files['order_info']

	filename1 = secure_filename(driver_info.filename)
	driver_info.save(os.path.join('./', filename1))
	driver_data = pd.read_csv(os.path.join('./', filename1))

	filename2 = secure_filename(order_info.filename)
	order_info.save(os.path.join('./', filename2))
	order_data = pd.read_csv(os.path.join('./', filename2))

	num_order = order_data['order_id'].nunique()
	num_driver = driver_data['driver_id'].nunique()

	driver_data_temp = driver_data.copy()
	driver_loc_array = np.tile(driver_data_temp.values, (num_order, 1))
	order_data_temp = order_data.copy()
	request_array = np.repeat(order_data_temp.values, num_driver, axis=0)
	dis_array = distance_array(request_array[:, :2], driver_loc_array[:, :2])
	if method == "broadcasting":
		order_driver_pair = np.vstack(
			[request_array[:, 0], request_array[:, 4], request_array[:, 2], request_array[:, 1], driver_loc_array[:, 0],
			 driver_loc_array[:, 3], driver_loc_array[:, 2], driver_loc_array[:, 1], request_array[:, 3],
			 dis_array[:]]).T
		order_driver_info = pd.DataFrame(order_driver_pair.tolist(),
		                                 columns=['order_id', 'order_region', 'order_lat', 'order_lng', 'driver_id',
		                                          'driver_region', 'driver_lat', 'driver_lng', 'reward_units',
		                                          'pick_up_distance'])
		matched_result = order_broadcasting(order_driver_info, radius)
	elif method == "dispatch":
		print("get in dispatch", type(radius))
		flag = np.where(dis_array <= radius)[0]
		print("flag", flag)
		if len(flag) > 0:
			order_driver_pair = np.vstack(
				[request_array[flag, 0], request_array[flag, 4], request_array[flag, 2], request_array[flag, 1],
				 driver_loc_array[flag, 0], driver_loc_array[flag, 3], driver_loc_array[flag, 2],
				 driver_loc_array[flag, 1], request_array[flag, 3], dis_array[flag]]).T
		order_driver_info = pd.DataFrame(order_driver_pair.tolist(),
		                                 columns=['order_id', 'order_region', 'order_lat', 'order_lng', 'driver_id',
		                                          'driver_region', 'driver_lat', 'driver_lng', 'reward_units',
		                                          'pick_up_distance'])
		matched_result = order_dispatch(order_driver_info, radius)
	else:
		matched_result = jsonify("ERROR: Method Not Found !")
	print("matched_result", matched_result)
	return matched_result, 200


def driver_decision(distance, reward, lr_model):
	"""

    :param reward: numpyarray, price of order
    :param distance: numpyarray, distance between current order to all drivers
    :param numpyarray: n, price of order
    :return: pandas.DataFrame, the probability of drivers accept the order.
    """
	r_dis, c_dis = distance.shape
	temp_ = np.dstack((distance, reward)).reshape(-1, 2)
	result = lr_model.predict_proba(temp_).reshape(r_dis, c_dis, 2)
	result = np.delete(result, 0, axis=2)
	result = np.squeeze(result, axis=2)
	return result


def generate_random_num(length):
	if length < 1:
		res = 0
	else:
		res = random.randint(0, length)
	return res


def distance_array(coord_1, coord_2):
	"""
    :param coord_1: array of coordinate
    :type coord_1: numpy.array
    :param coord_2: array of coordinate
    :type coord_2: numpy.array
    :return: the array of manhattan distance of these two-point pair
    :rtype: numpy.array
    """
	coord_1 = coord_1.astype(float)
	coord_2 = coord_2.astype(float)
	coord_1_array = np.radians(coord_1)
	coord_2_array = np.radians(coord_2)
	dlon = np.abs(coord_2_array[:, 0] - coord_1_array[:, 0])
	dlat = np.abs(coord_2_array[:, 1] - coord_1_array[:, 1])
	r = 6371

	alat = np.sin(dlat / 2) ** 2
	clat = 2 * np.arctan2(alat ** 0.5, (1 - alat) ** 0.5)
	lat_dis = clat * r

	alon = np.sin(dlon / 2) ** 2
	clon = 2 * np.arctan2(alon ** 0.5, (1 - alon) ** 0.5)
	lon_dis = clon * r

	manhattan_dis = np.abs(lat_dis) + np.abs(lon_dis)

	return manhattan_dis


def order_broadcasting(order_driver_info, broadcasting_scale=1):
	"""

    :param order_driver_info: the information of drivers and orders
    :param broadcasting_scale: the radius of order broadcasting
    :return: matched driver order pair
    """

	id_order = order_driver_info['order_id'].unique()
	id_driver = order_driver_info['driver_id'].unique()

	# num of orders and drivers
	num_order = order_driver_info['order_id'].nunique()
	num_driver = order_driver_info['driver_id'].nunique()
	dis_array = np.array(order_driver_info['pick_up_distance'], dtype='float32').reshape(num_order, num_driver)
	distance_driver_order = dis_array.reshape(num_order, num_driver)
	driver_region_array = np.array(order_driver_info['driver_region'], dtype='float32').reshape(num_order, num_driver)
	order_region_array = np.array(order_driver_info['order_region'], dtype='float32').reshape(num_order, num_driver)
	driver_lat_array = np.array(order_driver_info['driver_lat'], dtype='float32').reshape(num_order, num_driver)
	driver_lng_array = np.array(order_driver_info['driver_lng'], dtype='float32').reshape(num_order, num_driver)
	order_lat_array = np.array(order_driver_info['order_lat'], dtype='float32').reshape(num_order, num_driver)
	order_lng_array = np.array(order_driver_info['order_lng'], dtype='float32').reshape(num_order, num_driver)

	price_array = np.array(order_driver_info['reward_units'], dtype='float32').reshape(num_order, num_driver)

	radius_array = np.full((num_order, num_driver), broadcasting_scale, dtype='float32')
	driver_decision_info = driver_decision(distance_driver_order, price_array, driver_behaviour_model)
	'''
    Choose Driver with probability
    '''
	for i in range(num_order):
		for j in range(num_driver):
			if distance_driver_order[i, j] > radius_array[i, j]:
				driver_decision_info[i, j] = 0  # delete drivers further than broadcasting_scale
				# match_state_array[i, j] = 2

	random.seed(10)
	temp_random = np.random.random((num_order, num_driver))
	driver_pick_flag = (driver_decision_info > temp_random) + 0
	driver_id_list = []
	order_id_list = []
	reward_list = []
	pick_up_distance_list = []
	index = 0
	for row in driver_pick_flag:
		temp_line = np.argwhere(row == 1)
		if len(temp_line) >= 1:
			temp_num = generate_random_num(len(temp_line) - 1)
			row[:] = 0
			row[temp_line[temp_num, 0]] = 1
			driver_pick_flag[index, :] = row
			driver_pick_flag[index + 1:, temp_line[temp_num, 0]] = 0

		index += 1
	matched_pair = np.argwhere(driver_pick_flag == 1)
	matched_dict = {}
	for item in matched_pair:
		matched_dict[id_order[item[0]]] = [order_region_array[item[0], item[1]], order_lat_array[item[0], item[1]],
		                                   order_lng_array[item[0], item[1]], id_driver[item[1]],
		                                   driver_region_array[item[0], item[1]], driver_lat_array[item[0], item[1]],
		                                   driver_lng_array[item[0], item[1]]]
		driver_id_list.append(id_driver[item[1]])
		order_id_list.append(id_order[item[0]])
		reward_list.append(price_array[item[0], item[1]])
		pick_up_distance_list.append(distance_driver_order[item[0], item[1]])
	result = []
	for item in id_order.tolist():
		if item in matched_dict:
			result.append(
				[item, matched_dict[item][0], matched_dict[item][1], matched_dict[item][2], matched_dict[item][3],
				 matched_dict[item][4], matched_dict[item][5], matched_dict[item][6], broadcasting_scale])

	result_columns = ['order_id', 'order_region', 'order_lat', 'order_lng', 'driver_id', 'driver_region', 'driver_lat',
	                  'driver_lng', 'radius']

	# 创建 DataFrame
	result_df = pd.DataFrame(result, columns=result_columns)
	return result_df.to_json(orient='records')


def order_dispatch(dispatch_observ, radius=1):
	dispatch_observ['reward_units'] = 0. + dispatch_observ['reward_units'].values
	dic_dispatch_observ = dispatch_observ.to_dict(orient='records')
	# print("after =",dispatch_observ['reward_units'])

	dispatch_action = []

	# get orders and drivers
	l_orders = dispatch_observ['order_id'].unique()  # df: order id
	l_drivers = dispatch_observ['driver_id'].unique()  # df: driver id
	# print("before =",l_orders)
	# print('ledrivers =',l_drivers.dtype)
	M = len(l_orders)  # the number of orders
	N = len(l_drivers)  # the number of drivers

	# coefficients and parameters, formulated as M * N matrix
	non_exist_link_value = 0.
	matrix_reward = non_exist_link_value + np.zeros(
		[M, N])  # reward     # this value should be smaller than any possible weights
	matrix_driver_region = np.zeros([M, N])
	matrix_driver_lat = np.zeros([M, N])
	matrix_driver_lng = np.zeros([M, N])
	matrix_order_lat = np.zeros([M, N])
	matrix_order_lng = np.zeros([M, N])
	matrix_order_region = np.zeros([M, N])  # pick up distance
	matrix_x_variables = np.zeros([M, N])  # 1 means there is potential match. otherwise, 0

	index_order = np.where(dispatch_observ['order_id'].values.reshape(dispatch_observ.shape[0], 1) == l_orders)[1]
	index_driver = np.where(dispatch_observ['driver_id'].values.reshape(dispatch_observ.shape[0], 1) == l_drivers)[1]

	matrix_reward[index_order, index_driver] = dispatch_observ['reward_units'].values
	matrix_driver_region[index_order, index_driver] = dispatch_observ['driver_region'].values
	matrix_driver_lat[index_order, index_driver] = dispatch_observ['driver_lat'].values
	matrix_driver_lng[index_order, index_driver] = dispatch_observ['driver_lng'].values
	matrix_order_region[index_order, index_driver] = dispatch_observ['order_region'].values
	matrix_order_lat[index_order, index_driver] = dispatch_observ['order_lat'].values
	matrix_order_lng[index_order, index_driver] = dispatch_observ['order_lng'].values

	matrix_x_variables[index_order, index_driver] = 1

	# algorithm ----------------------------------------------------------------------------------------------------
	# initialize lower bound of the solution
	initial_best_reward = 0
	initial_best_solution = np.zeros([M, N])
	dic_dispatch_observ.sort(key=lambda od_info: od_info['reward_units'], reverse=True)
	assigned_order = set()
	assigned_driver = set()
	initial_dispatch_action = []
	for od in dic_dispatch_observ:
		# make sure each order is assigned to one driver, and each driver is assigned with one order
		if (od["order_id"] in assigned_order) or (od["driver_id"] in assigned_driver):
			continue
		assigned_order.add(od["order_id"])
		assigned_driver.add(od["driver_id"])
		initial_dispatch_action.append(dict(order_id=od["order_id"], driver_id=od["driver_id"]))
	df_init_dis = pd.DataFrame(initial_dispatch_action)
	index_order_init = np.where(df_init_dis['order_id'].values.reshape(df_init_dis.shape[0], 1) == l_orders)[1]
	index_driver_init = np.where(df_init_dis['driver_id'].values.reshape(df_init_dis.shape[0], 1) == l_drivers)[1]
	initial_best_reward += np.sum(matrix_reward[index_order_init, index_driver_init])
	initial_best_solution[index_order_init, index_driver_init] = 1

	max_iterations = 30  # 25
	u = np.zeros(N)  # initialization
	Z_LB = initial_best_reward  # the lower bound of original problem that is initialized with the naive algorithm
	Z_UP = float('inf')  # infinity
	theta = 1.0
	gap = 0.0001

	# ---------------------------------------------Start iteration--------------------------------------------------
	for t in range(1, max_iterations + 1):
		matrix_x = np.zeros([M, N])
		QI = matrix_reward - u
		QI_masked = np.ma.masked_where(matrix_x_variables != 1, QI)
		idx_col_array = np.argmax(QI_masked, axis=1)
		idx_row_array = np.array(range(M))
		matrix_x[idx_row_array, idx_col_array] = 1

		# calculate Z_UP and Z_D
		Z_D = np.sum(u) + np.sum(matrix_reward * matrix_x)
		Z_UP = Z_D if Z_D < Z_UP else Z_UP

		# stage 1
		copy_matrix_reward = non_exist_link_value + np.zeros([M, N])
		copy_matrix_reward[idx_row_array, idx_col_array] = matrix_reward[idx_row_array, idx_col_array]
		copy_matrix_x = np.zeros([M, N])
		idx_col_array = np.array(range(N))
		idx_row_array = np.argmax(copy_matrix_reward, axis=0)
		con = copy_matrix_reward[idx_row_array, idx_col_array] > non_exist_link_value
		idx_col_array = idx_col_array[con]
		idx_row_array = idx_row_array[con]
		if len(idx_row_array) > 0:
			copy_matrix_x[idx_row_array, idx_col_array] = 1

		# stage 2
		index_existed_pair = np.where(copy_matrix_x == 1)
		index_drivers_with_order = np.unique(index_existed_pair[1])
		index_drivers_without_order = np.setdiff1d(np.array(range(N)), index_drivers_with_order)
		index_orders_with_driver = np.unique(index_existed_pair[0])
		index_orders_without_driver = np.setdiff1d(np.array(range(M)), index_orders_with_driver)

		if len(index_orders_without_driver) != 0:
			second_allocated_driver = []
			for m in index_orders_without_driver.tolist():
				con_second = np.isin(index_drivers_without_order, second_allocated_driver)
				if np.all(con_second):
					break
				else:
					reward_array = matrix_reward[m][index_drivers_without_order]
					masked_reward_array = np.ma.masked_where(con_second, reward_array)
					index_reward = np.argmax(masked_reward_array)
					if masked_reward_array[index_reward] > 0:
						index_driver = index_drivers_without_order[index_reward]
						second_allocated_driver.append(index_driver)
						copy_matrix_x[m][index_driver] = 1

		# stage 3
		new_Z_LB = np.sum(copy_matrix_x * matrix_reward)
		if new_Z_LB > Z_LB:
			Z_LB = new_Z_LB
			initial_best_solution = np.zeros([M, N])
			initial_best_solution[copy_matrix_x == 1] = 1

		# update u
		sum = 0
		sum_m = np.sum(matrix_x, axis=0)
		sum = np.sum((1 - sum_m) ** 2)
		if sum == 0:
			sum = 0.00001  # given a small value
		k_t = theta * (Z_D - Z_LB) / sum

		u = u + k_t * (sum_m - 1) / t
		u[u < 0] = 0

		if (Z_UP == 0) or ((Z_UP - Z_LB) / Z_UP <= gap):
			matrix_x = initial_best_solution
			break
		if t == max_iterations:
			matrix_x = initial_best_solution
			break

	# solution
	index_existed = np.where(matrix_x == 1)
	for m, index_driver in zip(index_existed[0].tolist(), index_existed[1].tolist()):
		dispatch_action.append([l_orders[m], matrix_order_region[m][index_driver], matrix_order_lat[m][index_driver],
		                        matrix_order_lng[m][index_driver], l_drivers[index_driver],
		                        matrix_driver_region[m][index_driver], matrix_driver_lat[m][index_driver],
		                        matrix_driver_lng[m][index_driver], radius])
	# print("type of dispatch_action :",type(dispatch_action))
	result_columns = ['order_id', 'order_region', 'order_lat', 'order_lng', 'driver_id', 'driver_region', 'driver_lat',
	                  'driver_lng', 'radius']
	result_df = pd.DataFrame(dispatch_action, columns=result_columns)
	return result_df.to_json(orient='records')


if __name__ == '__main__':
	app.run(host='0.0.0.0')