"""
This file contains code to infer symmetry correspondences for real-world objects. 
"""
import numpy as np 
import trimesh
import pandas as pd
from utils import get_postgres_engine, to_homogeneous, get_max_distance_between_any_two_points
from scipy.spatial import KDTree, ConvexHull
from dataclasses import dataclass
from tqdm import tqdm
from numba import njit
from joblib import Parallel, delayed
import time
from sqlalchemy import text
from scipy.spatial._qhull import QhullError

class ReflectionTransform:
	def __init__(self, p, q):
		self.p = p
		self.q = q
		self.midpoint = (p + q) / 2
		v = p - q
		v /= np.linalg.norm(v)
		self.symmetry_plane = np.hstack((v, -v @ self.midpoint))

	def reflect_points(self, pts):
		ptsh = np.atleast_2d(to_homogeneous(pts))
		dists_to_plane = ptsh @ self.symmetry_plane
		reflected_points = pts - 2 * np.outer(dists_to_plane, self.symmetry_plane[:3])
		return np.squeeze(reflected_points)

	def reflect_vertex_normals(self, vns):
		return self.reflect_points(self.midpoint + vns) - self.midpoint



@dataclass
class RansacConfig:
	perpendicularity_threshold: float
	radius: float
	n_samples: int
	primary_sym_plane_normal: np.ndarray
	seed: int
	label: str



def ransac(mesh : trimesh.Trimesh, params : RansacConfig):
	pts = np.asarray(mesh.vertices)
	n_pts = len(pts)
	kd = KDTree(pts)
	best_plane = None
	max_volume = 0
	best_cors = None
	rng = np.random.default_rng(seed=params.seed)
	# for k in tqdm(range(params.n_samples)):
	for k in range(params.n_samples):
		i,j = rng.choice(n_pts, 2, replace=False)
		RT = ReflectionTransform(p = pts[i], q = pts[j])
		cos_theta = np.abs(params.primary_sym_plane_normal @ RT.symmetry_plane[:3])
		if cos_theta < params.perpendicularity_threshold:
			# print("primary sym plane: ", params.primary_sym_plane_normal)
			# print("psoposed syn planeJ: ", RT.symmetry_plane[:3])
			# print(f"Iter {k}: cos(theta) = {cos_theta}, threshold = {params.perpendicularity_threshold}")
			dists_to_plane = to_homogeneous(pts) @ RT.symmetry_plane
			one_side_fltr = dists_to_plane > 0
			P = pts[one_side_fltr]
			P_reflected = RT.reflect_points(P)
			result = kd.query_ball_point(P_reflected, r = params.radius)
			i_small = np.arange(len(pts))[one_side_fltr]
			try:
				i = np.repeat(i_small, [len(x) for x in result])
			except TypeError:
				# hacky way to handle an edge case where a symmetry plane is proposed  
				# that has only a single point on one side of the symmetry plane.  
				continue
			j = np.hstack(result).astype(np.int64)
			D = np.linalg.norm(pts[i] - pts[j], axis=1)
			K = np.argsort(D)
			i = i[K]
			j = j[K]
			segment_lengths = np.linalg.norm(pts[i] - pts[j], axis=1)
			L, cors = process_result(i,j,segment_lengths)
			try:
				hull = ConvexHull(mesh.vertices[cors].reshape(2*len(cors), 3))
			except QhullError:
				continue
			score = hull.volume * (hull.volume**2) / (hull.area**3)
			if score > max_volume:
				# print(f"cos(theta) = {cos_theta}")
				max_volume = score
				best_plane = RT.symmetry_plane
				best_cors = cors
				# print(f"volume(convex hull) = {max_volume}, plane = {np.round(best_plane, 2)}")
	return best_plane, np.vstack(best_cors)



@njit
def process_result(i,j,segment_lengths):
	n = len(i)
	visited = set()
	cors = []
	L = 0
	for l in range(n):
		I = i[l]
		J = j[l]
		if (I not in visited) and (J not in visited) and (I != J):
			cors.append((I,J))
			visited.add(I)
			visited.add(J)
			L += segment_lengths[l]
	return L, cors


def get_xyz_ijk(label, engine):
	xyz = pd.read_sql(f"SELECT x,y,z FROM xyz_raw WHERE label = '{label}';", con=engine)
	ijk = pd.read_sql(f"SELECT i,j,k FROM triangles WHERE label = '{label}';", con=engine)
	return xyz, ijk



def build_object_dcts(engine):
	labels = pd.read_sql("SELECT DISTINCT label FROM triangles;", con=engine).label.tolist()
	lst = []
	for label in labels:
		xyz, ijk = get_xyz_ijk(label, engine)
		mesh = trimesh.Trimesh(
			vertices=xyz.to_numpy(), 
			faces=ijk.to_numpy()
		)
		params1 = RansacConfig(
			perpendicularity_threshold=np.cos(np.radians(89)),
			radius = 0.005 * get_max_distance_between_any_two_points(xyz.to_numpy()),
			n_samples=10000,
			primary_sym_plane_normal=np.zeros(3), 
			seed = 1234,
			label=label
		)
		params2 = RansacConfig(
			perpendicularity_threshold=np.cos(np.radians(89)),
			radius = 0.005 * get_max_distance_between_any_two_points(xyz.to_numpy()),
			n_samples=25000,
			primary_sym_plane_normal=np.zeros(3), 
			seed = 1533,
			label=label
		)
		dct = {
			"label":label, 
			"mesh":mesh, 
			"params1":params1,
			"params2":params2
		}
		lst.append(dct)
	return lst 


def run_RANSACs(example_dct):
	plane1, cors1 = ransac(example_dct['mesh'], example_dct['params1'])
	params2 = example_dct['params2']
	params2.primary_sym_plane_normal = plane1[:3]
	plane2, cors2 = ransac(example_dct['mesh'], params2)
	v1 = example_dct['mesh'].vertices[cors1[:, 0]] - example_dct['mesh'].vertices[cors1[:, 1]]
	v2 = example_dct['mesh'].vertices[cors2[:, 0]] - example_dct['mesh'].vertices[cors2[:, 1]]
	L1_avg = np.linalg.norm(v1, axis=1).mean()
	L2_avg = np.linalg.norm(v2, axis=1).mean()
	if (len(cors1) == len(cors2)) and (L1_avg < L2_avg):
		# handle the case when the entire object is exactly mirror symmetrical. 
		return {
				"label":example_dct['label'],
				"plane1":plane2, 
				"plane2":plane1, 
				"cors1":cors2, 
				"cors2":cors1
			}
	else:
		return {
			"label":example_dct['label'],
			"plane1":plane1, 
			"plane2":plane2, 
			"cors1":cors1, 
			"cors2":cors2
		}


def symmetry_plane_data_as_dataframe(results):
	lst = []
	for dct in results:
		# primary symmetry plane
		tmp1 = {}
		tmp1['label'] = dct['label']
		tmp1['sym_plane'] = 'primary'
		tmp1['a'],tmp1['b'],tmp1['c'],tmp1['d'] = dct['plane1']

		# secondary symmetry plane
		tmp2 = {}
		tmp2['label'] = dct['label']
		tmp2['sym_plane'] = 'secondary'
		tmp2['a'], tmp2['b'], tmp2['c'],tmp2['d'] = dct['plane2']

		# append
		lst.extend([tmp1, tmp2])
	return pd.DataFrame(lst)


def cors_data_as_dataframe(results):
	df = pd.DataFrame(columns = ['label', 'sym_plane', 'idx', 'i', 'j'])
	for dct in results:
		# Primary symmetry cors
		tmp1 = pd.DataFrame(columns = ['label', 'sym_plane', 'idx', 'i', 'j'])
		n1 = len(dct['cors1'])
		cors1 = np.hstack((
			np.arange(n1)[:, None],
			dct['cors1']
		))
		tmp1[['idx', 'i', 'j']] = cors1
		tmp1['sym_plane'] = ['primary'] * n1
		tmp1['label'] = [dct['label']] * n1

		# Secondary Symmetry cors
		tmp2 = pd.DataFrame(columns = ['label', 'sym_plane', 'idx', 'i', 'j'])
		n2 = len(dct['cors2'])
		cors2 = np.hstack((
			np.arange(n2)[:, None],
			dct['cors2']
		))
		tmp2[['idx', 'i', 'j']] = cors2
		tmp2['sym_plane'] = ['secondary'] * n2
		tmp2['label'] = [dct['label']] * n2

		# Combine 
		df = pd.concat([df, tmp1, tmp2], axis=0, ignore_index=True)
	return df




if __name__ == "__main__":

	engine = get_postgres_engine()

	# Pull Data from SQL and format
	examples = build_object_dcts(engine)

	# Run RANSAC for all objects
	start = time.time() 
	RANSAC_results = Parallel(n_jobs=-1)(delayed(run_RANSACs)(dct) for dct in examples)
	print(f"Total time = {time.time() - start}")


	# RANSAC_results is a list of dictionaries - extract data and format as DataFrames
	sym_plane_df = symmetry_plane_data_as_dataframe(RANSAC_results)
	sym_cors_df = cors_data_as_dataframe(RANSAC_results)


	# # Write dataframes to database
	sym_plane_df.to_sql(
		name = "symmetry_planes_raw", 
		index=False,
		if_exists='replace',
		con=engine)

	sym_cors_df.to_sql(
		name = "sym_cors", 
		index=False,
		method='multi',
		chunksize=5000,
		if_exists='replace',
		con=engine)

	with engine.begin() as connection:
		connection.execute(text("ALTER TABLE symmetry_planes_raw ADD PRIMARY KEY (label, sym_plane);"))
		connection.execute(text("ALTER TABLE sym_cors ADD PRIMARY KEY (label, sym_plane, idx);"))
		print("created indices")




	### Single example below of what is parallelized above. 
	# label = 'bookshelf_0566'
	# xyz, ijk = get_xyz_ijk(label, engine)
	# mesh = trimesh.Trimesh(
	# 	vertices=xyz.to_numpy(), 
	# 	faces=ijk.to_numpy()
	# )


	# params1 = RansacConfig(
	# 	perpendicularity_threshold=85,
	# 	radius = 0.005 * get_max_distance_between_any_two_points(xyz.to_numpy()),
	# 	n_samples=10000,
	# 	primary_sym_plane_normal=np.zeros(3), 
	# 	seed = 1234,
	# 	label=label
	# )

	# plane1, cors1 = ransac(mesh, params1)

	# params2 = RansacConfig(
	# 	perpendicularity_threshold=np.cos(np.radians(89)),
	# 	radius = 0.01 * get_max_distance_between_any_two_points(xyz.to_numpy()),
	# 	n_samples=50000,
	# 	primary_sym_plane_normal=plane1[:3], 
	# 	seed = 1533,
	# 	label=label
	# )
	# # print("params2 = ", params2)

	# plane2, cors2 = ransac(mesh, params2)

	# print(label)
	# print(f"plane1 = {plane1}")
	# print(f"plane2 = {plane2}")

	# L1_avg = np.linalg.norm(mesh.vertices[cors1[:, 0]] - mesh.vertices[cors1[:, 1]], axis=1).mean()
	# L2_avg = np.linalg.norm(mesh.vertices[cors2[:, 0]] - mesh.vertices[cors2[:, 1]], axis=1).mean()


	# radius = 10 
	# theta = np.linspace(0,2*np.pi, 150)
	# B1 = np.linalg.svd(plane1[:3][:, None])[0][:, 1:]
	# xyz1 = radius * (np.outer(np.cos(theta), B1[:,0]) + np.outer(np.sin(theta), B1[:, 1])) - plane1[3] * plane1[:3]

	# # Z1 = -(plane1[0] * X + plane1[1] * Y + plane1[3]) / plane1[2]
	# # Z2 = -(plane2[0] * X + plane2[1] * Y + plane2[3]) / plane2[2]

	# # 3. Create a 3D plot
	# import matplotlib.pyplot as plt


	# xyz, cors1, cors2 = load_data_single("bookshelf_0565", engine)

	# fig = plt.figure(figsize=(8, 6))
	# ax = fig.add_subplot(projection='3d')

	# # 4. Plot the surface
	# ax.plot(*xyz.T, alpha=.5)
	# ax.scatter(*xyz.T)
	# for i,j in cors2:
	# 	ax.plot(*xyz[[i,j]].T, color = 'red')

	# for i,j in cors1:
	# 	ax.plot(*xyz[[i,j]].T, color = 'blue')



	# ax.axis('equal')

	# fig.show()



