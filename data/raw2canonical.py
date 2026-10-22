"""
This file contains code to transform real-world objects to "canonical" position/orientation/scale
which is:
(1) Primary symmetry plane: x = 0
(2) Secondary symmetry plane normal close to (0,1,0)
(3) Smallest sphere containing object has radius 1. 
(4) (max(y) + min(y)) / 2 = 0 & (max(z) + min(z)) / 2 = 0 
"""
import numpy as np 
import pandas as pd
from utils import get_postgres_engine, SimilarityTransform, get_best_fit_plane, load_data_single
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
from collections import namedtuple
from tqdm import tqdm
from dataclasses import asdict 
import warnings

def get_smallest_enclosing_sphere(xyz):
	"""
	SES: Smallest Enclosing Sphere 
	Solved via ipopt 

	min radius 
		subject to ||xyz[i] - center||^2 <= radius^2, for all points i on Convex Hull
	"""

	# Define Results Container 
	SES_result = namedtuple("SES", ["radius", "center"]) 

	# Extract Points on Convex Hull
	hull = ConvexHull(xyz)
	pts = hull.points[hull.vertices]

	# Define model
	model = pyo.ConcreteModel()
	model.I = pyo.RangeSet(1, pts.shape[0])
	model.J = pyo.RangeSet(1, pts.shape[1])
	model.radius = pyo.Var(domain=pyo.NonNegativeReals)
	model.center = pyo.Var(model.J, domain = pyo.Reals)
	model.pts = pyo.Param(model.I, model.J, initialize = lambda model, i,j: pts[i-1, j-1])

	# Objective
	model.obj = pyo.Objective(expr=model.radius, sense=pyo.minimize)

	# Constraints
	@model.Constraint(model.I)
	def pts_in_sphere_constraint(model, i):
		return sum((model.pts[i,j] - model.center[j]) ** 2 for j in model.J) <= model.radius**2

	# Run Optimization 
	solver = pyo.SolverFactory('ipopt')
	results = solver.solve(model, tee=False)
	if results.solver.termination_condition != TerminationCondition.optimal:
		raise ValueError("Optimization Failed")

	# Summarize Optimization Results
	smallest_radius = pyo.value(model.radius)
	center = np.array([pyo.value(model.center[j]) for j in model.J])
	return SES_result(smallest_radius, center)

def get_object_coordinate_system(xyz, cors1, cors2):
	"""
	With a 3D object and two pairs of symmetry correspondences, we can estimate two symmetry planes 
	with two normals n1 and n2. A third orthogonal normal can be found via cross(n1, n2).  

	In the 'canonical' coordinate system 
		(1) Primary symmetry plane normal is (1,0,0)
		(2) Secondary symmetry plane normal of the object is as close as possible to (0,1,0)
		(3) Third vector is as close as possible to (0,0,1)

	So, the idea is to find a rotation such that:
	(--       n1     --)                    (1,0,0)
	(--       n2     --)       ---->        (0,1,0)
	(-- cross(n1,n2) --) 		            (0,0,1)
	------------------------------------------------
	In practice n2 and n1 are not exactly orthogonal. So, the approach I took was to fix n1 
	and find n2_hat as close as possible to n2, but where n2_hat exactly orthogonal to n1. 
	------------------------------------------------------------
	FINDING n2_hat = B @ x is done as follows. 
	argmin ||B @ x - n2||^2
	  x
	such that  ||x|| = 1
	
	where B is an orthonormal basis for the null space of n1. 
	Lagrangian: L(x, lam) = x.T @ B.T @ B @ x + ||n2||^2 - 2 x.T @ B.T @ n2 - lam *(x.T @ x - 1)
	Steps 
		(1) dL/dx = 0 & algebra --> x(lam) = v / (1-lam), where v = B.T @ n2
		(2) Apply Constraint: x(lam).T @ x(lam) = 1 ----> (1-lam)^2 = v.T @ v
		(3) Solve for lam using quadratic formala: lam = 1 +/- ||v||, but want smallest lam for minimum. 
		(4) x(1 - ||v||) = v / ||v||
		(5) Therefore the vector closest to n2 that is orthogonal to n1 is given by 
			B @ v / ||v|| 
	------------------------------------------------------------
	"""
	# Get estimates of symmetry planes from correspondences 
	plane1 = get_best_fit_plane(xyz[cors1].mean(1))
	plane2 = get_best_fit_plane(xyz[cors2].mean(1))
	U,S,vt = np.linalg.svd(plane1[:3][:, None])
	B = U[:, 1:] # orthonormal basis of null space of normal of plane1
	c = plane2[:3] 
	v = B.T @ c
	x = v / np.linalg.norm(v)
	c_hat = B @ x
	coordinates = np.vstack((
		plane1[:3],
		c_hat, 
		np.cross(plane1[:3], c_hat)
	))
	return coordinates


def wahbas(coordinates, count=0):
	"""
	Implements solution found on Wikipedia with some extra checks to handle the 
	case where I pass an invalid coordinates. 
	https://en.wikipedia.org/wiki/Wahba%27s_problem
	"""
	if count == 2:
		raise ValueError("Infinite Recursion - shouldn't be possible and yet...")
	I = np.eye(3)
	B = np.einsum("ni, nj -> ij", I, coordinates)
	U,S,Vt = np.linalg.svd(B)
	M = np.diag([1,1,np.linalg.det(U) * np.linalg.det(Vt)])
	R2 = U @ M @ Vt
	cost = np.sum(np.square(I - R2 @ coordinates.T))
	if cost < 1e-8: # cost should be zero exactly. 
		return coordinates, R.from_matrix(R2)
	else:
		c2 = np.diag([1,1,-1.]) @ coordinates
		return wahbas(c2, count + 1)


def raw_to_canonical(xyz_init, cors1, cors2):
	"""
	Canonical has the following properties
	(1) The primary symmetry plane - re-estimated via least squares fit to midpoints of symmetry cors
		is set to be exactly x = 0. 
	(2) The object is rotated about the x-axis so that the normal 
		of the second symmetry plane is as close to (0,1,0) as possible. 
	(3) The object is translated so that (max(y) + min(y)) / 2 = 0 
		and (max(z) + min(z)) / 2 = 0
	(4) The object is scaled so that the maximum distance from the origin to 
		any point in the point cloud is 1.  
	"""

	xyz = xyz_init.copy()

	# Scale
	SES = get_smallest_enclosing_sphere(xyz)
	xyz /= SES.radius

	# Rotation
	coordinates = get_object_coordinate_system(xyz, cors1, cors2)
	coordinates, rotmat = wahbas(coordinates)
	rz, ry, rx = rotmat.as_euler("ZYX", degrees=True)
	xyz = rotmat.apply(xyz)

	# Translations
	tx = get_best_fit_plane(xyz[cors1].mean(axis=1))[3]
	ty, tz = -(xyz[:, 1:].max(axis=0) + xyz[:, 1:].min(axis=0)) / 2
	T = np.array([tx, ty, tz])
	xyz = xyz + T

	ST = SimilarityTransform(
		rot_x=rx,
		rot_y=ry,
		rot_z=rz,
		_translation_x=tx,
		_translation_y=ty,
		_translation_z=tz,
		scale=1 / SES.radius,
		degrees=True
	)
	return ST, xyz


def validate_run_single_output(xyz_raw, cors1, cors2, xyz2, ST, label):
	"""
	Check to ensure output has all desired properties. 
	"""
	assert np.allclose(ST.apply(xyz_raw), xyz2)
	plane1 = get_best_fit_plane(xyz2[cors1].mean(1))
	angle = np.degrees(np.arccos(plane1[0]))
	assert angle < .1, f"{label}: Primary symmetry plane normal {angle} degrees from (1,0,0)"
	assert plane1[3] < 1e-2
	n2 = get_best_fit_plane(xyz2[cors2].mean(1))[:3]
	angle = np.degrees(np.arccos(np.clip(n2[1], -1, 1)))
	# assert -1 <= angle <= 1, f"{label}: Angle was {angle} degrees"
	if abs(angle) > 1:
		warnings.warn(f"{label}: Symmetry plane normals are {angle:.03f} degrees apart")
	SES = get_smallest_enclosing_sphere(xyz2)
	assert np.abs(SES.radius - 1) < 1e-4
	YZ = xyz2[:, 1:]
	assert np.allclose((YZ.max(axis=0) + YZ.min(axis=0)) / 2, np.zeros(2))


def run_single(label, engine):
	"""
	Run code and check to confirm that all output has desired properties. 
	"""
	xyz, cors1, cors2 = load_data_single(label, engine, ['xyz', 'cors1', 'cors2'])
	ST, xyz2 = raw_to_canonical(xyz, cors1, cors2)
	validate_run_single_output(xyz, cors1, cors2, xyz2, ST, label)
	dct = asdict(ST)
	dct['label'] = label
	return dct


def run_all(engine):
	labels = pd.read_sql("SELECT DISTINCT label FROM triangles;", con=engine).label.tolist()
	lst = []
	for label in tqdm(labels):
		lst.append(run_single(label, engine))
	df = pd.DataFrame(lst)
	df['transform'] = ['raw2canonical'] * len(df)
	cols = ['label', 'transform', 'rot_x', 'rot_y', 'rot_z', '_translation_x',
				 '_translation_y', '_translation_z', 'scale']
	df = df[cols]
	return df



if __name__ == "__main__":
	engine = get_postgres_engine()
	df = run_all(engine)
	df.to_sql(name = "transforms", 
		index=False,
		if_exists='replace',
		con=engine)



	# label = "range_hood_0113"
	# xyz, cors1, cors2 = load_data_single(label, engine, ['xyz', 'cors1', 'cors2'])
	# ST, xyz2 = raw_to_canonical(xyz, cors1, cors2)
	# plane1 = get_best_fit_plane(xyz[cors1].mean(1))
	# plane2 = get_best_fit_plane(xyz[cors2].mean(1))


	# xyz3 = xyz.copy()

	# # Scale
	# SES = get_smallest_enclosing_sphere(xyz3)
	# xyz3 = xyz3 / SES.radius

	# # Rotation
	# coordinates = infer_rotation(xyz3, cors1, cors2)
	# rotmat, rssd = R.align_vectors(a = np.eye(3), b = coordinates)

	# xyz3 = rotmat.apply(xyz3)

	# # Translations
	# tx = get_best_fit_plane(xyz3[cors1].mean(axis=1))[3]
	# ty, tz = -(xyz3[:, 1:].max(axis=0) + xyz3[:, 1:].min(axis=0)) / 2
	# T = np.array([tx, ty, tz])
	# xyz3 = xyz3 + T



