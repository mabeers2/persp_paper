"""
Computes theta_persp
"""
import numpy as np 
from numpy.typing import NDArray
from utils import get_postgres_engine, to_homogeneous, SimilarityTransform, load_data_single, get_persp_image
import pandas as pd
from scipy.optimize import minimize


class VanishingPoint:
	def __init__(self, vp: NDArray, focal_length: int|float):
		"""
		Assumes that (1) if vp is 3D then vanishing point is homogeneous
		(2) if vp is 2D then vanishing point has been computed as (fx/z, fy/z)
		-------------------------
		Has 3 properties:
		(1) xy: vanishing point of the form (fx/z, fy/z)
		(2) xyz: vanishing point of the form (a,b,c) where vp in image is (a/c, b/c) and ||abc|| = 1
		(3) focal_length
		(4) slant: Convert Vanishing point location to slant. 
		"""
		self.focal_length = focal_length
		if len(vp) == 2:
			vph = np.hstack((vp, focal_length))
			vph /= np.linalg.norm(vph)
			self.xy = vp
			self.xyz = vph
		elif len(vp) == 3:
			self.xyz = vp / np.linalg.norm(vp)
			self.xy = focal_length * vp[:2] / vp[2]
		else:
			raise ValueError("Vanishing Point must be NDArray of length 2 or 3.")

	@property
	def slant(self):
		"""
		Suppose parallel 3D line segments given by L(s) = u_i + n * s, 
		where s a scalar, n in R^3 a direction and u_i an intercept. 
		Then, vanishing point is (f * n_x / n_z, f*n_y/n_z) = (v_x,v_y)
		If focal length f known, then can solve for n_z. 
		Viewing direction assumed to be (0,0,1) so 
		cos(slant) = dot([0,0,1], [n_x,n_y,n_z]) = n_z
		"""
		nz = self.focal_length / np.sqrt(self.focal_length**2 + np.sum(np.square(self.xy)))
		return np.degrees(np.arccos(np.abs(nz)))

	@property
	def tilt(self):
		return np.degrees(np.atan2(self.xyz[1], self.xyz[0]))

	def __repr__(self):
		s = f"VanishingPoint(slant={self.slant:.02f} degrees, vp_homogeneous = {self.xyz.round(3)})"
		return s



def get_initial_vp_estimate(persp: NDArray, cors: NDArray, focal_length: int|float):
	"""
	Step 1: Get equations of lines for each symmetry line segment in form ax+by+c = 0
		Given two homogeneous points u = (x1, y1, 1) & v = (x2,y2,1)
		Want dot(u, abc) = 0 & dot(v, abc) = 0. 
		abc satisfying that given by cross(u,v)

	Step 2: Given many such lines abc_i, want a point that lies on all of them. 
		Not possible given noise in symmetry line segments so instead minimize 
		argmin ||A @ x||^2 
		   x    s.t. ||x|| = 1
		where each row of A contains coefficients (a,b,c)
		This solved via SVD/eigh and returns homogeneous point (u,v,w) 
		which is converted VanishingPoint object 
	"""
	persp_homogeneous = to_homogeneous(persp, focal_length)
	w = np.cross( persp_homogeneous[cors[:, 0]], persp_homogeneous[cors[:, 1]] ) 
	u,s,vt = np.linalg.svd(w)
	vp_homogeneous = vt[-1]
	return VanishingPoint(vp = vp_homogeneous, focal_length=focal_length)


def refinement_cost(vp: NDArray, persp:NDArray, cors:NDArray):
	"""See paper for cost function explanation."""
	midpoints = np.mean(persp[cors], axis=1)
	P = persp[cors[:, 0]]

	u = vp - midpoints
	alpha = np.vecdot(P - midpoints, u) / np.vecdot(u,u)
	projected_points = midpoints + u * alpha[:, None]
	d = np.linalg.norm(P - projected_points, axis=1)
	return d.mean()


def get_vanishing_point(persp: NDArray, cors: NDArray, focal_length: int|float):
	# Run optimization 
	VP_init = get_initial_vp_estimate(persp, cors, focal_length)
	opt_out = minimize(
		fun=refinement_cost, 
		x0=VP_init.xy, 
		method='Nelder-Mead',
		args = (persp, cors, )
	)

	# Return VanishingPoint unless optimization failed
	if opt_out.success: 
		return VanishingPoint(vp = opt_out.x, focal_length = focal_length)
	else:
		raise ValueError("Optimization Failed")


def get_angle_s(vp: VanishingPoint, midpoints: NDArray):
	"""
	Angle s = maximum angle between symmetry line segments in degrees. 
	---------------------------
	(1) Compute vectors from vp to midpoints of symmetry line segments
	(2) Randomly select one vector v as a reference
	(3) Compute signed angles A between all w and v.
	(4) return maximum(A) - minimum(A)
	"""
	u = midpoints - vp.xy
	v = u[0]
	w = u[1:]
	l = w[:, 0] * v[1] - w[:, 1] * v[0]
	r = w @ v
	angles = np.degrees(np.atan2(l, r))
	return angles.max() - angles.min()







if __name__ == "__main__":
	engine = get_postgres_engine()
	label = "bed_0262"
	xyz, cors1 = load_data_single(label, engine, ['xyz', 'cors1'])
	st = SimilarityTransform.from_sql(label, "raw2canonical", engine)
	st2 = SimilarityTransform(123,60,231,0,0,-4,1)
	xyzR = st2.apply(st.apply(xyz))
	focal_length = 1
	persp = get_persp_image(xyzR, focal_length)
	VP = get_vanishing_point(persp, cors1, focal_length)
	angle_s = get_angle_s(vp=VP, midpoints=persp[cors1].mean(axis=1))
	print(f"Object {label}: theta_persp = {VP.slant:.03f} Degrees, Angle s = {angle_s:.03f}")

	# import matplotlib.pyplot as plt
	# fig,ax = plt.subplots(1,1)
	# ax.axis('equal')
	# for i,j in cors1:
	# 	x = img[[i,j], 0]
	# 	y = img[[i,j], 1]
	# 	ax.plot(x,y, color = 'blue')

	# ax.scatter(*vanishing_point, color = 'red')

	# fig.show()



	# xyz_new, cors_new = persp_to_3D(persp, cors1, VP)
	# xyz1 = xyzR[cors1].reshape(-1,3)
	# scaler = np.median(xyz1 / xyz_new)
	# xyz_new2 = scaler * xyz_new






