"""
This file contains functions for taking 2D point correspondences (images) and turning them into 3D point clouds. 
Functions
------------------------------------------------------------------
(1) one_parameter_family -> Appropriate for orthographic images 
(2) persp_to_3D -> Suitable for perspective images - solves a linear least squares problem to get reconstruction.
	(2a)  _get_minimizing_params: The 3D symmetric shape is parameterized and this function does the minimization 
	(2b)  _reconstruct_persp: Given minimizing params from _get_minimizing_params, return 3D point cloud
(3a) to3dIterObject -> Class containing residuals/jacobian functions useful for picking 3D symmetrical shape 
	that minimizes reprojection error. 
(3b) persp_to_3D_LM: Given a to3dIterObject class, does Levenberg-Marquardt via scipy.optimize.least_squares
"""
import numpy as np 
import pandas as pd
from numpy.typing import NDArray
from theta_persp import VanishingPoint, get_vanishing_point
from utils import get_postgres_engine, to_homogeneous, SimilarityTransform, load_data_single, get_persp_image
from scipy.optimize import least_squares
from scipy.sparse import block_diag, coo_array, hstack
from sympy import symbols, Matrix, lambdify, sin, cos
from sympy.abc import theta, tau
import time
from typing import Literal

def one_parameter_family(slant:float, img:NDArray, cors:NDArray, intercept:float=0.0):
	"""
	Implementation of the One Parameter Family that assumes perfect
	Orthographic image of a perfectly symmetrical object. 
	Tilt can be anything, Slant expected in degrees. 

	Inputs: 
	slant (degrees)
	img: ndarray (n x 2)
	cors: ndarray (m x 2)
	
	Outputs:
	xyz: ndarray (n x 3), where z_k = np.nan if k not in cors.  
	"""
	assert 0 < np.abs(slant) < 90
	i, j = cors.T
	edges = img[i] - img[j]
	edge_lengths = np.linalg.norm(edges, axis=1, keepdims=True)
	signs = np.sign(edges @ edges[np.argmax(edge_lengths)])
	edges = edges / edge_lengths * signs[:, None]
	slant_rad = np.radians(slant)
	nx = edges[:, 0] * np.sin(slant_rad)
	ny = edges[:, 1] * np.sin(slant_rad)
	nz = np.repeat(np.cos(slant_rad), len(edges))
	xi, yi = img[i].T
	xj, yj = img[j].T
	z = np.full(len(img), np.nan)
	alpha = (nx * (xi-xj) + ny * (yi-yj)) / (2 * (nx**2 + ny**2))
	zi = (alpha - xi*nx - yi*ny - intercept) / nz
	zj = zi - 2*(nx*xi + ny*yi + nz*zi + intercept)*nz
	z[i], z[j] = zi, zj
	return np.hstack((img, z[:, None]))



def _get_minimizing_params(persp:NDArray, cors:NDArray, VP:VanishingPoint, sym_plane_intercept:float=1.0):
	# Get vp_homogeneous null space
	n = VP.xyz
	# u,s,vt = np.linalg.svd(n[:, None])
	# BASIS = u[:, 1:] # Orthonormal basis for null space of n, same as null_space function output.
	slant = np.radians(VP.slant)
	tilt = np.radians(VP.tilt)
	b1 = np.array([np.cos(slant)*np.cos(tilt), np.cos(slant)*np.sin(tilt), -np.sin(slant)])
	b2 = np.array([[-np.sin(tilt), np.cos(tilt), 0]])
	BASIS = np.vstack((b1, b2)).T

	# Build One Matrix A per symmetry correspondence. Where is vmap when you need it. 
	m = len(cors)
	B = np.tile(BASIS, (m,1,1))
	N = np.tile(n[:, None], (m,1,1))
	p,q = to_homogeneous(persp[cors], VP.focal_length)[...,None].transpose(1,0,2,3)
	Z = np.zeros((m,3,1))
	A = np.block([
		[B,  N, -p,  Z],
		[B, -N,  Z, -q]
	])
	b = sym_plane_intercept * np.tile(n, 2)
	x = np.linalg.pinv(A) @ b
	return 	x, n, BASIS


def _reconstruct_persp(n, B, alpha, D, sym_plane_intercept):
	"""
	Implements: 
	P = midpoints + D * n
	Q = midpoints - D * n
	and puts P,Q into an array. 
	"""
	midpoints = alpha @ B.T - sym_plane_intercept * n
	P = midpoints + np.outer(D,n)
	Q = midpoints - np.outer(D,n)
	M = 2 * len(D)
	xyz = np.zeros((M, 3))
	xyz[::2]  = P
	xyz[1::2] = Q
	new_cors = np.arange(M).reshape(-1,2)
	return xyz, new_cors



def persp_to_3D_closed_form(persp:NDArray, cors:NDArray, VP: VanishingPoint, sym_plane_intercept:float=1.0):
	"""
	Perspective image to 3D via vanishing point + symmetry assumption. Closed form solution using distanced in 3D. 
	----------------------------------------- 
	Conceptually, want to find Symmetric XYZ s.t. ||(fX/Z, fY/Z) - p||^2 + ||(fX'/Z', fY'/Z') - q||^2
	minimal BUT the division by Z renders closed form solution impossible. (Need iterative algorithm like levenberg-marquardt)
	So, here introduce a closed form solution to a similar problem which is:
	-----------------------------------------
	   argmin     ||[(-d*n + B@alpha) + D * n] - s*Q||^2 + 
	alpha,D,s,t        ||[(-d*n + B@alpha) - D * n] - t*P||^2

	where P, Q are the homogeneous version of p,q
	-----------------------------------------
	Explanation: 
	(1) Symmetry plane given by [n, d] where n = (a,b,c) in ax + by + cz + d = 0, and ||n|| = 1
	(2) Points in symmetry plane can be defined as (point in plane) + (linear combination of basis for null space of n)
		(2.1) (point in plane) given by (-d * n) 
		(2.2) orthonormal basis for null space of n computed via SVD (name the basis B)
		(2.3) Point in symmetry plane is therefore M = (-d*n + B @ alpha) for some alpha
	(3) The actual symmetrical points are then P, P' = [(-d*n + B @ alpha) +/- D * n] for some D and alpha.  
	(4) Instead of looking to minimize re-projection error, minimize squared distance in 3D between P, P' 
		and s*(p, focal_length), t*(p',focal_length) respectively. 
	(5) The objective function can be rearranged into ||Ax - b||^2 which is then solved for x via least squares. 
		Specifically, x is a vector containing [alpha, D,s, t]. The variables s & t are thrown away and the 3D reconstruction 
		is the midpoint +/- D*n
	"""
	params, n, B = _get_minimizing_params(persp, cors, VP, sym_plane_intercept)
	xyz, new_cors = _reconstruct_persp(
		n=n, 
		B=B, 
		alpha=params[:, :2], 
		D=params[:, 2], 
		sym_plane_intercept=sym_plane_intercept
	)
	return xyz, new_cors


class to3dIterObject:

	def __init__(self, persp: NDArray, cors:NDArray, focal_length: float=1.0, sym_plane_intercept:float=1.0):
		self.img = persp[cors].reshape(-1,2)
		self.x1 = self.img[0::2, 0]
		self.y1 = self.img[0::2, 1]
		self.x2 = self.img[1::2, 0]
		self.y2 = self.img[1::2, 1]
		self.focal_length = focal_length
		self.d = sym_plane_intercept
		self.cors = np.arange(2*len(cors)).reshape(-1,2)
		self.n_residuals = 4 * len(cors)
		self.n_variables = 3 * len(cors) + 2

		# Define SymPy symbols
		x1,x2 = symbols('x_1 x_2')
		y1,y2 = symbols('y_1 y_2')
		a1,a2, D,d, f = symbols('a1 a2 D d f')

		# Define symmetry plane normal and orthonormal basis for null space of normal. 
		n  = Matrix([sin(theta)*cos(tau), sin(theta)*sin(tau),  cos(theta)]) # n: Symmetry plane normal
		b1 = Matrix([cos(theta)*cos(tau), cos(theta)*sin(tau), -sin(theta)]) # b1 @ n = 0 
		b2 = Matrix([-sin(tau), cos(tau), 0])                                # b2 @ n = b1 @ b2 = 0
		point_on_plane = -d * n # plane = [nx,ny,nz, d]
		midpoint = point_on_plane + a1*b1 + a2*b2

		# Points P, Q in 3D are parameterized as a point on the plane +/- a scalar multiple of n
		P = midpoint + D * n
		Q = midpoint - D * n

		# Perspective image of 3D points P, Q
		p = f * P[:2, :] / P[2]
		q = f * Q[:2, :] / Q[2]

		# Residuals are error between observed image location and location of p,q in image
		residual_p = Matrix([x1, y1]) - p
		residual_q = Matrix([x2, y2]) - q
		residuals = Matrix.vstack(residual_p, residual_q) # symbolic residuals
		jac = residuals.jacobian(Matrix([a1, a2, D, theta, tau])) # symbolic jacobian

		# Convert symbolic equations to numpy via sympy's lambdify
		self._block_jac = lambdify((theta, tau, a1, a2, d, D, f), jac, "numpy")
		self._block_residuals = lambdify((theta, tau, a1, a2, d, D, f, x1, y1, x2, y2), residuals, "numpy")


	def _extract_params(self, params):
		slant, tilt = params[-2:]
		x = params[:-2].reshape(-1,3)
		alpha1 = x[:, 0]
		alpha2 = x[:, 1]
		D = x[:, 2]
		return slant, tilt, alpha1, alpha2, D


	def forward(self, params):
		slant, tilt, alpha1, alpha2, D = self._extract_params(params)
		b1 = np.array([np.cos(slant)*np.cos(tilt), np.cos(slant)*np.sin(tilt), -np.sin(slant)])
		b2 = np.array([-np.sin(tilt), np.cos(tilt), 0])
		n =  np.array([np.sin(slant)*np.cos(tilt), np.sin(slant)*np.sin(tilt),  np.cos(slant)])
		B = np.vstack([b1,b2]).T
		alpha = np.vstack((alpha1, alpha2)).T
		midpoints = alpha @ B.T - self.d * n
		v = np.outer(D, n)
		PQ = np.zeros((2 * len(midpoints), 3))
		PQ[0::2] = midpoints + v
		PQ[1::2] = midpoints - v
		return PQ


	def residuals(self, params):
		slant, tilt, alpha1, alpha2, D = self._extract_params(params)
		residuals = self._block_residuals(
			slant, 
			tilt, 
			alpha1, 
			alpha2, 
			self.d, 
			D, 
			self.focal_length, 
			self.x1,
			self.y1,
			self.x2,
			self.y2).squeeze().T.ravel()
		return residuals


	def jacobian(self, params):
		# Extract Params
		slant, tilt, alpha1, alpha2, D = self._extract_params(params)

		# Compute all non-zero derivatives [block_jac has size (n_cors, 4, 5)]
		block_jac = self._block_jac(slant, tilt, alpha1, alpha2, self.d, D, self.focal_length).transpose(2,0,1)

		# Convert (n_cors, 4, 5) dense array to desired (4 * n_cors, 3*n_cors + 2) sparse array
		A = block_diag(coo_array(block_jac[..., :3]), format='csr')
		B = block_jac[..., 3:].reshape(-1,2)
		full_jac = hstack([A,B], format='csr')
		return full_jac


def persp_to_3D_iterative(persp:NDArray, cors:NDArray, focal_length:float=1.0, sym_plane_intercept:float=1.0):
	obj = to3dIterObject(persp, cors, focal_length, sym_plane_intercept)
	opt_out = least_squares(
		fun=obj.residuals, 
		x0=x0, 
		jac=obj.jacobian, 
		loss='linear',
		verbose = 2,
		method='trf'
	)
	if opt_out.success: 
		xyz = obj.forward(opt_out.x)
		return xyz, obj.cors
	else:
		raise ValueError("Optimization Failed")



def persp_to_3D(
	image:NDArray, 
	cors:NDArray, 
	how:list[Literal["closed_form", "iterative"]], 
	focal_length:float=1.0,
	sym_plane_intercept:float=1.0,
	verbose:int=2,
	max_nfev:int=100,
	x0: NDArray | None = None):
	"""
	Convenience function for converting perspective image to 3D point cloud 
	via symmetry + perspective. Iterative is slower but generally produces
	lower reprojection error.
	"""
	assert how in ["closed_form", "iterative"]
	VP = get_vanishing_point(image, cors, focal_length)

	if how == "closed_form":
		xyz_new, cors_new = persp_to_3D_closed_form(image, cors, VP, sym_plane_intercept)
		return xyz_new, cors_new
	else:
		if x0 is None:
			x,n,BASIS = _get_minimizing_params(image, cors, VP, sym_plane_intercept)
			x0 = np.hstack((x[:, :3].ravel(), np.radians(VP.slant), np.radians(VP.tilt)))
		obj = to3dIterObject(image, cors, focal_length, sym_plane_intercept)
		opt_out = least_squares(
			fun=obj.residuals, 
			x0=x0, 
			jac=obj.jacobian, 
			loss='linear', # 'soft_l1' ...
			verbose = verbose,
			method='trf',
			max_nfev= max_nfev,
			x_scale = 'jac'
		)
		xyz_new = obj.forward(opt_out.x)
		return xyz_new, obj.cors




if __name__ == "__main__":
	engine = get_postgres_engine()
	label = "bench_0004"
	xyz, cors1 = load_data_single(label, engine, ['xyz', 'cors1'])
	st = SimilarityTransform.from_sql(label, "raw2canonical", engine)
	st2 = SimilarityTransform(123,60,231,0,0,-12,1)
	xyzR = st2.apply(st.apply(xyz))
	focal_length = 1
	persp = get_persp_image(xyzR, focal_length)
	persp += np.min(persp.max(0) -persp.min(0)) * .01 * np.random.randn(*persp.shape)
	sym_plane_intercept = 4.23

	xyzA, cors1a = persp_to_3D(
		image=persp, 
		cors=cors1, 
		how='closed_form', 
		focal_length=focal_length, 
		sym_plane_intercept=sym_plane_intercept)


	xyzB, cors1b = persp_to_3D(
		image=persp, 
		cors=cors1, 
		how='iterative', 
		focal_length=focal_length, 
		sym_plane_intercept=sym_plane_intercept, 
		max_nfev=150)


	imgA = get_persp_image(xyzA, focal_length)
	imgB = get_persp_image(xyzB, focal_length)
	SSE_iter = np.sum(np.square(imgB - persp[cors1].reshape(-1,2)))
	SSE_cf = np.sum(np.square(imgA - persp[cors1].reshape(-1,2)))
	print(f"ReprojectionError(closed_form) = {SSE_cf:.04f}\nReprojectionError(iter) = {SSE_iter:.04f}")



