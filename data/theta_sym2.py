import numpy as np 
from numpy.typing import NDArray
from utils import get_postgres_engine, to_homogeneous, SimilarityTransform, get_persp_image, load_data_single
import pandas as pd
from theta_persp import get_vanishing_point, VanishingPoint
import autograd.numpy as anp
import pymanopt
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers import ConjugateGradient


def get_steifel_opt_inputs(img, cors1, cors2, focal_length):
	#Vanishing Points
	VP1 = get_vanishing_point(img, cors1, focal_length)
	VP2 = get_vanishing_point(img, cors2, focal_length)

	# Get Starting Point for Optimization
	Q = np.vstack((VP1.xyz, VP2.xyz)).T
	X0, R0 = np.linalg.qr(Q)

	# Get A,B
	imgh = to_homogeneous(img)
	A = np.cross( imgh[cors1[:, 0]], imgh[cors1[:, 1]] ) 
	B = np.cross( imgh[cors2[:, 0]], imgh[cors2[:, 1]] ) 
	return A, B, X0


def run_steifel_opt(A, B, X0, focal_length, max_iterations=1000):
	"""
	This particular optimization exploits the assumption that the two sets of 
	symmetry line segments are orthogonal in 3D. Uses the same ideas as the
	function get_initial_vp_estimate in theta_persp.py but does so for both 
	homogeneous vanishing points at once. 
	---------------------------------------------------
	Optimization problem becomes 
	     argmin   ||A @ vph1||^2 + ||B @ vph2||^2
	   vph1, vph2
	   s.t. ||vph1|| = 1, ||vph2|| = 1, dot(vph1, vph2) = 0
	---------------------------------------------------
	Where 
	(1) A is an (n x 3) matrix where n is the number of symmetry line segments 
		associated with the primary symmetry plane and each row corresponds to
		coefficients (a,b,c) describing a symmetry line segment (ax + by + c = 0) 
	(2) B is the same as A but for the second set of symmetry correspondences. 
	---------------------------------------------------
	There are situations where angle S associated with the second set of correspondences 
	is large, but angle S associated with the first set of correspondences is small. 
	In this scenario, the second vanishing point can be estimated accurately and the 
	orthogonality constraint leads to more accurate estimation of the first vanishing point. 
	"""
	n = A.shape[1]
	manifold = Stiefel(n, 2)

	@pymanopt.function.autograd(manifold)
	def cost(X):
		x = X[:, 0]
		y = X[:, 1]
		return anp.sum((A @ x) ** 2) + anp.sum((B @ y) ** 2)

	problem = pymanopt.Problem(
		manifold=manifold,
		cost=cost
	)

	optimizer = ConjugateGradient(max_iterations=max_iterations, verbosity=0)
	result = optimizer.run(problem, initial_point=X0)

	# Put optimization results into VanishingPoint object. 
	vp1 = VanishingPoint(vp=result.point[:, 0], focal_length=focal_length)
	vp2 = VanishingPoint(vp=result.point[:, 1], focal_length=focal_length)
	return vp1, vp2


if __name__ == "__main__":
	engine = get_postgres_engine()
	label = "bench_0004"
	xyz, cors1, cors2 = load_data_single(label, engine, ['xyz', 'cors1', 'cors2'])
	st = SimilarityTransform.from_sql(label, "raw2canonical", engine)
	st2 = SimilarityTransform(123,60,231,0,0,-4,1)
	xyzR = st2.apply(st.apply(xyz))
	focal_length = 1
	img = get_persp_image(xyzR, focal_length = focal_length)


	A,B,X0 = get_steifel_opt_inputs(img, cors1, cors2, focal_length)
	VP1, VP2 = run_steifel_opt(A, B, X0, focal_length)
	print(f"Object {label}: theta_persp = {VP1.slant:.03f} Degrees")












