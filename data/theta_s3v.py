"""
Simple image correction strategy & subsequent computation of theta_s3v.
-------------------------------
In an orthographic image of a symmetrical object, symmetry line segments are parallel. 
We have a perspective image of a symmetrical object, but still want to apply compactness via 
the one parameter family somehow. So, the approach taken is to take a perspective image, and 
(1) modify it in a least square manner so that all symmetry line segments are parallel
(2) generate a one parameter family based on this corrected image 
(3) select the member of the one parameter family whose convex hull has 
	volume / surface_area^3 (S3V) maximal and return the slant of its symmetry plane. 
"""
import numpy as np 
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from image_to_3D import one_parameter_family


def get_corrected_image(persp, cors1):
	"""
	Least squares correction for a single set of symmetry correspondences. 
	---------------------------
	A line can be expressed as ax + by + c = 0. 
	---------------------------
	Optimization problem expressed as: 
		   argmin   sum_{i=1}^{n}  ||P_i @ alpha + c_i * 1||^2
		alpha, c_i
		    s.t. ||alpha|| = 1
	---------------------------
	Where:
	(1) alpha = (a,b)^T
	(2) P_i = [[x_i , y_i ], 
			   [x_i', y_i']] (these are the positions of symmetry cors in image)
	(3) i indexes the number of symmetry correspondences in image 
	(4) 1 is a (2 x 1) array of ones. 
	(5) c_i is the scalar intercept in ax + by + c = 0
	---------------------------
	Taking Lagrangian and derivatives, find that 
	(1) c_i = -p_bar.T @ alpha, where p_bar = [(x_i + x_i') / 2, (y_i + y_i') / 2]^T
	(2) Then cost function components become || A_i @ alpha||^2 s.t. ||alpha|| = 1
		where A_i = P_i - 1 @ p_bar^T (A_i is a centered version of P_i)
	(3) sum_{i=1}^{n}  ||A_i @ alpha||^2 = alpha.T @ [sum_{i=1}^{n} A_i.T @ A_i ] @ alpha 
										 = alpha.T @ B @ alpha
	(4) Therefore alpha is the eigenvector associated with the smallest eigenvalue of B. 
	(5) dists from ax + by + c_i given by d_i = P_i @ alpha + outer(c_i, ones(2))
	(6) Orthogonal projection of points in image onto  ax + by + c_i is then: 
		P_i - d_i * alpha (This is the corrected image)
	"""
	
	# Compute orientation of line segments in corrected image 
	P = persp[cors1]
	p_bar = P.mean(axis=1)
	A = P - p_bar[:, None, :]
	B = np.sum(A.mT @ A, axis=0)
	vals, vecs = np.linalg.eigh(B)
	alpha = vecs[:, 0]

	# Generate corrected image based on best orientation
	c = - p_bar @ alpha
	d = P @ alpha + np.outer(c, np.ones(2)) # computes ax + by + c 
	shifts = -np.outer(d, alpha)
	corrected_img = P.reshape(-1,2) + shifts

	# Create new correspondences
	cors1_new = np.arange(len(corrected_img)).reshape(-1,2)
	return corrected_img, cors1_new



def get_log_compactness(xyz, n=2, m=3):
	"""
	Unscaled version of compactness
	"""
	ch = ConvexHull(xyz)
	return n * np.log(ch.volume) - m * np.log(ch.area)


def get_theta_s3v(img, cors):
	corrected, cors_new = get_corrected_image(img, cors)
	f = lambda slant: -get_log_compactness(one_parameter_family(slant, corrected, cors_new), n=1, m=3) 
	opt_out = minimize(
		fun=f, 
		x0 = 45,
		bounds = [(1,89)],
		method='Nelder-Mead')
	if opt_out.success: 
		return opt_out.x.item()
	else:
		raise ValueError("Optimization Failed")



if __name__ == "__main__":
	import pandas as pd 
	from utils import get_postgres_engine, SimilarityTransform, load_data_single

	engine = get_postgres_engine()
	label = "bed_0262"
	xyz, cors1 = load_data_single(label, engine, ['xyz', 'cors1'])
	st = SimilarityTransform.from_sql(label, "raw2canonical", engine)
	st2 = SimilarityTransform(123,75,231,0,0,-4,1)
	xyzR = st2.apply(st.apply(xyz))
	img = xyzR[:, :2] / xyzR[:, 2][:, None]
	theta_s3v = get_theta_s3v(img, cors1)
	print(f"Object {label}: theta_s3v = {theta_s3v:.03f} Degrees")





