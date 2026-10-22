import numpy as np 
import pandas as pd
from utils import get_postgres_engine, SimilarityTransform
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector


def load_data_single(label, engine):
	xyz = pd.read_sql(sql=f"SELECT x,y,z FROM xyz_raw WHERE label = '{label}';", 
		con=engine).to_numpy()
	ijk = pd.read_sql(sql=f"SELECT i,j,k FROM triangles WHERE label = '{label}';", 
		con=engine).to_numpy()
	cors1 = pd.read_sql(
		sql=f"SELECT i,j FROM sym_cors WHERE label = '{label}' AND sym_plane = 'primary';", 
		con=engine
	).to_numpy()
	cors2 = pd.read_sql(
		sql=f"SELECT i,j FROM sym_cors WHERE label = '{label}' AND sym_plane = 'secondary';", 
		con=engine
	).to_numpy()
	return xyz, ijk, cors1, cors2


def get_triangle(radius = 1):
	angles = np.radians([90,210,330])
	tri_xyz = np.vstack((
		radius * np.cos(angles),
		radius * np.sin(angles), 
		np.zeros(3)
	)).T
	return tri_xyz


def get_visible_vert_filter(xyzR, ijk, eps=1e-8):
	"""
	THIS FUNCTION ASSUMES CAMERA CENTER AT ORIGIN AND PERSPECTIVE PROJECTION. 
	--------------------------------------------------------
	Vertex v is visible if the ray from v to camera center c
	doesn't intersect another triangle in the mesh before it gets to c. 
	--------------------------------------------------------
	This function creates a triangle at c, and for each vertex v:
	(1) Defines direction d = (c-v) / ||c-v||
	(2) Checks if the ray defined as (v + eps*d) + s * d intersects 
		the triangle at c before any other triangle. s is a scalar. 
		If so, the ray from vertex to camera center was unobstructed and 
		the vertex is visible. 
	"""
	tri_xyz = get_triangle()
	n_pts = len(xyzR)
	n_triangles = len(ijk)
	xyzR_and_triangle = np.vstack((xyzR, tri_xyz))
	ijk_and_triangle = np.vstack((ijk, n_pts + np.arange(3)))
	mesh = trimesh.Trimesh(vertices = xyzR_and_triangle, faces = ijk_and_triangle)
	RM = RayMeshIntersector(mesh)
	ray_directions = -xyzR
	ray_directions /= np.linalg.norm(ray_directions, axis=1, keepdims=True)
	ray_origins = xyzR + eps * ray_directions
	index_triangle = RM.intersects_first(
		ray_origins = ray_origins,
		ray_directions = ray_directions	
	)
	return index_triangle == n_triangles



if __name__ == "__main__":
	engine = get_postgres_engine()
	label = 'chair_0384'
	xyz, ijk, cors1, cors2 = load_data_single(label, engine)
	st = SimilarityTransform.from_sql(label, 'raw2canonical', engine)
	st2 = SimilarityTransform(
		rot_x = 45,
		rot_y = 75,
		rot_z = 23,
		_translation_x = 0,
		_translation_y = 0,
		_translation_z = -4,
		scale = 1
	)
	xyzR = st2.apply(st.apply(xyz))
	visible_fltr = get_visible_vert_filter(xyzR, ijk, eps=1e-4)
	mesh = pv.make_tri_mesh(xyzR, ijk)


	# Plot visible points
	import pyvista as pv
	pl = pv.Plotter()
	pl.camera_position = 'xy'
	pl.camera_position = [
	    (0.0, 0.0, 1.0),  # Position
	    (0.0, 0.0, 0.0),  # Focal point
	    (0.0, 1.0, 0.0),  # View up
	]
	translated_mesh = mesh.translate([-x for x in mesh.center], inplace=False)
	c = np.array(mesh.center)
	xyzRc = xyzR - c
	pl.add_mesh(translated_mesh, color="grey", show_edges=True)
	pl.add_points(
	    xyzRc[visible_fltr],
	    style="points_gaussian",
	    render_points_as_spheres=True,
	    point_size=3, 
	    color="tomato",
	    opacity=0.9
	)
	pl.add_points(
	    xyzRc[~visible_fltr],
	    style="points_gaussian",
	    render_points_as_spheres=True,
	    point_size=3, 
	    color="blue",
	    opacity=0.9
	)
	pl.show()


