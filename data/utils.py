import json 
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass
import numpy as np 
from sqlalchemy import create_engine
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
import pandas as pd
from sqlalchemy.engine.base import Engine
from typing import Literal

def get_max_distance_between_any_two_points(pts):
	hull = ConvexHull(pts)
	hull_pts = hull.points[hull.vertices]
	D = cdist(hull_pts, hull_pts)
	return D.max()



def to_homogeneous(pts: np.ndarray, focal_length: int|float = 1.):
	"""
	Given an ndarray pts of any size, adds a column of ones to the last dimension.
	So if pts has size (3, ) output is size (4, ), if size (N, 2), output is (N,3), 
	If input is (B, N, 2), output is (B,N,3), and so on...
	"""
	ONES = focal_length * np.ones( tuple(i if j < pts.ndim-1 else 1 for j, i in enumerate(pts.shape)) )
	return np.concatenate((pts, ONES), axis=-1)



@dataclass
class SimilarityTransform:
	rot_x:float
	rot_y:float
	rot_z:float
	_translation_x: float
	_translation_y: float
	_translation_z: float
	scale: float = 1.0
	degrees: bool = True
	transform_type: str = ""
	label: str = ""

	@property
	def translation(self):
		return np.array([self._translation_x, self._translation_y, self._translation_z])

	@property
	def rot_mat(self):
		return R.from_euler('ZYX', [self.rot_z, self.rot_y, self.rot_x], degrees=self.degrees)

	def apply(self, xyz):
		return self.rot_mat.apply(self.scale * xyz) + self.translation

	@staticmethod
	def from_sql(label, transform_type, engine):
		df = pd.read_sql(sql = f"""
			SELECT
				*
			FROM 
				transforms 
			WHERE 
				transform = '{transform_type}' AND
				label = '{label}';
			""",
			con = engine)
		ST = SimilarityTransform(
			rot_x = df.rot_x.item(),
			rot_y = df.rot_y.item(),
			rot_z = df.rot_z.item(),
			_translation_x = df._translation_x.item(),
			_translation_y = df._translation_y.item(),
			_translation_z = df._translation_z.item(),
			scale = df.scale.item(),
			transform_type=  transform_type,
			label = label)
		return ST



def get_api_keys(path = 'KEYS.json'):
	"""
	API_KEYS.json just contains all the stuff I don't want to upload to github.
	The file is in the gitignore  
	It has the format: 
	{
		"nlr": my api key for nlr
		"email": my email
		"some other api key": .... 
		etc. 
	}
	"""
	with open(path, 'r') as file:
		API_KEYS = json.load(file)
	return API_KEYS


def get_postgres_engine(path_to_KEYS="KEYS.json"):
	api_keys = get_api_keys(path_to_KEYS)
	username = api_keys['username_psql']
	password = api_keys['password_psql']
	host = "localhost"
	port = "5432"    
	database = "persp"
	connection_string = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"
	engine = create_engine(connection_string)
	return engine


def get_best_fit_plane(pts):
	"""
	Plane defined as ax + by + cz + d = 0
	Returns plane coefficients (a,b,c,d) where they have been selected 
	such that sum of orthogonal squared distances of pts to plane is minimal. 
	Mathematically, solution to below. 
	--------------------------------------
	    argmin     ||P @ abc + d*1||^2   
	   (abc, d)

		such that ||abc|| = 1
	--------------------------------------
	where 
	(1) P is an (n x 3) array of points 
	(2) abc is a unit length normal of the best fit plane (3x1)
	(3) d is the scalar intercept of the plane 
	(4) 1 is an (n x 1) column of ones. 
	--------------------------------------
	Writing out Lagrangian and taking derivatives, find that:
	(1) alpha given by eigenvector associated with smallest eigenvalue of P_c.T @ P_c 
		where P_c is P centered so that column means are all zero. 
	(2) plane intercept d given by - center @ abc 
	"""
	center = pts.mean(axis = 0)
	P = pts - center
	vals, vecs = np.linalg.eigh(P.T @ P)
	abc = vecs[:, 0] # eigh returns eigenvalues in ascending order 
	# abc = np.linalg.svd(P)[2][-1] # If prefer svd. 
	d = - center @ abc
	s = np.sign(abc[np.argmax(np.abs(abc[:3]))])
	return s * np.hstack((abc, d))


def get_persp_image(xyz, focal_length):
	return focal_length * xyz[:, :2] / xyz[:, 2][:, None]


def load_data_single(label:str, engine:Engine, what:list[Literal["xyz", "cors1", "cors2"]]):
	sqls = {
		"xyz":f"SELECT x,y,z FROM xyz_raw WHERE label = '{label}';",
		"cors1":f"SELECT i,j FROM sym_cors WHERE label = '{label}' AND sym_plane = 'primary';",
		"cors2":f"SELECT i,j FROM sym_cors WHERE label = '{label}' AND sym_plane = 'secondary';"
	}
	lst = []
	for key in what:
		lst.append(pd.read_sql(sql=sqls[key], con=engine).to_numpy())
	return lst



if __name__ == "__main__":
	print(5)










