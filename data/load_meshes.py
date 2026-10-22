"""
This file creates xyz_raw and triangles tables in Postgres Database. 
"""
import pandas as pd
import numpy as np
from pathlib import Path
from utils import get_postgres_engine
from sqlalchemy import text
from tqdm import tqdm
import trimesh


def add_label_and_index_columns(df:pd.DataFrame, label:str):
	n = len(df)
	labels = pd.DataFrame({"label":[label]*n, 'idx':list(range(n))})
	return pd.concat([labels, df], axis=1)


def build_tables(path_to_modelnet40:Path, labels:list):
	xyz_all = pd.DataFrame(columns = ['label', 'idx', 'x', 'y', 'z'])
	ijk_all = pd.DataFrame(columns = ['label', 'idx', 'i', 'j', 'k'])
	for label in tqdm(labels):

		# Load Mesh
		m = path_to_modelnet40.rglob(f'{label}.off')
		mesh = trimesh.load_mesh(next(m))

		# Extract coordinates & place in dataframe
		xyz = pd.DataFrame(np.asarray(mesh.vertices), columns=['x','y', 'z'])
		xyz = add_label_and_index_columns(xyz, label)
		xyz_all = pd.concat([xyz_all, xyz], ignore_index=True)

		# Extract triangles & place in dataframe
		ijk = pd.DataFrame(np.array(mesh.faces), columns = ['i','j','k'])
		ijk = add_label_and_index_columns(ijk, label)
		ijk_all = pd.concat([ijk_all, ijk], ignore_index=True)

	# reset dtypes
	xyz_all = xyz_all.astype(xyz.dtypes.to_dict())
	ijk_all = ijk_all.astype(ijk.dtypes.to_dict())
	return xyz_all, ijk_all


def send_tables_to_db(xyz, ijk, con):
	print('Starting xyz table')
	xyz.to_sql(
		name = "xyz_raw", 
		index=False,
		method='multi',
		chunksize=5000,
		if_exists='replace',
		con=con)

	print('Starting ijk table')
	ijk.to_sql(
		name = "triangles", 
		index=False,
		method='multi',
		chunksize=5000,
		if_exists='replace',
		con=con)

	print("Creating index")
	with con.begin() as connection:
		connection.execute(text("ALTER TABLE xyz_raw ADD PRIMARY KEY (label, idx);"))
		connection.execute(text("ALTER TABLE triangles ADD PRIMARY KEY (label, idx);"))
		print("created indices")



if __name__ == "__main__":

	path_to_modelnet40 = Path("./../../../../dropbox/rigidity/modelnet40")
	labels = pd.read_csv("objects.csv").label.tolist()
	xyz, ijk = build_tables(path_to_modelnet40, labels)
	con = get_postgres_engine()
	send_tables_to_db(xyz, ijk, con)





















