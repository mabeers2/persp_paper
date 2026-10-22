"""
This file loads subject data into database 
"""
import pandas as pd 
from pathlib import Path
from utils import get_postgres_engine, get_api_keys
from sqlalchemy import text


def trim_path(p: str):
	return (p.split("/")[-1]).split(".")[0]


def get_category(s: str):
	if s.startswith("rp"):
		return "polyhedron"
	else:
		return "real"


def load_subject_data(data_directory: Path):
	dfs = []
	for path in data_directory.glob("*.csv"):
		dfs.append(pd.read_csv(path))
	df = pd.concat(dfs, axis=0, ignore_index=True)
	return df


def clean_subject_data(df: pd.DataFrame, sub_to_anon_dct: dict[str, str]):
	df = df.drop(columns = ['set', 'condition', 'trial'])
	df['label'] = df.path.apply(trim_path)
	df['obj_type'] = df.label.apply(get_category)
	df['subject'] = df.subject.replace(sub_to_anon_dct)
	columns = ['subject', 'obj_type', 'label', 'dist', 'veridical_sa', 'recovered_sa']
	df = df[columns]
	return df


if __name__ == "__main__":

	# Load and Clean Data
	sub_to_anon_dct = get_api_keys()['sub_keys'] #dict mapping identifying name in raw data to anonymous label e.g. {"mark":"S1", ...}
	directory = Path.cwd() / "subject_data_raw"
	df = load_subject_data(directory)
	df = clean_subject_data(df, sub_to_anon_dct)

	# Send to database
	engine = get_postgres_engine()
	df.to_sql(
		name="subject_data", 
		con=engine, 
		index=False, 
		if_exists="replace"
	)

	# Create index
	with engine.begin() as connection:
		connection.execute(text("ALTER TABLE subject_data ADD PRIMARY KEY (subject, obj_type,label,dist);"))
		print("created index")






