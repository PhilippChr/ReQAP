import csv
import pandas as pd
from io import TextIOWrapper
from typing import List

CSV_SETTINGS = {"delimiter": ",", "quotechar": '"', "quoting": csv.QUOTE_MINIMAL}


def initialize_csv_writer(csv_output_file: TextIOWrapper):
	"""
	Initialize the .csv writer.
	Sets the CSV settings and returns the writer.
	"""
	csv_writer = csv.writer(csv_output_file, **CSV_SETTINGS)
	return csv_writer


def initialize_csv_file(csv_output_file: TextIOWrapper, csv_header: List[str]):
	"""
	Initialize the .csv output file.
	Writes out the csv_header.
	Returns the csv writer.
	"""
	csv_writer = csv.writer(csv_output_file, **CSV_SETTINGS)
	csv_writer.writerow(csv_header)
	return csv_writer
	

def initialize_csv_reader(csv_input_file: TextIOWrapper,):
	"""
	Returns a .csv reader for the input file.
	"""
	reader = csv.DictReader(csv_input_file)
	return reader


def read_csv_content_into_pandas(csv_input_file: TextIOWrapper):
	"""
	Returns a list with the non-NaN values of the provided
	column_name in the csv_input_file.
	"""
	csv_content = pd.read_csv(csv_input_file)
	return csv_content
