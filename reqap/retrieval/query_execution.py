import re
import json
import glob
import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Any
from loguru import logger

from reqap.library.library import flatten_nested_df
from reqap.library.temporal import date_to_datetime, datetime_to_timestamp


TABLE_RENAMING = {
    "annual_doctor_appointment": "doctor_appointment",
    "oneoff_event": "personal_milestone"
}

class QueryExecution:
    STR_EVENTS_TABLES = {
        "annual_celebration": """CREATE TABLE IF NOT EXISTS annual_celebration (id BIGINT, start_date VARCHAR, start_time VARCHAR, end_date VARCHAR, end_time VARCHAR, event VARCHAR, participants STRUCT("name" VARCHAR, "type" VARCHAR)[], location VARCHAR, cuisine VARCHAR, restaurant VARCHAR, entity_name VARCHAR, entity_start_date VARCHAR, entity_end_date VARCHAR, entity_type VARCHAR, entity_birth_date VARCHAR);""",
        "doctor_appointment": """CREATE TABLE IF NOT EXISTS doctor_appointment (id BIGINT, start_date VARCHAR, start_time VARCHAR, end_date VARCHAR, end_time VARCHAR, event VARCHAR, entity_name VARCHAR, entity_type VARCHAR, entity_start_date VARCHAR, entity_end_date VARCHAR, entity_birth_date VARCHAR);""",
        "meet_up": """CREATE TABLE IF NOT EXISTS meet_up (id BIGINT, start_date VARCHAR, start_time VARCHAR, end_date VARCHAR, end_time VARCHAR, event VARCHAR, participants STRUCT("name" VARCHAR, "type" VARCHAR)[], location VARCHAR, cuisine VARCHAR, restaurant VARCHAR);""",
        "personal_milestone": """CREATE TABLE IF NOT EXISTS personal_milestone (id BIGINT, start_date VARCHAR, start_time VARCHAR, end_date VARCHAR, end_time VARCHAR, event VARCHAR, name VARCHAR, country VARCHAR, city VARCHAR, company_name VARCHAR, job_title VARCHAR, type VARCHAR);""",
        "trip": """CREATE TABLE IF NOT EXISTS trip (id BIGINT, start_date VARCHAR, start_time VARCHAR, end_date VARCHAR, end_time VARCHAR, event VARCHAR, destination VARCHAR, participants STRUCT("name" VARCHAR, "type" VARCHAR)[], start_day VARCHAR, end_day VARCHAR, traveling_by VARCHAR, notable_events VARCHAR, features VARCHAR[]);""",
        "trip_highlight": """CREATE TABLE IF NOT EXISTS trip_highlight (id BIGINT, start_date VARCHAR, start_time VARCHAR, end_date VARCHAR, end_time VARCHAR, event VARCHAR, attraction VARCHAR, type VARCHAR, participants STRUCT("name" VARCHAR, "type" VARCHAR)[], during_trip_to VARCHAR, restaurant VARCHAR, cuisine VARCHAR, food VARCHAR, location VARCHAR);""", # , team VARCHAR, sport VARCHAR, event_name VARCHAR, cafe VARCHAR, drink VARCHAR, museum VARCHAR, hotel VARCHAR, features VARCHAR, bands VARCHAR
    }

    OBS_EVENTS_TABLES = {
        "calendar": """CREATE TABLE IF NOT EXISTS calendar (id BIGINT, start_date VARCHAR, start_time VARCHAR, end_date VARCHAR, end_time VARCHAR, summary VARCHAR, location VARCHAR, description VARCHAR);""",
        "mail": """CREATE TABLE IF NOT EXISTS mail (id BIGINT, start_date VARCHAR, start_time VARCHAR, end_date VARCHAR NULL, end_time VARCHAR NULL, subject VARCHAR, timestamp VARCHAR, sender VARCHAR, recipient VARCHAR, text VARCHAR);""",
        "movie_stream": """CREATE TABLE IF NOT EXISTS movie_stream (id BIGINT, start_date VARCHAR, start_time VARCHAR, end_date VARCHAR, end_time VARCHAR, event VARCHAR, duration DOUBLE, duration_unit VARCHAR, country VARCHAR, stream_end_time VARCHAR, movie_title VARCHAR, stream_full_title VARCHAR, watching_continuation VARCHAR);""",
        "music_stream": """CREATE TABLE IF NOT EXISTS music_stream (id BIGINT, start_date VARCHAR, start_time VARCHAR, end_date VARCHAR, end_time VARCHAR, event VARCHAR, duration DOUBLE, duration_unit VARCHAR, song_name VARCHAR, song_artist VARCHAR[]);""",
        "online_purchase": """CREATE TABLE IF NOT EXISTS online_purchase (id BIGINT, start_date VARCHAR, start_time VARCHAR, end_date VARCHAR, end_time VARCHAR, event VARCHAR, "order" VARCHAR, product_quantity BIGINT, product VARCHAR, price VARCHAR);""",
        "social_media": """CREATE TABLE IF NOT EXISTS social_media (id BIGINT, start_date VARCHAR, start_time VARCHAR, end_date VARCHAR NULL, end_time VARCHAR NULL, text VARCHAR);""",
        "tvseries_stream": """CREATE TABLE IF NOT EXISTS tvseries_stream (id BIGINT, start_date VARCHAR, start_time VARCHAR, end_date VARCHAR, end_time VARCHAR, event VARCHAR, duration DOUBLE, country VARCHAR, stream_end_time VARCHAR, tvseries_title VARCHAR, season_name VARCHAR, episode_name VARCHAR, episode_number BIGINT, tvseries_season BIGINT, watching_continuation VARCHAR);""",
        "workout": """CREATE TABLE IF NOT EXISTS workout (id BIGINT, start_date VARCHAR, start_time VARCHAR, end_date VARCHAR, end_time VARCHAR, event VARCHAR, workout_type VARCHAR, duration BIGINT, duration_unit VARCHAR, minimum_heart_rate BIGINT, maximum_heart_rate BIGINT, average_heart_rate DOUBLE, minimum_speed DOUBLE, maximum_speed DOUBLE, average_speed DOUBLE, speed_unit VARCHAR, distance DOUBLE, distance_unit VARCHAR);""",
    }

    def __init__(self, obs_events_csv_path: Optional[str]=None, str_events_csv_path: Optional[str]=None):
        # load observable events (if any)
        if obs_events_csv_path is not None:
            obs_csv_paths = [p for p in glob.glob(obs_events_csv_path)]
            obs_df = pd.concat([pd.read_csv(f, converters={"event_data": json.loads}) for f in obs_csv_paths], ignore_index=True)
        else:
            obs_csv_paths = []
            obs_df = None
        
        # load structured events (if any)
        if str_events_csv_path is not None:
            str_csv_paths = [p for p in glob.glob(str_events_csv_path)]
            str_df = pd.concat([pd.read_csv(f, converters={"event_data": json.loads}) for f in str_csv_paths], ignore_index=True)
        else:
            str_csv_paths = []
            str_df = None
        
        self.db = duckdb.connect(database=':memory:', read_only=False)
        self.init_db(str_df, obs_df)
        logger.info(f"Loaded {len(obs_csv_paths)} files with observable events and {len(str_csv_paths)} files with structured events.")

    def init_db(self, str_df: Optional[pd.DataFrame]=None, obs_df: Optional[pd.DataFrame]=None) -> None:
        """
        Initiate the DuckDB with data.
        """
        # drop previous tables
        df = self.db.execute("SELECT table_name, table_type, * FROM information_schema.tables").df()
        print("df", df)
        tables = self.db.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' AND table_type = 'BASE TABLE';").fetchall()
        for (table_name,) in tables:
            self.db.execute(f"DROP TABLE IF EXISTS {table_name};")
        views = self.db.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' AND table_type = 'VIEW';").fetchall()
        for (table_name,) in views:
            self.db.execute(f"DROP VIEW IF EXISTS {table_name};")
        # for sql_query in self.STR_EVENTS_TABLES.values():
        #     self.db.execute(sql_query)

        # create tables for structured data
        if not str_df is None:
            self.db.register("all_events", str_df)   # general table with all events
            df = str_df.groupby(["event_type"])
            for event_type, df_group in df:
                df_group = flatten_nested_df(df_group)
                df_group.columns = df_group.columns.str.replace('event_data.', '')
                df_group.columns = df_group.columns.str.replace('.', '_')
                event_type = event_type[0]
                if event_type in TABLE_RENAMING:
                    event_type = TABLE_RENAMING[event_type]
                self.db.register(event_type, df_group)
        
        # create tables for observable data
        if not obs_df is None:
            df = obs_df.groupby(["event_type"])
            for event_type, df_group in df:
                event_type = event_type[0]
                # skip structured sources only in case str_df is set
                if not event_type in ["calendar", "mail", "social_media"] and str_df is not None:
                    continue
                df_group = flatten_nested_df(df_group)
                df_group.columns = df_group.columns.str.replace('event_data.', '')
                df_group.columns = df_group.columns.str.replace('.', '_')
                self.db.register(event_type, df_group)
            for table_name, sql_query in self.OBS_EVENTS_TABLES.items():
                self.db.execute(sql_query)

    def run_sql_query(self, sql_query: str, reference_date: str=None) -> pd.DataFrame:
        """
        Run the SQL query via DuckDB.
        The given reference_date (in YYYY-MM-DD format) will be used to
        replace CURRENT_DATE in the queries.
        """
        # apply reference date (if set)
        sql_query = self.adjust_sql_query_reference_data(sql_query, reference_date)

        # dedicated mechanism to break ties for argmin/argmax
        sql_query = sql_query.strip() 
        if sql_query.endswith("LIMIT 1;") or sql_query.endswith("LIMIT 1"):
            # "case": query ends with "LIMIT 1;"
            new_sql_query = sql_query.replace("LIMIT 1;", ";")
            # "case": query ends with "LIMIT 1"
            if new_sql_query.endswith("LIMIT 1"):
                new_sql_query = new_sql_query[:-7]
            df = self.run_sql_query(new_sql_query, reference_date)
            if self.is_none_answer(df):
                return df
            sort_values = df.iloc[:, [-1]]
            reference_value = sort_values.iloc[0, 0]  # MAX/MIN value in column
            df = df[df.iloc[:, -1] == reference_value]
            return df

        # run query
        try:
            res = self.db.query(sql_query).df()
        except Exception as e:
            error_message = f"Error with SQL query: `{sql_query}`: {e}."
            logger.error(error_message)
            raise self.SQLError(error_message)
        return res
    
    @staticmethod
    def adjust_sql_query_reference_data(sql_query: str, reference_date: str=None) -> pd.DataFrame:
        if reference_date is None:
            return sql_query
        date_to_datetime(reference_date)  # check if format valid
        sql_query = sql_query.replace("CURRENT_DATE", f"'{reference_date}'::DATE")
        return sql_query
    
    @staticmethod
    def parse_query_result(df: pd.DataFrame | None, json_serializable: bool=False) -> Any:
        """
        Function which extracts the answer / list of answers / answer tuples from
        the resulting df. This is a different implementation than in the dataset
        creation repo.
        """
        def _normalize(value: Any, json_serializable: bool):
            if type(value) == pd.Timestamp:
                if json_serializable:
                    return datetime_to_timestamp(value.date())
                else:
                    return value.date()
            elif type(value) == datetime:
                if json_serializable:
                    return datetime_to_timestamp(value)
            elif type(value) in [timedelta, pd.Timedelta]:
                if json_serializable:
                    return value.days
            elif type(value) == str:
                return value.replace("\"", "")
            elif type(value) == float:
                value = round(value, 2)
            elif type(value) is np.ndarray:
                value = value.tolist()
            return value
        def _apply_normalization(nested_list: List, json_serializable: bool):
            for i, element in enumerate(nested_list):
                if isinstance(element, list):
                    _apply_normalization(element, json_serializable)
                else:
                    nested_list[i] = _normalize(element, json_serializable)
        if df is None or df.empty:
            return None
        with pd.option_context('future.no_silent_downcasting', True):
            df = df.fillna(0)
        answers = df.values.tolist()
        answers = [a[0] if type(a) is list and len(a) == 1 else a for a in answers]
        _apply_normalization(answers, json_serializable)
        return answers

    @staticmethod
    def is_simple_query(query: str) -> bool:
        """ Check whether the provided query is "simple". """
        # check whole query
        BLACKLIST = ["WITH", "JOIN"]
        BLACKLIST_BEFORE_WHERE = ["<", ">"]
        BLACKLIST_AFTER_WHERE = []  # ["::DATE"]
        for w in BLACKLIST:
            if w in query:
                # logger.debug(f"Query `{query}` dropped because of {w}.")
                return False
        # check final part of query (after WHERE clause)
        if "WHERE" in query:
            before_where, after_where = query.rsplit("WHERE", 1)
            for w in BLACKLIST_BEFORE_WHERE:
                if w in before_where:
                    # logger.debug(f"Query `{query}` dropped because of `{w}` before WHERE clause.")
                    return False
            for w in BLACKLIST_AFTER_WHERE:
                if w in after_where:
                    # logger.debug(f"Query `{query}` dropped because of `{w}` after WHERE clause.")
                    return False
        return True
    
    def derive_retrieval_query(sql_query: str, retain_where_clause: bool=False, drop_dates_from_where_clause: bool=True) -> str:
        """ For the simple query, derive the query that searches for the relevant IDs that are aggregated. """
        EXPRESSIONS_BLACKLIST = ["date", "time", "<", ">"]  # expressions we do not want to capture in RETRIEVE
        # SELECT_PATTERN = r"SELECT\s+(.*?)\s+FROM"
        FROM_PATTERN = r"FROM\s+(\w+)(?:\s|$)"
        WHERE_PATTERN = r"WHERE\s+(.*?)(?:\s+GROUP BY|\s+ORDER BY|$)"

        sql_query = sql_query.replace(";", "")  # drop delimiter

        # extract clauses: assumption is that there are no JOINs or WITH clauses
        # extract_clause = re.search(SELECT_PATTERN, sql_query, re.S).group(1).strip()  # currently not in use
        from_clause_match = re.search(FROM_PATTERN, sql_query, re.S)
        from_clause = from_clause_match.group(1).strip() if from_clause_match else "1"
        if not from_clause_match:
            logger.warning(f"No FROM clause detected for sql_query={sql_query}")
        where_clause_match = re.search(WHERE_PATTERN, sql_query, re.S)
        where_clause = where_clause_match.group(1).strip() if where_clause_match else "1"

        # remove any additional columns from output
        query = re.sub(r"SELECT\s+.*?\s+FROM", "SELECT id FROM", sql_query)
        query = re.sub(r"ORDER BY.*", ";", query)
        
        # drop dates from WHERE clause
        if drop_dates_from_where_clause:
            where_clauses = re.split(r'\s+(AND|OR)\s+', where_clause)
            where_clauses = [clause.strip() if not any(exp in clause.lower() for exp in EXPRESSIONS_BLACKLIST) else "1" for clause in where_clauses]
            where_clause = " ".join(where_clauses).strip()

        # drop WHERE clause
        if not retain_where_clause:
            where_clause = "1"

        # derive retrieve SQL query
        retrieval_sql_query = f"SELECT id FROM {from_clause.strip()} WHERE {where_clause.strip()};"
        return retrieval_sql_query
    
    @staticmethod
    def derive_select_query(simple_query: str) -> str:
        """ For the simple query, derive the query that searches for the relevant IDs that are aggregated. """
        # drop any FROM clauses in the SELECT part (as used in e.g., "EPOCH FROM DURATION")
        first_part, second_part = simple_query.split("FROM", 1)
        # first_part, second_part = simple_query.rsplit("FROM", 1)
        first_part = first_part.replace("FROM", "TMP")  
        simple_query = first_part + " FROM " + second_part
        # remove any additional columns from output
        query = re.sub(r"ORDER BY.*", ";", simple_query)
        while "  " in query:
            query = query.replace("  ", " ").strip()
        if not query[-1] == ";":
            query = query + ";"
        return query
    
    @staticmethod
    def derive_columns_from_sql_query(sql_query: str):
        """
        Extracts column names from all clauses of the SQL query.
        NOT IN USE DUE TO LOTS OF NOISE.
        """
        BLACKLIST = {"*"}
        # normalize
        query = sql_query.lower()
        query = re.sub(r"\s+", " ", query)
        # patterns
        clause_patterns = [
            r"select (.+?) from",                  
            r"where (.+?)( group by| order by| limit|$)",
            r"group by (.+?)( order by| limit|$)",
            r"order by (.+?)( limit|$)"
        ]
        # derive all columns
        columns = set()
        aliases = set()
        for pattern in clause_patterns:
            match = re.search(pattern, query)
            if match:
                clause_content = match.group(1)
                for item in clause_content.split(","):
                    item = item.strip()
                    # remove aliases (e.g., "column AS alias" -> "column")
                    if " as " in item:
                        item, alias = item.split(" as ", 1)
                        item = item.strip()
                        alias = alias.strip()
                        aliases.add(alias)
                    # remove function wrappers (e.g., "MAX(duration)" -> "duration")
                    item = re.sub(r"\w+\((.+?)\)", r"\1", item)
                    # remove additional operators and literals (e.g., "column = value" -> "column")
                    item = re.split(r"[<>=!:;]+|\s+", item)[0]
                    columns.add(item)
        columns = columns - BLACKLIST
        columns = columns - aliases
        return columns
    
    class SQLError(Exception):
        pass

    class ValueDoesNotExist(Exception):
        pass

        