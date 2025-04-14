import os
from dotenv import load_dotenv
import sqlite3
import pandas as pd

load_dotenv()

db_name = os.getenv("data_db")
table_name = os.getenv("data_db_table")

class SQLHelper:
    def __init__(self, db_name=db_name, data_table_name=table_name):
        try:
            self.db = db_name
            self.data_table = data_table_name
            self.conn = sqlite3.connect(db_name,check_same_thread=False)
            print(f"Connected to database: {db_name}")
        except Exception as e:
            print(f"[ERROR] Could not connect to database: {e}")
            self.conn = None
            self.data_table_status=False

    def check_table_exists(self, table_name=None):
        try:
            table = table_name if table_name else self.data_table

            if isinstance(table, list):
                results = {}
                for t in table:
                    cursor = self.conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
                        (t,)
                    )
                    exists = cursor.fetchone() is not None
                    print(f"Table '{t}' exists: {exists}")
                    results[t] = exists
                return results

            elif isinstance(table, str):
                cursor = self.conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
                    (table,)
                )
                exists = cursor.fetchone() is not None
                print(f"Table '{table}' exists: {exists}")
                return exists

            else:
                raise ValueError(f"Expected table_name as str or list[str], got: {type(table)}")

        except Exception as e:
            print(f"[ERROR] Could not check if table exists: {e}")
            return False


    def check_data_table(self):
        return self.check_table_exists(table_name=self.data_table)

    def create_data_table(self,csv_path):
        try:
            print(f"Writing data to table '{self.data_table}'...")

            if isinstance(csv_path, str):
                df = pd.read_csv(csv_path)
            else:
                raise ValueError("Data must be a file path (str) or a pandas DataFrame")

            df.to_sql(
                self.data_table,
                self.conn,
                if_exists='replace',
                index=False
            )
            print(f"Successfully created Data table '{self.data_table}' with {len(df)} rows")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to write CSV to SQL: {e}")
            return False

    def csv_to_sql(self, data, table ,replace=True): 
        try:
            print(f"Writing data to table '{table}'...")

            if isinstance(data, str):
                df = pd.read_csv(data)
            elif isinstance(data, pd.DataFrame):
                df = data
            else:
                raise ValueError("Data must be a file path (str) or a pandas DataFrame")

            df.to_sql(
                table,
                self.conn,
                if_exists='replace' if replace else 'fail',
                index=False
            )
            print(f"Successfully created table '{table}' with {len(df)} rows")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to write CSV to SQL: {e}")
            return False


    def data_fetch(self, table_name=None, apply_limit=False, limit=5):
        try:
            table = table_name if table_name else self.data_table

            if not self.check_table_exists(table):
                print(f"[ERROR] Table '{table}' does not exist.")
                return None

            cursor = self.conn.execute(f"PRAGMA table_info({table})")
            columns_info = cursor.fetchall()
            columns = [info[1] for info in columns_info]
            print(f"Table columns: {', '.join(columns)}")

            query = f"SELECT * FROM {table}"
            if apply_limit:
                query += f" LIMIT {limit}"

            df = pd.read_sql_query(query, self.conn)
            return df
        except Exception as e:
            print(f"[ERROR] Failed to fetch data: {e}")
            return None

    def analytical_data_fetch(self, tables: list, apply_limit=False, limit=5):
        results = []
        try:
            for table_name in tables:
                if not self.check_table_exists(table_name):
                    print(f"[WARNING] Skipping non-existent table: {table_name}")
                    continue

                query = f"SELECT * FROM {table_name}"
                if apply_limit:
                    query += f" LIMIT {limit}"

                df = pd.read_sql_query(query, self.conn)
                results.append({table_name: df.to_dict()})
        except Exception as e:
            print(f"[ERROR] Failed to fetch analytical data: {e}")
        return results
    
    def fetch_analytical_data(self, tables: list):

        results = {}
        try:
            for table_name in tables:
                if not self.check_table_exists(table_name):
                    print(f"[WARNING] Skipping non-existent table: {table_name}")
                    continue

                # Get column names
                cursor = self.conn.execute(f"PRAGMA table_info({table_name})")
                columns = [info[1] for info in cursor.fetchall()]
                
                # Fetch all rows
                cursor = self.conn.execute(f"SELECT * FROM {table_name}")
                rows = cursor.fetchall()
                
                # Convert to list of dictionaries (JSON serializable)
                table_data = []
                for row in rows:
                    table_data.append(dict(zip(columns, row)))
                
                results[table_name] = table_data
                
        except Exception as e:
            print(f"[ERROR] Failed to fetch analytical data for JSON: {e}")
        
        return results

    def close_connection(self):
        """Close the database connection"""
        self.conn.close()
