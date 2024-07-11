import mysql.connector
import pandas as pd
from mysql.connector import Error


def create_db_connection(host_name, user_name, user_password, db_name):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=db_name
        )
        print("MySQL Database connection successful")
    except Error as err:
        print(f"Error: '{err}'")

    return connection


def execute_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("Query successful")
    except Error as err:
        print(f"Error: '{err}'")


pop_sample = """
INSERT INTO data VALUES
('tst_ctrl_000',  '2024-07-05 18:57:00', 53, 1837.555744, 840.866086, 0.451228919, 34643.53957, 3922.04539, 1322.281093, 283.9104028, 0.039449559, 3356.026332, 80.9627251, 48.54391861);
"""

connection = create_db_connection("hugf", "root", '', 'simage_data')
execute_query(connection, pop_sample)
