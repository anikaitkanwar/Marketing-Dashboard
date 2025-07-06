import psycopg2
import pandas as pd
import os
import streamlit as st
from dotenv import load_dotenv
from datetime import date, timedelta

load_dotenv()

current_date = date.today()
yesterday = current_date - timedelta(days=1)

def get_env(key):
    return st.secrets[key]
    

def execute_query(query):
    conn = None
    cur = None

    try:
        conn = psycopg2.connect(
            host=get_env("DB_HOST"),
            database=get_env("DB_NAME"),
            user=get_env("DB_USER"),
            password=get_env("DB_PASSWORD"),
            port=int(get_env("DB_PORT"))
        )
        cur = conn.cursor()
        cur.execute(query)
        results = cur.fetchall()
        return results

    except (Exception, psycopg2.Error) as error:
        print("Error while executing query:", error)
        return None

    finally:
        if conn:
            if cur:
                cur.close()
            conn.close()
            print("PostgreSQL connection is closed")
            print(query)
