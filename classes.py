import gspread
from gspread_dataframe import set_with_dataframe
from oauth2client.service_account import ServiceAccountCredentials
import streamlit as st
import json
def get_env(key):
    return st.secrets[key]

def classes(report_date):
    query2 = f'''
    SELECT DISTINCT l."masterclassSlotId",mc."title",
FIRST_VALUE(l."createdAt") OVER (PARTITION BY l."masterclassSlotId" ORDER BY l."createdAt" ASC) AS "First lead",
LAST_VALUE(l."createdAt") OVER (PARTITION BY l."masterclassSlotId" ORDER BY l."createdAt"  ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS "last lead"
FROM "Leads" l
JOIN "MasterClassSlots" mcs ON l."masterclassSlotId"=mcs."id"
JOIN "MasterClass" mc ON mc."id"=mcs."masterClassId"
JOIN "Bootcamp" b ON b."id"=mc."bootcampId"
WHERE DATE(mcs."startDateTime")='{report_date}'
and source not in ('ret', 'arvind.tech', 'act*', 'retdm', 'null', 'cal','calendar', 'dm', 'email', 'push-notification', 'sms', 'freshdm', 'retp', 'api-ops', 'act', 'Zoom Reschedule', 'Livekit_Reschedule','default_source','gifff','Instagram','LMS','TradeWise')

'''
    return query2

def google_cred():
    # Load service account info from st.secrets
    keyfile_dict = dict(st.secrets["JSON"])
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    credentials = ServiceAccountCredentials.from_json_keyfile_dict(keyfile_dict, scope)
    client = gspread.authorize(credentials)
    return client


    
