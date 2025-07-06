import os
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import pytz
from concurrent.futures import ThreadPoolExecutor
import psycopg2
from executer import execute_query  
from spend import categorize
from leads import Leads_query
from joins import joins_query
from payments import payment_query
from funnels import create_funnel_table,create_funnel_table2
from spend import ads_query
from spend import ads_query2
from classes import classes
from classes import google_cred
import time
st.sidebar.title('TFU Marketing Dashboard')
side_radio = st.sidebar.radio(
    'Select one of these',
    ('Overall Analysis','Platform Level Analysis','Funnels Analysis','Class spend','Campaign Level'),
)
if 'summary_table' not in st.session_state:
    st.session_state['summary_table'] = pd.DataFrame()

if side_radio == 'Overall Analysis':
    input_placeholder = st.empty()

    with input_placeholder.container():
        start_date = st.date_input('Choose the Starting Date', value=None, format="YYYY/MM/DD")
        end_date = st.date_input('Choose the Ending Date', value=None, format="YYYY/MM/DD")
        end_date2 = end_date
        ist = pytz.timezone('Asia/Kolkata')
        if start_date:
            start_date = start_date.strftime('%Y-%m-%d')
            start = datetime.strptime(start_date + " 00:00:00", "%Y-%m-%d %H:%M:%S")
            start_ist = ist.localize(start)
            start_utc = start_ist.astimezone(pytz.utc).strftime("%Y-%m-%d %H:%M:%S")
            print(start_utc)
        else:
             print("start_date is None or empty")
        if end_date:
            end_date = end_date + pd.Timedelta(days=1)
            end_date = end_date.strftime('%Y-%m-%d')
            end = datetime.strptime(end_date + " 00:00:00", "%Y-%m-%d %H:%M:%S")
            end_ist = ist.localize(end)
            end_utc = end_ist.astimezone(pytz.utc).strftime("%Y-%m-%d %H:%M:%S")
            print(end_utc)
        else:
             print("end_date is None or empty")
        get_value = st.button("Get Value")
    if get_value:
        with st.spinner('Please hold tight, this will take some time', show_time=True):
            queries = [
                Leads_query(start_utc, end_utc),
                joins_query(start_utc, end_utc),
                payment_query(start_utc, end_utc),
                ads_query(start_date,end_date2)
            ]

            with ThreadPoolExecutor() as executor:
                results = list(executor.map(execute_query, queries))

            lead_df = pd.DataFrame(results[0])
            joins_df = pd.DataFrame(results[1])
            payment_df = pd.DataFrame(results[2])
            spend_df = pd.DataFrame(results[3])


            lead_df.columns = ['Category', 'Source','Comment' ,'LeadId', "UserId", "CreatedAt", "MasterClass",'UserCreated']
            lead_df = lead_df[lead_df['Category'].isin(['Finance', 'Spirituality'])]
            # lead_df.loc[(lead_df['Comment'].str.contains('_aj|astroji|astro_ji',case = False, na=False)),'Category'] = 'Astroji'
            joins_df.columns = ['Category', 'Source', 'Comment','LeadId', "UserId", "CreatedAt", "MasterClass",'meetingId',"UserCreated"]
            joins_df = joins_df[joins_df['Category'].isin(['Finance', 'Spirituality'])]
            # joins_df.loc[(joins_df['Comment'].str.contains('_aj|astroji|astro_ji', na=False,case = False)),'Category'] = 'Astroji'
            payment_df.columns = ['PaymentId', 'Amount', 'Payment Category', "UserId", "Pay Created", "MC/BX Paid For"]
            spend_df.columns = ['Platform', 'CampaignName', 'AdsetName', 'AdName', 'AdId', 'Clicks', 'Impressions', 'Reach','Cost', 'ReportDate']
            payment_df['Amount'] = payment_df['Amount'].astype(int)
            spend_df = spend_df.replace(-1, 0)
            spend_df['Category'] = spend_df.apply(categorize, axis=1)
            # spend_df.loc[(spend_df['CampaignName'].str.contains('_aj', na=False,case = False)),'Category'] = 'Astroji'
            # spend_df['Category'] = spend_df['CampaignName'].apply(lambda x: 'Spirituality' if ('_al' in str(x).lower() and '_tw' not in str(x).lower()) else ('Finance' if '_tw' in str(x).lower() else None))
            start_date2 = start_date
            start_date = pd.to_datetime(start_utc)
            categories = lead_df['Category'].unique()
            result = {}
            new_result = {}
            daily_leads = pd.DataFrame()
            for i, cat in enumerate(categories):  
                category_leads = lead_df[lead_df['Category']==cat]
                category_joins = joins_df[joins_df['Category']==cat]
                category_spend = spend_df[spend_df['Category'] == cat]
                spend = category_spend['Cost'].sum()
                lead_counts = category_leads['LeadId'].nunique()
                user_counts = category_leads['UserId'].nunique()
                join_counts = category_joins['UserId'].nunique()
                total_join_counts = category_joins['meetingId'].nunique()
                new_lead_counts = category_leads[category_leads['UserCreated'] >= start_date]['LeadId'].nunique()
                new_user_counts = category_leads[category_leads['UserCreated'] >= start_date]['UserId'].nunique()
                new_join_counts = category_joins[category_joins['UserCreated'] >= start_date]['UserId'].nunique()
                new_total_join_counts = category_joins[category_joins['UserCreated'] >= start_date]['meetingId'].nunique()
                rev_raw_df = pd.merge(lead_df, payment_df, on="UserId", how='inner')
                # rev_raw_df.loc[(rev_raw_df['Category'] == 'Astroji'),'Payment Category'] = 'Astroji'
                rev_raw_df = rev_raw_df[['UserId','CreatedAt','Pay Created','PaymentId','Category','Payment Category','Amount']]
                rev_raw_df.drop_duplicates()
                rev_df = rev_raw_df[rev_raw_df["Pay Created"] > rev_raw_df["CreatedAt"]]
                rev_df_final = rev_df[rev_df['Payment Category'] == cat]
                rev_corrected = rev_df_final[['Amount','PaymentId']].drop_duplicates()
                revenue = rev_corrected['Amount'].sum()
                converted = rev_corrected[rev_corrected["Amount"] > 399]['PaymentId'].count()
                new_df = category_leads[category_leads['UserCreated'] >= start_date]
                new_rev_raw_df = pd.merge(new_df, payment_df, on="UserId", how='inner')
                new_rev_raw_df.drop_duplicates()
                rev_new = new_rev_raw_df[new_rev_raw_df["Pay Created"] > new_rev_raw_df["CreatedAt"]]
                new_rev_df_final = rev_new[rev_new['Category'] == rev_new['Payment Category']]
                new_rev_corrected = new_rev_df_final[['Amount','PaymentId']].drop_duplicates()
                rev_corrected.to_csv(f'{cat}total_data.csv')
                new_revenue = new_rev_corrected['Amount'].sum()
                new_converted = new_rev_corrected[new_rev_corrected["Amount"] > 299]['PaymentId'].count()
                if join_counts != 0:
                    join_mult = total_join_counts / join_counts
                else:
                    join_mult = 0
                if new_join_counts != 0:
                    new_join_mult = total_join_counts / join_counts
                else:
                    new_join_mult = 0
                new_spend = spend/lead_counts*new_lead_counts
                if spend != 0:
                    Roi = ((revenue / spend)-1)*100
                else:
                    Roi = 0
                if new_spend != 0:
                    new_Roi = ((new_revenue / new_spend)-1)*100
                else:
                    new_Roi = 0
                if new_user_counts != 0:
                    new_join_per = (new_join_counts/new_user_counts)*100
                else:
                    new_join_per = 0

                result[cat] = [spend,
                lead_counts,
                spend/user_counts,
                user_counts,
                join_counts,
                (join_counts/user_counts)*100,
                total_join_counts,
                join_mult,
                revenue,
                converted,
                Roi,
                revenue/converted if converted != 0 else 0,
                100 - ( new_user_counts*100/user_counts) if user_counts != 0 else 0]

                new_result[cat] = [new_spend,
                new_lead_counts,
                new_user_counts,
                new_join_counts,
                new_join_per,
                new_total_join_counts,
                new_join_mult,
                new_revenue,
                new_converted,
                new_Roi]

                
            cat_df = pd.DataFrame(
                result,
                index=[
                    "Spend",
                    'Lead Count',
                    "CPU",
                    'User Count',
                    'Join Count',
                    'Joining Percent',
                    'Total Join Count',
                    'Joining Multiplier',
                    'Revenue',
                    'Converted Count',
                    'ROI',
                    'AOV',
                    'Repeat_Old_users'

                ]
            )
            # Format numbers cleanly
            cat_df.loc[['Joining Percent', 'Joining Multiplier','ROI','CPU']] = cat_df.loc[['Joining Percent', 'Joining Multiplier','ROI','CPU']].astype(float)
            cat_df.loc[cat_df.index.difference(['Joining Percent', 'Joining Multiplier',"ROI","CPU"])] = cat_df.loc[cat_df.index.difference(['Joining Percent', 'Joining Multiplier',"ROI","CPU"])].astype(int)
            def format_value(x):
                try:
                    if isinstance(x, float) and not x.is_integer():
                        return f"{x:,.2f}"
                    else:
                        return f"{int(x):,}"
                except:
                    return x

            # Apply formatting
            cat_df = cat_df.map(format_value)
            




            new_cat_df = pd.DataFrame(
                new_result,
                index=['Adjusted Spend',
                    'Lead Count',
                    'User Count',
                    'Join Count',
                    'Joining Percent',
                    'Total Join Count',
                    'Joining Multiplier',
                    'Revenue',
                    'Converted Count',
                    'ROI'
                ]
            )
            new_cat_df.loc[['Joining Percent', 'Joining Multiplier','ROI']] = new_cat_df.loc[['Joining Percent', 'Joining Multiplier','ROI']].astype(float)
            new_cat_df.loc[new_cat_df.index.difference(['Joining Percent', 'Joining Multiplier',"ROI"])] = new_cat_df.loc[new_cat_df.index.difference(['Joining Percent', 'Joining Multiplier',"ROI"])].astype(int)
            new_cat_df = new_cat_df.map(format_value)
            st.session_state['summary_table'] = cat_df
            
            summary_table2 = st.session_state['summary_table2'] = new_cat_df
            if 'summary_table' in st.session_state:
                summary_table = st.session_state['summary_table']
                st.header('Overall Analysis')
                st.subheader(f"Date range {start_date2} and {end_date2}")
                st.dataframe(summary_table)
                csv_text = summary_table.to_csv(index=True, sep='\t')
                st.text_area("Copy this table (Excel-friendly):", csv_text, height=150)

            
            summary_table2 = st.session_state['summary_table2']
            st.subheader('Fresh Analysis')
            st.dataframe(summary_table2)
            csv_data = summary_table2.to_csv(index=True,sep = '\t')
            st.text_area("Copy the table below:", csv_data, height=200)
            categories = lead_df['Category'].unique()
            
            cols = st.columns(len(categories)) 
            
            for i, cat in enumerate(categories):
                
                category_leads = lead_df[lead_df['Category']==cat]
                category_joins = joins_df[joins_df['Category']==cat]
                new_category_leads = category_leads[category_leads['UserCreated'] >= start_date]
                category_spend = spend_df[spend_df['Category']==cat]

                daywise_leads_df = category_leads[['LeadId', 'CreatedAt']].copy()
                new_daywise_leads_df = new_category_leads[['LeadId', 'CreatedAt']].copy()
                new_daywise_leads_df['CreatedAt'] = pd.to_datetime(new_daywise_leads_df['CreatedAt'])
                new_daywise_leads_df['CreatedAt'] = new_daywise_leads_df['CreatedAt'].dt.tz_localize('UTC').dt.tz_convert(ist)
                new_daywise_leads_df['Day'] = new_daywise_leads_df['CreatedAt'].dt.date
                new_daywise_leads_df = new_daywise_leads_df.groupby('Day')['LeadId'].count().reset_index(name='Leads')
                daywise_leads_df['CreatedAt'] = pd.to_datetime(daywise_leads_df['CreatedAt'])
                daywise_leads_df['CreatedAt'] = daywise_leads_df['CreatedAt'].dt.tz_localize('UTC').dt.tz_convert(ist)
                daywise_leads_df['Day'] = daywise_leads_df['CreatedAt'].dt.date
                daywise_leads_df = daywise_leads_df.groupby('Day')['LeadId'].count().reset_index(name='Leads')
                category_spend['ReportDate'] = category_spend['ReportDate'].dt.date
                daywise_spend_df = category_spend.groupby('ReportDate')['Cost'].sum().reset_index(name='Spend')
                daywise_leads_df['CPL'] = daywise_spend_df['Spend']/daywise_leads_df['Leads']
                daywise_leads_df['Spend'] = daywise_spend_df['Spend']
                st.session_state['daywise_leads_df'] = daywise_leads_df
                st.session_state['new_daywise_leads_df'] = new_daywise_leads_df
                daywise_leads_df['Type'] = 'Total Leads'
                new_daywise_leads_df['Type'] = 'New Leads'
                
                
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=st.session_state['daywise_leads_df']['Day'],
                    y=st.session_state['daywise_leads_df']['Leads'],
                    mode='lines+markers',
                    name='Total Leads'
                ))
                fig.add_trace(go.Scatter(
                    x=st.session_state['daywise_leads_df']['Day'],
                    y=st.session_state['daywise_leads_df']['CPL'],
                    mode='lines+markers',
                    name='CPL'
                ))
                fig.add_trace(go.Scatter(
                    x=st.session_state['daywise_leads_df']['Day'],
                    y=st.session_state['daywise_leads_df']['Spend'],
                    mode='lines+markers',
                    name='Spend'
                ))

                fig.add_trace(go.Scatter(
                    x=st.session_state['new_daywise_leads_df']['Day'],
                    y=st.session_state['new_daywise_leads_df']['Leads'],
                    mode='lines+markers',
                    name='New Leads'
                ))
                fig.update_layout(title=f"Leads and CPL Trend Over Days for {cat}", xaxis_title='Day', yaxis_title=f'Number of Leads for {cat}', template='plotly_dark')
                with cols[i]:
                    st.plotly_chart(fig,use_container_width=True)
                daywise_leads_df = pd.DataFrame()
                new_daywise_leads_df = pd.DataFrame()
            stages = ["Leads", "Joining", "Sales"]

        
            list1 = cat_df.iloc[[3, 4, 9], 0].tolist()
            list2 = cat_df.iloc[[3, 4, 9], 1].tolist()

            # Reset index and add necessary columns for df_mtl (Finance)
            df_mt1 = pd.DataFrame()
            df_mt1['metrics'] = list1
            df_mt1['stage'] = stages
            df_mt1['office'] = 'Finance'

            # Reset index and add necessary columns for df_mt2 (Spirituality)
            df_mt2 = pd.DataFrame()
            df_mt2['metrics'] = list2
            df_mt2['stage'] = stages
            df_mt2['office'] = 'Spirituality'

            # Combine both DataFrames
            df = pd.concat([df_mt1, df_mt2], axis=0).reset_index(drop=True)

            # Rename column for metrics and ensure we have the right structure
            df.columns = ['metrics', 'stage', 'office']
            df['metrics'] = df['metrics'].str.replace(',', '').astype(float)
            print(df_mt1)
            # Now, create the funnel chart
            fig = px.funnel(df, x='metrics', y='stage', color='office')

            # Plot the chart
            st.plotly_chart(fig, use_container_width=True)
        st.success('Your analysis is Ready!')


if side_radio == 'Platform Level Analysis':
    input_placeholder = st.empty()

    with input_placeholder.container():
        start_date = st.date_input('Choose the Starting Date', value=None, format="YYYY/MM/DD")
        end_date = st.date_input('Choose the Ending Date', value=None, format="YYYY/MM/DD")
        platform = st.selectbox('Select the Platform',['Google','Facebook','Taboola','twitter','direct'])
        end_date2 = end_date
        ist = pytz.timezone('Asia/Kolkata')
        if start_date:
            start_date = start_date.strftime('%Y-%m-%d')
            start = datetime.strptime(start_date + " 00:00:00", "%Y-%m-%d %H:%M:%S")
            start_ist = ist.localize(start)
            start_utc = start_ist.astimezone(pytz.utc).strftime("%Y-%m-%d %H:%M:%S")
            print(start_utc)
        else:
             print("start_date is None or empty")
        if end_date:
            end_date = end_date + pd.Timedelta(days=1)
            end_date = end_date.strftime('%Y-%m-%d')
            end = datetime.strptime(end_date + " 00:00:00", "%Y-%m-%d %H:%M:%S")
            end_ist = ist.localize(end)
            end_utc = end_ist.astimezone(pytz.utc).strftime("%Y-%m-%d %H:%M:%S")
            print(end_utc)
        else:
             print("end_date is None or empty")
        get_value = st.button("Get Value")
    if get_value:
        queries = [
            Leads_query(start_utc, end_utc),
            joins_query(start_utc, end_utc),
            payment_query(start_utc, end_utc),
            ads_query(start_date,end_date2)
        ]

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(execute_query, queries))

        lead_df = pd.DataFrame(results[0])
        joins_df = pd.DataFrame(results[1])
        payment_df = pd.DataFrame(results[2])
        spend_df = pd.DataFrame(results[3])


        lead_df.columns = ['Category', 'Source','Comment' ,'LeadId', "UserId", "CreatedAt", "MasterClass",'UserCreated']
        lead_df = lead_df[lead_df['Source'].fillna('').str.lower() == platform.lower()]
        lead_df.loc[(lead_df['Comment'].str.contains('_aj|astroji|astro_ji',case = False, na=False)),'Category'] = 'Astroji'
        joins_df.columns = ['Category', 'Source', 'Comment','LeadId', "UserId", "CreatedAt", "MasterClass",'meetingId',"UserCreated"]
        joins_df = joins_df[joins_df['Source'].fillna('').str.lower() == platform.lower()]
        joins_df.loc[(joins_df['Comment'].str.contains('_aj|astroji|astro_ji',case = False, na=False)),'Category'] = 'Astroji'
        payment_df.columns = ['PaymentId', 'Amount', 'Payment Category', "UserId", "Pay Created", "MC/BX Paid For"]
        spend_df.columns = ['Platform', 'CampaignName', 'AdsetName', 'AdName', 'AdId', 'Clicks', 'Impressions', 'Reach','Cost', 'ReportDate']
        spend_df = spend_df[spend_df['Platform'].fillna('').str.lower() == platform.lower()]
        payment_df['Amount'] = payment_df['Amount'].astype(int)
        spend_df = spend_df.replace(-1, 0)
        spend_df['Category'] = spend_df.apply(categorize, axis=1)
        spend_df.loc[(spend_df['CampaignName'].str.contains('_aj', na = False, case = False)),'Category'] = 'Astroji'
        # spend_df['Category'] = spend_df['CampaignName'].apply(lambda x: 'Spirituality' if ('_al' in str(x).lower() and '_tw' not in str(x).lower()) else ('Finance' if '_tw' in str(x).lower() else None))
        start_date2 = start_date
        start_date = pd.to_datetime(start_utc)
        categories = lead_df['Category'].unique()
        result = {}
        new_result = {}
        
        daily_leads = pd.DataFrame()
        for i, cat in enumerate(categories):  
            category_leads = lead_df[lead_df['Category']==cat]
            category_joins = joins_df[joins_df['Category']==cat]
            category_spend = spend_df[(spend_df['Category'] == cat) & (spend_df['Platform'].str.lower() == platform.lower())]
            spend = category_spend['Cost'].sum()
            lead_counts = category_leads['LeadId'].nunique()
            user_counts = category_leads['UserId'].nunique()
            join_counts = category_joins['UserId'].nunique()
            total_join_counts = category_joins['meetingId'].nunique()
            new_lead_counts = category_leads[category_leads['UserCreated'] >= start_date]['LeadId'].nunique()
            new_user_counts = category_leads[category_leads['UserCreated'] >= start_date]['UserId'].nunique()
            new_join_counts = category_joins[category_joins['UserCreated'] >= start_date]['UserId'].nunique()
            new_total_join_counts = category_joins[category_joins['UserCreated'] >= start_date]['meetingId'].nunique()
            rev_raw_df = pd.merge(lead_df, payment_df, on="UserId", how='inner')
            rev_raw_df.drop_duplicates()
            rev_df = rev_raw_df[rev_raw_df["Pay Created"] > rev_raw_df["CreatedAt"]]
            rev_df_final = rev_df[rev_df['Payment Category'] == cat]
            rev_corrected = rev_df_final[['Amount','PaymentId']].drop_duplicates()
            revenue = rev_corrected['Amount'].sum()
            converted = rev_corrected[rev_corrected["Amount"] > 299]['PaymentId'].count()
            new_df = category_leads[category_leads['UserCreated'] >= start_date]
            new_rev_raw_df = pd.merge(new_df, payment_df, on="UserId", how='inner')
            new_rev_raw_df.drop_duplicates()
            rev_new = new_rev_raw_df[new_rev_raw_df["Pay Created"] > new_rev_raw_df["CreatedAt"]]
            new_rev_df_final = rev_new[rev_new['Category'] == rev_new['Payment Category']]
            new_rev_corrected = new_rev_df_final[['Amount','PaymentId']].drop_duplicates()
            new_revenue = new_rev_corrected['Amount'].sum()
            new_converted = new_rev_corrected[new_rev_corrected["Amount"] > 299]['PaymentId'].count()
            if join_counts != 0:
                join_mult = total_join_counts / join_counts
            else:
                join_mult = 0
            if new_join_counts != 0:
                new_join_mult = total_join_counts / join_counts
            else:
                new_join_mult = 0
            new_spend = spend/lead_counts*new_lead_counts
            if spend != 0:
                Roi = ((revenue / spend)-1)*100
            else:
                Roi = 0
            if new_spend != 0:
                new_Roi = ((new_revenue / new_spend)-1)*100
            else:
                new_Roi = 0
            if new_user_counts != 0:
                new_join_per = (new_join_counts/new_user_counts)*100
            else:
                new_join_per = 0

            result[cat] = [spend,
            lead_counts,
            spend/user_counts,
            user_counts,
            join_counts,
            (join_counts/user_counts)*100,
            total_join_counts,
            join_mult,
            revenue,
            converted,
            Roi,
            revenue/converted if converted != 0 else 0,
            100 - ( new_user_counts*100/user_counts) if user_counts != 0 else 0]

            new_result[cat] = [new_spend,
            new_lead_counts,
            new_user_counts,
            new_join_counts,
            new_join_per,
            new_total_join_counts,
            new_join_mult,
            new_revenue,
            new_converted,
            new_Roi]

            
        cat_df = pd.DataFrame(
            result,
            index=[
                "Spend",
                'Lead Count',
                "CPU",
                'User Count',
                'Join Count',
                'Joining Percent',
                'Total Join Count',
                'Joining Multiplier',
                'Revenue',
                'Converted Count',
                'ROI',
                'AOV',
                'W-- Repeat'

            ]
        )
        # Format numbers cleanly
        cat_df.loc[['Joining Percent', 'Joining Multiplier','ROI','CPU']] = cat_df.loc[['Joining Percent', 'Joining Multiplier','ROI','CPU']].astype(float)
        cat_df.loc[cat_df.index.difference(['Joining Percent', 'Joining Multiplier',"ROI","CPU"])] = cat_df.loc[cat_df.index.difference(['Joining Percent', 'Joining Multiplier',"ROI","CPU"])].astype(int)
        def format_value(x):
            try:
                if isinstance(x, float) and not x.is_integer():
                    return f"{x:,.2f}"
                else:
                    return f"{int(x):,}"
            except:
                return x

        # Apply formatting
        cat_df = cat_df.map(format_value)
        




        new_cat_df = pd.DataFrame(
            new_result,
            index=['Adjusted Spend',
                'Lead Count',
                'User Count',
                'Join Count',
                'Joining Percent',
                'Total Join Count',
                'Joining Multiplier',
                'Revenue',
                'Converted Count',
                'ROI'
            ]
        )
        new_cat_df.loc[['Joining Percent', 'Joining Multiplier','ROI']] = new_cat_df.loc[['Joining Percent', 'Joining Multiplier','ROI']].astype(float)
        new_cat_df.loc[new_cat_df.index.difference(['Joining Percent', 'Joining Multiplier',"ROI"])] = new_cat_df.loc[new_cat_df.index.difference(['Joining Percent', 'Joining Multiplier',"ROI"])].astype(int)
        new_cat_df = new_cat_df.map(format_value)
        st.session_state['summary_table'] = cat_df
        
        summary_table2 = st.session_state['summary_table2'] = new_cat_df
        if 'summary_table' in st.session_state:
            summary_table = st.session_state['summary_table']
            st.header('Overall Analysis')
            st.subheader(f"Date range {start_date2} and {end_date2}")
            st.dataframe(summary_table)
            csv_text = summary_table.to_csv(index=True, sep='\t')
            st.text_area("Copy this table (Excel-friendly):", csv_text, height=150)

        
        summary_table2 = st.session_state['summary_table2']
        st.subheader('Fresh Analysis')
        st.dataframe(summary_table2)
        csv_data = summary_table2.to_csv(index=True,sep = '\t')
        st.text_area("Copy the table below:", csv_data, height=200)
        categories = lead_df['Category'].unique()
        
        cols = st.columns(len(categories)) 
        
        for i, cat in enumerate(categories):
            
            category_leads = lead_df[lead_df['Category']==cat]
            category_joins = joins_df[joins_df['Category']==cat]
            new_category_leads = category_leads[category_leads['UserCreated'] >= start_date]
            category_spend = spend_df[spend_df['Category']==cat]

            daywise_leads_df = category_leads[['LeadId', 'CreatedAt']].copy()
            new_daywise_leads_df = new_category_leads[['LeadId', 'CreatedAt']].copy()
            new_daywise_leads_df['CreatedAt'] = pd.to_datetime(new_daywise_leads_df['CreatedAt'])
            new_daywise_leads_df['CreatedAt'] = new_daywise_leads_df['CreatedAt'].dt.tz_localize('UTC').dt.tz_convert(ist)
            new_daywise_leads_df['Day'] = new_daywise_leads_df['CreatedAt'].dt.date
            new_daywise_leads_df = new_daywise_leads_df.groupby('Day')['LeadId'].count().reset_index(name='Leads')
            daywise_leads_df['CreatedAt'] = pd.to_datetime(daywise_leads_df['CreatedAt'])
            daywise_leads_df['CreatedAt'] = daywise_leads_df['CreatedAt'].dt.tz_localize('UTC').dt.tz_convert(ist)
            daywise_leads_df['Day'] = daywise_leads_df['CreatedAt'].dt.date
            daywise_leads_df = daywise_leads_df.groupby('Day')['LeadId'].count().reset_index(name='Leads')
            category_spend['ReportDate'] = category_spend['ReportDate'].dt.date
            daywise_spend_df = category_spend.groupby('ReportDate')['Cost'].sum().reset_index(name='Spend')
            daywise_leads_df['CPL'] = daywise_spend_df['Spend']/daywise_leads_df['Leads']
            daywise_leads_df['Spend'] = daywise_spend_df['Spend']
            st.session_state['daywise_leads_df'] = daywise_leads_df
            st.session_state['new_daywise_leads_df'] = new_daywise_leads_df
            daywise_leads_df['Type'] = 'Total Leads'
            new_daywise_leads_df['Type'] = 'New Leads'
            
            
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=st.session_state['daywise_leads_df']['Day'],
                y=st.session_state['daywise_leads_df']['Leads'],
                mode='lines+markers',
                name='Total Leads'
            ))
            fig.add_trace(go.Scatter(
                x=st.session_state['daywise_leads_df']['Day'],
                y=st.session_state['daywise_leads_df']['CPL'],
                mode='lines+markers',
                name='CPL'
            ))
            fig.add_trace(go.Scatter(
                x=st.session_state['daywise_leads_df']['Day'],
                y=st.session_state['daywise_leads_df']['Spend'],
                mode='lines+markers',
                name='Spend'
            ))

            fig.add_trace(go.Scatter(
                x=st.session_state['new_daywise_leads_df']['Day'],
                y=st.session_state['new_daywise_leads_df']['Leads'],
                mode='lines+markers',
                name='New Leads'
            ))
            fig.update_layout(title=f"Leads and CPL Trend Over Days for {cat}", xaxis_title='Day', yaxis_title=f'Number of Leads for {cat}', template='plotly_dark')
            with cols[i]:
                st.plotly_chart(fig,use_container_width=True)
            daywise_leads_df = pd.DataFrame()
            new_daywise_leads_df = pd.DataFrame()





            


if side_radio == 'Funnels Analysis':
    input_placeholder = st.empty()

    with input_placeholder.container():
        start_date = st.date_input('Choose the Starting Date', value=None, format="YYYY/MM/DD")
        end_date = st.date_input('Choose the Ending Date', value=None, format="YYYY/MM/DD")
        platform = st.selectbox('Select the Platform',['Google','Facebook','Taboola'])
        category = st.selectbox('Select the Platform',['Spirituality','Finance','Astroji'])
        end_date2 = end_date
        ist = pytz.timezone('Asia/Kolkata')
        if start_date:
            start_date = start_date.strftime('%Y-%m-%d')
            start = datetime.strptime(start_date + " 00:00:00", "%Y-%m-%d %H:%M:%S")
            start_ist = ist.localize(start)
            start_utc = start_ist.astimezone(pytz.utc).strftime("%Y-%m-%d %H:%M:%S")
            print(start_utc)
        else:
             print("start_date is None or empty")
        if end_date:
            end_date = end_date + pd.Timedelta(days=1)
            end_date = end_date.strftime('%Y-%m-%d')
            end = datetime.strptime(end_date + " 00:00:00", "%Y-%m-%d %H:%M:%S")
            end_ist = ist.localize(end)
            end_utc = end_ist.astimezone(pytz.utc).strftime("%Y-%m-%d %H:%M:%S")
            print(end_utc)
        else:
             print("end_date is None or empty")
        get_value = st.button("Get Value")
    if get_value:
        queries = [
            Leads_query(start_utc, end_utc),
            joins_query(start_utc, end_utc),
            payment_query(start_utc, end_utc),
            ads_query(start_date,end_date2)
        ]

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(execute_query, queries))

        lead_df = pd.DataFrame(results[0])
        joins_df = pd.DataFrame(results[1])
        payment_df = pd.DataFrame(results[2])
        spend_df = pd.DataFrame(results[3])


        lead_df.columns = ['Category', 'Source','Comment' ,'LeadId', "UserId", "CreatedAt", "MasterClass",'UserCreated']
        lead_df = lead_df[lead_df['Source'].fillna('').str.lower() == platform.lower()]
        # lead_df.loc[(lead_df['Comment'].str.contains('_aj|astroji|astro_ji',case = False, na=False)),'Category'] = 'Astroji'
        lead_df = lead_df[lead_df['Category'].fillna('').str.lower() == category.lower()]
        lead_df = create_funnel_table(lead_df,platform)
        joins_df.columns = ['Category', 'Source', 'Comment','LeadId', "UserId", "CreatedAt", "MasterClass",'meetingId',"UserCreated"]
        # joins_df.loc[(joins_df['Comment'].str.contains('_aj|astroji|astro_ji',case = False, na=False)),'Category'] = 'Astroji'
        joins_df = joins_df[joins_df['Source'].fillna('').str.lower() == platform.lower()]
        joins_df = joins_df[joins_df['Category'].fillna('').str.lower() == category.lower()]
        joins_df  = create_funnel_table(joins_df,platform)
        payment_df.columns = ['PaymentId', 'Amount', 'Payment Category', "UserId", "Pay Created", "MC/BX Paid For"]
        spend_df.columns = ['Platform', 'CampaignName', 'AdsetName', 'AdName', 'AdId', 'Clicks', 'Impressions', 'Reach','Cost', 'ReportDate']
        spend_df = spend_df[spend_df['Platform'].fillna('').str.lower() == platform.lower()]
        payment_df['Amount'] = payment_df['Amount'].astype(int)
        spend_df = spend_df.replace(-1, 0)
        spend_df['Category'] = spend_df.apply(categorize, axis=1)
        # spend_df['Category'] = spend_df['CampaignName'].apply(lambda x: 'Spirituality' if ('_al' in str(x).lower() and '_tw' not in str(x).lower()) else ('Finance' if '_tw' in str(x).lower() else None))
        # spend_df.loc[(spend_df['CampaignName'].str.contains('_aj|astroji|astro_ji',case = False, na=False)),'Category'] = 'Astroji'
        spend_df= spend_df[spend_df['Category'].fillna('').str.lower() == category.lower()]
        spend_df = create_funnel_table2(spend_df,platform)
        start_date2 = start_date
        start_date = pd.to_datetime(start_utc)
        funnels = lead_df['Funnel'].unique()
        result = {}
        new_result = {}
        
        daily_leads = pd.DataFrame()
        for i, funnel in enumerate(funnels):  
            category_leads = lead_df[lead_df['Funnel']==funnel]
            category_joins = joins_df[joins_df['Funnel']==funnel]
            category_spend = spend_df[spend_df['Funnel'] == funnel]
            spend = category_spend['Cost'].sum()
            lead_counts = category_leads['LeadId'].nunique()
            user_counts = category_leads['UserId'].nunique()
            join_counts = category_joins['UserId'].nunique()
            total_join_counts = category_joins['meetingId'].nunique()
            new_lead_counts = category_leads[category_leads['UserCreated'] >= start_date]['LeadId'].nunique()
            new_user_counts = category_leads[category_leads['UserCreated'] >= start_date]['UserId'].nunique()
            new_join_counts = category_joins[category_joins['UserCreated'] >= start_date]['UserId'].nunique()
            new_total_join_counts = category_joins[category_joins['UserCreated'] >= start_date]['meetingId'].nunique()
            rev_raw_df = pd.merge(category_leads, payment_df, on="UserId", how='inner')
            # rev_raw_df.loc[(rev_raw_df['Category'] == 'Astroji'),'Payment Category'] = 'Astroji'
            rev_raw_df.drop_duplicates()
            rev_df = rev_raw_df[rev_raw_df["Pay Created"] > rev_raw_df["CreatedAt"]]
            rev_df_final = rev_df[rev_df['Category'] == rev_df['Payment Category']]
            rev_corrected = rev_df_final[['Amount','PaymentId']].drop_duplicates()
            revenue = rev_corrected['Amount'].sum()
            converted = rev_corrected[rev_corrected["Amount"] > 299]['PaymentId'].count()
            new_df = category_leads[category_leads['UserCreated'] >= start_date]
            new_rev_raw_df = pd.merge(new_df, payment_df, on="UserId", how='inner')
            new_rev_raw_df.drop_duplicates()
            rev_new = new_rev_raw_df[new_rev_raw_df["Pay Created"] > new_rev_raw_df["CreatedAt"]]
            new_rev_df_final = rev_new[rev_new['Category'] == rev_new['Payment Category']]
            new_rev_corrected = new_rev_df_final[['Amount','PaymentId']].drop_duplicates()
            new_revenue = new_rev_corrected['Amount'].sum()
            new_converted = new_rev_corrected[new_rev_corrected["Amount"] > 299]['PaymentId'].count()
            if join_counts != 0:
                join_mult = total_join_counts / join_counts
            else:
                join_mult = 0
            if new_join_counts != 0:
                new_join_mult = total_join_counts / join_counts
            else:
                new_join_mult = 0
            new_spend = spend/lead_counts*new_lead_counts
            if spend != 0:
                Roi = ((revenue / spend)-1)*100
            else:
                Roi = 0
            if new_spend != 0:
                new_Roi = ((new_revenue / new_spend)-1)*100
            else:
                new_Roi = 0
            if new_user_counts != 0:
                new_join_per = (new_join_counts/new_user_counts)*100
            else:
                new_join_per = 0

            result[funnel] = [spend,
            lead_counts,
            spend/user_counts,
            user_counts,
            join_counts,
            (join_counts / user_counts) * 100 if user_counts != 0 else 0,
            total_join_counts,
            join_mult,
            revenue,
            converted,
            Roi,
            revenue/converted if converted != 0 else 0,
            100 - ( new_user_counts*100/user_counts) if user_counts != 0 else 0]

            new_result[funnel] = [new_spend,
            new_lead_counts,
            new_user_counts,
            new_join_counts,
            new_join_per,
            new_total_join_counts,
            new_join_mult,
            new_revenue,
            new_converted,
            new_Roi]

            
        cat_df = pd.DataFrame(
            result,
            index=[
                "Spend",
                'Lead Count',
                "CPU",
                'User Count',
                'Join Count',
                'Joining Percent',
                'Total Join Count',
                'Joining Multiplier',
                'Revenue',
                'Converted Count',
                'ROI',
                'AOV',
                'W-- User Repeat'

            ]
        )
        # Format numbers cleanly
        cat_df.loc[['Joining Percent', 'Joining Multiplier','ROI','CPU']] = cat_df.loc[['Joining Percent', 'Joining Multiplier','ROI','CPU']].astype(float)
        cat_df.loc[cat_df.index.difference(['Joining Percent', 'Joining Multiplier',"ROI","CPU"])] = cat_df.loc[cat_df.index.difference(['Joining Percent', 'Joining Multiplier',"ROI","CPU"])].fillna(0).astype(int)
        def format_value(x):
            try:
                if isinstance(x, float) and not x.is_integer():
                    return f"{x:,.2f}"
                else:
                    return f"{int(x):,}"
            except:
                return x

        # Apply formatting
        cat_df = cat_df.map(format_value)
        




        new_cat_df = pd.DataFrame(
            new_result,
            index=['Adjusted Spend',
                'Lead Count',
                'User Count',
                'Join Count',
                'Joining Percent',
                'Total Join Count',
                'Joining Multiplier',
                'Revenue',
                'Converted Count',
                'ROI'
            ]
        )
        new_cat_df.loc[['Joining Percent', 'Joining Multiplier','ROI']] = new_cat_df.loc[['Joining Percent', 'Joining Multiplier','ROI']].astype(float)
        new_cat_df.loc[new_cat_df.index.difference(['Joining Percent', 'Joining Multiplier',"ROI"])] = new_cat_df.loc[new_cat_df.index.difference(['Joining Percent', 'Joining Multiplier',"ROI"])].fillna(0).astype(int)
        new_cat_df = new_cat_df.map(format_value)
        st.session_state['summary_table'] = cat_df
        
        summary_table2 = st.session_state['summary_table2'] = new_cat_df
        if 'summary_table' in st.session_state:
            summary_table = st.session_state['summary_table']
            st.header(f'Funnel analysis for {category} and platform is {platform}')
            st.subheader(f"Date range {start_date2} and {end_date2}")
            st.dataframe(summary_table)
            csv_text = summary_table.to_csv(index=True, sep='\t')
            st.text_area("Copy this table (Excel-friendly):", csv_text, height=150)

        
        summary_table2 = st.session_state['summary_table2']
        st.subheader(f'Fresh Funnel for {category} and platform is {platform}')
        st.dataframe(summary_table2)
        csv_data = summary_table2.to_csv(index=True,sep = '\t')
        st.text_area("Copy the table below:", csv_data, height=200)
        funnels = lead_df['Funnel'].unique()
        
        cols = st.columns(len(funnels)) 
        
        for i, funnel in enumerate(funnels):
            
            category_leads = lead_df[lead_df['Funnel']==funnel]
            category_joins = joins_df[joins_df['Funnel']==funnel]
            new_category_leads = category_leads[category_leads['UserCreated'] >= start_date]
            category_spend = spend_df[spend_df['Funnel']==funnel]

            daywise_leads_df = category_leads[['LeadId', 'CreatedAt']].copy()
            new_daywise_leads_df = new_category_leads[['LeadId', 'CreatedAt']].copy()
            new_daywise_leads_df['CreatedAt'] = pd.to_datetime(new_daywise_leads_df['CreatedAt'])
            new_daywise_leads_df['CreatedAt'] = new_daywise_leads_df['CreatedAt'].dt.tz_localize('UTC').dt.tz_convert(ist)
            new_daywise_leads_df['Day'] = new_daywise_leads_df['CreatedAt'].dt.date
            new_daywise_leads_df = new_daywise_leads_df.groupby('Day')['LeadId'].count().reset_index(name='Leads')
            daywise_leads_df['CreatedAt'] = pd.to_datetime(daywise_leads_df['CreatedAt'])
            daywise_leads_df['CreatedAt'] = daywise_leads_df['CreatedAt'].dt.tz_localize('UTC').dt.tz_convert(ist)
            daywise_leads_df['Day'] = daywise_leads_df['CreatedAt'].dt.date
            daywise_leads_df = daywise_leads_df.groupby('Day')['LeadId'].count().reset_index(name='Leads')
            category_spend['ReportDate'] = category_spend['ReportDate'].dt.date
            daywise_spend_df = category_spend.groupby('ReportDate')['Cost'].sum().reset_index(name='Spend')
            daywise_leads_df['CPL'] = daywise_spend_df['Spend']/daywise_leads_df['Leads']
            daywise_leads_df['Spend'] = daywise_spend_df['Spend']
            st.session_state['daywise_leads_df'] = daywise_leads_df
            st.session_state['new_daywise_leads_df'] = new_daywise_leads_df
            daywise_leads_df['Type'] = 'Total Leads'
            new_daywise_leads_df['Type'] = 'New Leads'
            
            
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=st.session_state['daywise_leads_df']['Day'],
                y=st.session_state['daywise_leads_df']['Leads'],
                mode='lines+markers',
                name='Total Leads'
            ))
            fig.add_trace(go.Scatter(
                x=st.session_state['daywise_leads_df']['Day'],
                y=st.session_state['daywise_leads_df']['CPL'],
                mode='lines+markers',
                name='CPL'
            ))
            fig.add_trace(go.Scatter(
                x=st.session_state['daywise_leads_df']['Day'],
                y=st.session_state['daywise_leads_df']['Spend'],
                mode='lines+markers',
                name='Spend'
            ))

            fig.add_trace(go.Scatter(
                x=st.session_state['new_daywise_leads_df']['Day'],
                y=st.session_state['new_daywise_leads_df']['Leads'],
                mode='lines+markers',
                name='New Leads'
            ))
            fig.update_layout(title=f"Leads and CPL Trend Over Days for {funnel}", xaxis_title='Day', yaxis_title=f'Number of Leads for {funnel}', template='plotly_dark')
            with cols[i]:
                st.plotly_chart(fig,use_container_width=True)
            daywise_leads_df = pd.DataFrame()
            new_daywise_leads_df = pd.DataFrame()
   




if side_radio == 'MasterClass Analysis':
    input_placeholder = st.empty()
    

    with input_placeholder.container():
        category = st.selectbox('Choose the Category',['Spirituality','Finance','Both'],None)
        Platform = st.selectbox('Choose the Platform',['Google','Facebook'],None)
        start_date = st.date_input('Choose the Starting Date',value=None,format = "YYYY/MM/DD",)
        end_date = st.date_input('Choose the Ending Date',value = None, format = "YYYY/MM/DD")
        get_value = st.button("Get Data",key="get_data_button_1")

    if get_value:
        lead_df = pd.DataFrame(execute_query(Leads_query(start_date, end_date, category)))
        joins_df = pd.DataFrame(execute_query(joins_query(start_date, end_date, category)))
        payment_df = pd.DataFrame(execute_query(payment_query(start_date, end_date)))
        lead_df.columns = ['Category', 'Source', "Comment",'LeadId', "UserId", "CreatedAt", "MasterClass",'UserCreated']
        joins_df.columns = ['Category', 'Source','Comment' ,'LeadId', "UserId", "CreatedAt", "MasterClass",'meetingId',"UserCreated"]
        payment_df.columns = ['PaymentId', 'Amount', 'Payment Category', "UserId", "Pay Created", "MC/BX Paid For"]
        payment_df['Amount'] = payment_df['Amount'].astype(int)
        listt = lead_df['MasterClass'].unique().tolist()
        MasterClass = st.selectbox('Choose the MasterClass',listt,None)
        mc_button = st.button("Select MasterClass")
        if mc_button:
            start_date = pd.to_datetime(start_date)
            lead_df_plt = lead_df[lead_df['Source'].str.lower() == Platform.lower() & lead_df['MasterClass'] == MasterClass]['MasterClass','Comment',"LeadId","UserId",]
            joins_df_plt = lead_df[lead_df['Source'].str.lower() == Platform.lower() & lead_df['MasterClass'] == MasterClass]
            leads_count = lead_df_plt['LeadId'].nunique()
            new_leads_count = lead_df_plt[lead_df_plt['UserCreated'] >= start_date]['LeadId'].nunique()
            user_count = lead_df_plt['UserId'].nunique()
            new_users_count = lead_df_plt[lead_df_plt['UserCreated'] >= start_date]['UserId'].nunique()
            join_count = joins_df_plt['UserId'].nunique()
            new_joins_count = joins_df_plt[joins_df_plt['UserCreated'] >= start_date]['UserId'].nunique()
            total_joins_count = joins_df_plt['meetingId'].nunique()
            new_total_joins_count = joins_df_plt[joins_df_plt['UserCreated'] >= start_date]['meetingId'].nunique()

            rev_raw_df = pd.merge(lead_df_plt, payment_df, on="UserId", how='inner')
            rev_raw_df.drop_duplicates()
            rev_df = rev_raw_df[rev_raw_df["Pay Created"] > rev_raw_df["CreatedAt"]]
            rev_df_final = rev_df[rev_df['Category'] == rev_df['Payment Category']]
            rev_corrected = rev_df_final[['Amount','PaymentId']].drop_duplicates()
            revenue = rev_corrected['Amount'].sum()
            converted = rev_corrected[rev_corrected["Amount"] > 299]['PaymentId'].count()

            daywise_leads_df = lead_df_plt[['LeadId', 'CreatedAt']].copy()
            daywise_leads_df['CreatedAt'] = pd.to_datetime(daywise_leads_df['CreatedAt'])
            daywise_leads_df['Day'] = daywise_leads_df['CreatedAt'].dt.date
            daywise_leads_df = daywise_leads_df.groupby('Day')['LeadId'].count().reset_index(name='Leads')

            # Save the summary table to session state!
            st.session_state['summary_table'] = pd.DataFrame({
                'Metric': ['Leads ', 'Users', 'Unique Joins','Joining Percentage', 'Total Joins','User Multiplier','Sales','Revenue'],
                'Count': [leads_count, user_count, join_count,join_count/user_count*100 ,total_joins_count,total_joins_count/join_count,converted,revenue],
                'New Count': [new_leads_count,new_users_count,new_joins_count,new_joins_count/new_users_count*100, new_total_joins_count,new_total_joins_count/new_joins_count,'Nan','Nan']
            })

            st.session_state['daywise_leads_df'] = daywise_leads_df
            input_placeholder.empty()

        
        if 'summary_table' in st.session_state:
            summary_table = st.session_state['summary_table']
            st.header(f"{category}")
            st.markdown(f"<h4 style='color:Green;'>{Platform} {Funnel} Funnel from {start_date} to {end_date}</h3>", unsafe_allow_html=True)
            st.table(summary_table)

            if st.button("Copy to Clipboard (Manual)"):
                csv_data = summary_table.to_csv(index=False)
                st.text_area("Copy the table below:", csv_data, height=200)

            
            fig = px.line(
                st.session_state['daywise_leads_df'],
                x='Day', y='Leads',
                title='Leads Trend Over Days',
                markers=True
            )
            fig.update_layout(xaxis_title='Day', yaxis_title='Number of Leads', template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

            data = dict(number=[summary_table.iloc[0,1], summary_table.iloc[2,1], summary_table.iloc[6,1]],
                        stage=['Leads', 'Joins', 'Sales'])
            fig2 = px.funnel(data, x='number', y='stage')
            st.plotly_chart(fig2, use_container_width=True)
            stages = ["Leads", "Joining", "Sales"]


if side_radio == 'Class spend':
    input_placeholder = st.empty()

    with input_placeholder.container():
        report_date = st.date_input('Choose the Report Date', value=None, format="YYYY/MM/DD")
        if st.button("Get Value"):
            st.session_state['get_value_clicked'] = True

    if st.session_state.get('get_value_clicked', False):
        class_df = pd.DataFrame(execute_query(classes(report_date)))
        class_df.columns = ['MasterClassId', "MasterClass", "Start_date", "end_date"]
        teacher = class_df["MasterClass"].tolist()
        input_placeholder = st.empty()  # Not sure you intended this here — check logic

        teacher[0] = 'All Teachers'
        teacher_list = st.selectbox('Select MasterClass', teacher, index=0)

        if st.button('Get Spend'):
            with st.spinner("Wait for it...",show_time=True):
                st.session_state['teacher_selected'] = teacher_list

                if st.session_state.get('teacher_selected', None):
                    teacher_list = st.session_state['teacher_selected']
                    if teacher_list == 'All Teachers':
                        class_df = class_df
                    else:
                        class_df = class_df[class_df['MasterClass'] == teacher_list]
                    start_date = class_df["Start_date"].min().date()
                    end_date = class_df["end_date"].max().date()
                    spend_df = pd.DataFrame(execute_query(ads_query(start_date,end_date)))
                    spend_df.columns = ['Platform', 'CampaignName', 'AdsetName', 'AdName', 'AdId', 'Clicks', 'Impressions', 'Reach','Cost', 'ReportDate']
                    spend_df['ReportDate'] = pd.to_datetime(spend_df['ReportDate'])
                    client = google_cred()
                    codes_data = client.open_by_url('https://docs.google.com/spreadsheets/d/1QkU0jaDC2TKg4n-JeIq9_expBzkNtiLT8kQZjHKIjlI/edit?usp=sharing')
                    mapping_codes = codes_data.worksheet("Codes_filtered").get_all_values()
                    mapping_codes = pd.DataFrame(mapping_codes)
                    mapping_codes.columns = ['Code','MasterClass']
                    Taboola_free_codes = codes_data.worksheet("Taboola_free_codes").get_all_values()
                    Taboola_codes = pd.DataFrame(Taboola_free_codes)
                    Taboola_codes.columns = ['Code','MasterClass']
                    merged_classes = pd.merge(class_df,mapping_codes,how="inner",on = 'MasterClass')
                    merged_classes['end_date_splitted'] = merged_classes['end_date'].dt.date
                    merged_classes['end_time'] = merged_classes['end_date'].dt.time
                    merged_classes['start_date_splitted'] = merged_classes['Start_date'].dt.date
                    merged_classes['Start_time'] = merged_classes['Start_date'].dt.time
                    merged_classes['Google Paid'] = 0.0
                    merged_classes['Facebook Paid'] = 0.0
                    merged_classes['Google Free'] = 0.0
                    merged_classes['Facebook Free'] = 0.0
                    merged_classes['Taboola1'] = 0.0

                    # Adjust spend function
                    def adjusted_spend(start_date, end_date, start_time, end_time, spend_mc):
                        # Filter records within date range
                        spend_filtered = spend_mc[
                            (spend_mc['ReportDate'].dt.date >= start_date) &
                            (spend_mc['ReportDate'].dt.date <= end_date)
                        ].copy()

                        if spend_filtered.empty:
                            return 0.0

                        # Adjust spend for start date
                        mask_start = spend_filtered['ReportDate'].dt.date == start_date
                        spend_filtered.loc[mask_start, 'Cost'] *= (24 - start_time.hour) / 24

                        # Adjust spend for end date
                        mask_end = spend_filtered['ReportDate'].dt.date == end_date
                        spend_filtered.loc[mask_end, 'Cost'] *= end_time.hour / 24

                        return spend_filtered['Cost'].sum()

                    # Process each class and compute spends
                    for i in range(len(merged_classes)):
                        code = merged_classes.loc[i, "Code"]
                        start_date = merged_classes.loc[i, 'start_date_splitted']
                        end_date = merged_classes.loc[i, 'end_date_splitted']
                        start_time = merged_classes.loc[i, 'Start_time']
                        end_time = merged_classes.loc[i, 'end_time']

                        spend_mc = spend_df[(spend_df['CampaignName'].str.contains(code, na=False, regex=False)) | spend_df['AdName'].str.contains(code, na=False, regex=False)] 

                        # Google Paid
                        google_paid = spend_mc[
                            (spend_mc['Platform'] == 'google') &
                        ( (spend_mc['CampaignName'].str.contains('paid', case=False, na=False))|(spend_mc['AdsetName'].str.contains('mxpayment', case=False, na=False)))
                        ]
                        if not google_paid.empty:
                            merged_classes.loc[i, 'Google Paid'] = adjusted_spend(start_date, end_date, start_time, end_time, google_paid)

                        # Facebook Paid
                        facebook_paid = spend_mc[
                            (spend_mc['Platform'] == 'facebook') &
                        ( (spend_mc['CampaignName'].str.contains('paid|conversion', case=False, na=False))|(spend_mc['AdsetName'].str.contains('mxpayment', case=False, na=False)))
                        ]
                        if not facebook_paid.empty:
                            merged_classes.loc[i, 'Facebook Paid'] = adjusted_spend(start_date, end_date, start_time, end_time, facebook_paid)

                        # Google Free
                        google_free = spend_mc[
                            (spend_mc['Platform'] == 'google') &
                            (~spend_mc['CampaignName'].str.contains('paid|conversion', case=False, na=False)) & (~spend_mc['AdsetName'].str.contains('mxpayment', case=False, na=False))
                        ]
                        if not google_free.empty:
                            merged_classes.loc[i, 'Google Free'] = adjusted_spend(start_date, end_date, start_time, end_time, google_free)

                        # Facebook Free
                        facebook_free = spend_mc[
                            (spend_mc['Platform'] == 'facebook') &
                        (~spend_mc['CampaignName'].str.contains('paid|conversion', case=False, na=False)) & (~spend_mc['AdsetName'].str.contains('mxpayment', case=False, na=False))
                        ]
                        if not facebook_free.empty:
                            merged_classes.loc[i, 'Facebook Free'] = adjusted_spend(start_date, end_date, start_time, end_time, facebook_free)
                        # Facebook Free
                        taboola = spend_mc[
                            (spend_mc['Platform'] == 'Taboola')
                        ]
                        if not taboola.empty:
                            merged_classes.loc[i, 'Taboola'] = adjusted_spend(start_date, end_date, start_time, end_time, taboola)
                    # taboola_merged = pd.merge(merged_classes, Taboola_codes, how="left", on="MasterClass", suffixes=('', '_Taboola'))
                    # taboola_merged['Taboola Spend'] = 0.0
                    # for i in range(len(taboola_merged)):
                    #     code_main = taboola_merged.loc[i, "Code"]
                    #     code_taboola = taboola_merged.loc[i, "Code_Taboola"]
                    #     start_date = taboola_merged.loc[i, 'start_date_splitted']
                    #     end_date = taboola_merged.loc[i, 'end_date_splitted']
                    #     start_time = taboola_merged.loc[i, 'Start_time']
                    #     end_time = taboola_merged.loc[i, 'end_time']

                    #     spend_mc = spend_df[spend_df['CampaignName'].str.contains(code_main, na=False, regex=False)]

                    #     # Existing Google and Facebook spend logic stays unchanged here...

                    #     # Now Taboola spend
                    #     if pd.notna(code_taboola):
                    #         taboola_spend = spend_df[
                    #             (spend_df['Platform'].str.lower() == 'taboola') &
                    #             (spend_df['CampaignName'].str.contains(code_taboola, na=False, regex=False))
                    #         ]
                    #         if not taboola_spend.empty:
                    #             taboola_merged.loc[i, 'Taboola Spend'] = adjusted_spend(start_date, end_date, start_time, end_time, taboola_spend)
                    # taboola_merged = taboola_merged[['MasterClass', "Start_date", "end_date", 
                    #                                 "Google Paid", "Google Free", 
                    #                                 "Facebook Paid", "Facebook Free", 
                    #                                 'Taboola Spend','Taboola1']]

                    # # Define the grouping columns (all except 'Taboola Spend')
                    # group_cols = ['MasterClass', "Start_date", "end_date", 
                    #             "Google Paid", "Google Free", "Facebook Paid", "Facebook Free",'Taboola1']

                    # # Group by those columns and sum Taboola Spend
                    # final_result = taboola_merged.groupby(group_cols, as_index=False)['Taboola Spend'].sum()
                    # # Second groupby: sum all spend columns by MasterClass and dates
                    final_spend_summary = merged_classes.groupby(
                        ['MasterClass', 'Start_date', 'end_date'],
                        as_index=False
                    ).agg({
                        'Google Paid': 'sum',
                        'Google Free': 'sum',
                        'Facebook Paid': 'sum',
                        'Facebook Free': 'sum',
                        'Taboola1': 'sum'
                    })
                    final_spend_summary[['Google Paid','Google Free','Facebook Paid','Facebook Free','Taboola1']] = final_spend_summary[['Google Paid','Google Free','Facebook Paid','Facebook Free','Taboola1']].astype(int)
                    st.dataframe(final_spend_summary)
            st.success("Done!")

if side_radio == 'Campaign Level':
    input_placeholder = st.empty()
    with input_placeholder.container():
        start_date = st.date_input('Choose the Starting Date', value=None, format="YYYY/MM/DD")
        end_date = st.date_input('Choose the Ending Date', value=None, format="YYYY/MM/DD")
        end_date2 = end_date
        ist = pytz.timezone('Asia/Kolkata')
        if start_date:
            start_date = start_date.strftime('%Y-%m-%d')
            start = datetime.strptime(start_date + " 00:00:00", "%Y-%m-%d %H:%M:%S")
            start_ist = ist.localize(start)
            start_utc = start_ist.astimezone(pytz.utc).strftime("%Y-%m-%d %H:%M:%S")
            print(start_utc)
        else:
             print("start_date is None or empty")
        if end_date:
            end_date = end_date + pd.Timedelta(days=1)
            end_date = end_date.strftime('%Y-%m-%d')
            end = datetime.strptime(end_date + " 00:00:00", "%Y-%m-%d %H:%M:%S")
            end_ist = ist.localize(end)
            end_utc = end_ist.astimezone(pytz.utc).strftime("%Y-%m-%d %H:%M:%S")
            print(end_utc)
        else:
             print("end_date is None or empty")
        get_value = st.button("Get Value")
    if get_value:
        queries = [
            Leads_query(start_utc, end_utc),
            joins_query(start_utc, end_utc),
            payment_query(start_utc, end_utc),
            ads_query2(start_date,end_date2)
        ]

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(execute_query, queries))

        lead_df = pd.DataFrame(results[0])
        joins_df = pd.DataFrame(results[1])
        payment_df = pd.DataFrame(results[2])
        spend_df = pd.DataFrame(results[3])

        lead_df.columns = ['Category', 'Source', 'Comment', 'LeadId', 'UserId', 'CreatedAt', 'Title', 'UserCreated']
        joins_df.columns = ['Category', 'Source', 'Comment', 'LeadId', 'joinUserId', 'CreatedAt', 'Title', 'meetingId', 'UserCreated']
        payment_df.columns = ['PaymentId', 'Amount', 'Payment Category', 'UserId', 'Pay Created', 'MC/BX Paid For']
        spend_df.columns = ['Platform', 'campaignName', 'adsetName', 'adName', 'AdId', 'Clicks', 'Impressions', 'Reach','spend', 'ReportDate','landingUrl']
        spend_df =  spend_df.replace(-1, 0)
        sum_columns = ['Clicks', 'Impressions', 'Reach', 'spend']
        spend_df = spend_df.groupby(
            ['Platform', 'campaignName', 'adName', 'AdId', 'adsetName', 'landingUrl'],
            as_index=False
        )[sum_columns].sum()
        payment_df['Amount'] = payment_df['Amount'].astype(int)
        lead_joins_merged = pd.merge(
            lead_df,
            joins_df[['joinUserId', 'meetingId']],
            left_on='UserId',
            right_on='joinUserId',
            how='left'
        )

        final_merged_df = pd.merge(
            lead_joins_merged,
            payment_df[['UserId', 'Amount', 'PaymentId', 'Payment Category', 'Pay Created', 'MC/BX Paid For']],
            on='UserId',
            how='left'
        )
        # final_merged_df = final_merged_df[final_merged_df['CreatedAt'] < final_merged_df['Pay Created'] ]
        # Filter only valid conversions (Amount > 299) for counting unique PaymentId
        converted_df = final_merged_df[final_merged_df['Amount'] > 299]

        # Filter unique payments for accurate Revenue sum
        unique_payments_df = final_merged_df.drop_duplicates(subset='PaymentId')

        # Group for Users and Joins
        grouped_base = final_merged_df.groupby(['Comment', 'Source', 'Category', 'Title'], as_index=False).agg(
            Users=('UserId', 'nunique'),
            Joins=('joinUserId', 'nunique')
        )

        # Group for Converted
        converted_group = converted_df.groupby(['Comment', 'Source', 'Category', 'Title'], as_index=False).agg(
            Converted=('PaymentId', 'nunique')
        )

        # Group for Revenue (unique PaymentId only)
        revenue_group = unique_payments_df.groupby(['Comment', 'Source', 'Category', 'Title'], as_index=False).agg(
            Revenue=('Amount', 'sum')
        )

        # Merge all grouped data
        grouped_df = grouped_base.merge(converted_group, on=['Comment', 'Source', 'Category', 'Title'], how='left')
        grouped_df = grouped_df.merge(revenue_group, on=['Comment', 'Source', 'Category', 'Title'], how='left')

        # Fill NaNs in Converted/Revenue (where there were no payments)
        grouped_df['Converted'] = grouped_df['Converted'].fillna(0).astype(int)
        grouped_df['Revenue'] = grouped_df['Revenue'].fillna(0)

        # Sort by Revenue
        grouped_df = grouped_df.sort_values(by='Revenue', ascending=False)
        rev_funnel = grouped_df.copy()
        spend_funnel = spend_df.copy()
        rev_funnel['Comment'] = rev_funnel['Comment'].str.replace(' ', '+', regex=False)
        rev_funnel['Comment'] = rev_funnel['Comment'].str.replace('$', '', regex=False)
        spend_funnel =  spend_funnel.replace(-1, 0)
        spend_funnel['campaignName'] = spend_funnel['campaignName'].str.replace('$', '', regex=False)
        spend_funnel['adName'] = spend_funnel['adName'].str.replace('$', '', regex=False)
        spend_funnel['adsetName'] = spend_funnel['adsetName'].str.replace('$', '', regex=False)
        spend_funnel['landingUrl'] = spend_funnel['landingUrl'].str.replace('$', '', regex=False)
        import urllib.parse
        def extract_comment_raw(url):
            try:
                qs = urllib.parse.urlparse(url).query
                for part in qs.split('&'):
                    if part.startswith('comment='):
                        return part[len('comment='):]  # raw, unparsed
                return None
            except:
                return None
        spend_funnel['landing_para'] = spend_funnel['landingUrl'].apply(extract_comment_raw)
        spend_funnel['landing_para'] = spend_funnel['landing_para'].str.replace('$', '', regex=False)
        spend_funnel = spend_funnel.fillna(0)
        spend_funnel[spend_funnel['adsetName'].str.contains('&',case=False,na = False)]['adsetName'].tolist()
        rev_funnel['Comment'] = rev_funnel['Comment'].str.split('@@').str[0]
        rev_funnel = rev_funnel.groupby(['Category','Title','Source','Comment'],    as_index=False
        ).sum(numeric_only=True)
        grouped_spend = spend_funnel.groupby(
            ['adName', 'AdId', 'Platform', 'adsetName', 'landing_para','campaignName','landingUrl'],
            as_index=False
        ).sum(numeric_only=True)
        import pandas as pd

        # Create matching keys
        grouped_spend_df = grouped_spend.copy()
        grouped_spend_df['key1'] = grouped_spend_df['campaignName'].astype(str) + grouped_spend_df['adName'].astype(str)
        grouped_spend_df['key2'] = grouped_spend_df['campaignName'].astype(str) + grouped_spend_df['adsetName'].astype(str) + grouped_spend_df['adName'].astype(str)

        # Initialize
        rev = rev_funnel.copy()
        matched = pd.DataFrame()
        unmatched = rev.copy()

        # Helper function for each matching step
        def match_step(df_unmatched, right_col):
            step = df_unmatched.merge(grouped_spend_df, left_on='Comment', right_on=right_col, how='left', indicator=True)
            matched_rows = step[step['_merge'] == 'both'].drop(columns=['_merge'])
            matched_rows = matched_rows.drop_duplicates(subset=['Comment'])  # Keep only first match per Comment
            remaining_rows = step[step['_merge'] == 'left_only'][['Comment']]  # Only keep Comment to match again
            return matched_rows, remaining_rows

        # Step 1: match on landing_para
        step1, rem1 = match_step(unmatched, 'landing_para')
        matched = pd.concat([matched, step1])
        unmatched = rev[rev['Comment'].isin(rem1['Comment'])]

        # Step 2: match on adId
        step2, rem2 = match_step(unmatched, 'AdId')
        matched = pd.concat([matched, step2])
        unmatched = rev[rev['Comment'].isin(rem2['Comment'])]

        # Step 3: match on adName
        step3, rem3 = match_step(unmatched, 'adName')
        matched = pd.concat([matched, step3])
        unmatched = rev[rev['Comment'].isin(rem3['Comment'])]

        # Step 4: match on key1
        step4, rem4 = match_step(unmatched, 'key1')
        matched = pd.concat([matched, step4])
        unmatched = rev[rev['Comment'].isin(rem4['Comment'])]

        # Step 5: match on key2
        step5, rem5 = match_step(unmatched, 'key2')
        matched = pd.concat([matched, step5])
        unmatched = rev[rev['Comment'].isin(rem5['Comment'])]

        # Final deduplication: one match per Comment
        final_joined = matched.drop_duplicates(subset=['Comment'])

        # Optional: Merge with rev_funnel to keep all 508 rows and fill NaNs for unmatched
        final_result = rev.merge(final_joined, on='Comment', how='left', suffixes=('', '_matched'))
        final_result['CPL'] = final_result['spend']/final_result['Users']
        final_result['ROI'] =((final_result['Revenue']/final_result['spend'])-1)
        final_result['Joining perc'] = (final_result['Joins']/final_result['Users'])
        final_result['Conversion percent'] = (final_result['Converted']/final_result['Joins'])
        final_result['Conversion percent'] = (final_result['Converted']/final_result['Joins'])


        mapped_camp = final_result[['Category','Source','Title','Comment','spend','Users','CPL','Joins','Joining perc','Converted','Conversion percent','Revenue','Clicks','Impressions']]
        search_campaigns = mapped_camp[(mapped_camp['Comment'].str.contains('twfu|tfal',case=False, na = False))&(mapped_camp['Comment'].str.contains('search',case=False, na = False)) ]
        grouped = search_campaigns.groupby(['Comment', 'Category', 'Source', 'Title'], as_index=False).agg({
            'spend': 'sum',
            'Impressions': 'first',
            'Clicks': 'first',
            'Users': 'sum',
            'Joins': 'sum',
            'Revenue': 'sum',
            'Converted': 'sum'
        })
        mapped_camp = mapped_camp[
            ~((mapped_camp['Comment'].str.contains('twfu|tfal', case=False, na=False)) &
            (mapped_camp['Comment'].str.contains('search', case=False, na=False)))]
        Corrected_spends = pd.concat([mapped_camp,grouped])
        Corrected_spends['CPM'] = (Corrected_spends['spend']/Corrected_spends['Impressions'])*1000
        Corrected_spends['CTR'] = (Corrected_spends['Clicks']/Corrected_spends['Impressions'])*100
        Corrected_spends['Fill Rate'] = (Corrected_spends['Users']/Corrected_spends['Clicks'])*100
        st.dataframe(Corrected_spends)    
        
