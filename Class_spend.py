from classes import classes
import streamlit as st
from executer import execute_query
import pandas as pd 
from spend import ads_query
from classes import google_cred
side_radio = st.sidebar.radio("Choose any option",'Overall Analysis')
if side_radio == 'Overall Analysis':
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

                spend_mc = spend_df[spend_df['CampaignName'].str.contains(code, na=False, regex=False)]

                # Google Paid
                google_paid = spend_mc[
                    (spend_mc['Platform'] == 'google') &
                    (spend_mc['CampaignName'].str.contains('paid', case=False, na=False))
                ]
                if not google_paid.empty:
                    merged_classes.loc[i, 'Google Paid'] = adjusted_spend(start_date, end_date, start_time, end_time, google_paid)

                # Facebook Paid
                facebook_paid = spend_mc[
                    (spend_mc['Platform'] == 'facebook') &
                    (spend_mc['CampaignName'].str.contains('conversion', case=False, na=False))
                ]
                if not facebook_paid.empty:
                    merged_classes.loc[i, 'Facebook Paid'] = adjusted_spend(start_date, end_date, start_time, end_time, facebook_paid)

                # Google Free
                google_free = spend_mc[
                    (spend_mc['Platform'] == 'google') &
                    (~spend_mc['CampaignName'].str.contains('paid', case=False, na=False))
                ]
                if not google_free.empty:
                    merged_classes.loc[i, 'Google Free'] = adjusted_spend(start_date, end_date, start_time, end_time, google_free)

                # Facebook Free
                facebook_free = spend_mc[
                    (spend_mc['Platform'] == 'facebook') &
                    (~spend_mc['CampaignName'].str.contains('paid', case=False, na=False))
                ]
                if not facebook_free.empty:
                    merged_classes.loc[i, 'Facebook Free'] = adjusted_spend(start_date, end_date, start_time, end_time, facebook_free)
            

            taboola_merged = pd.merge(merged_classes, Taboola_codes, how="left", on="MasterClass", suffixes=('', '_Taboola'))
            taboola_merged['Taboola Spend'] = 0.0
            for i in range(len(taboola_merged)):
                code_main = taboola_merged.loc[i, "Code"]
                code_taboola = taboola_merged.loc[i, "Code_Taboola"]
                start_date = taboola_merged.loc[i, 'start_date_splitted']
                end_date = taboola_merged.loc[i, 'end_date_splitted']
                start_time = taboola_merged.loc[i, 'Start_time']
                end_time = taboola_merged.loc[i, 'end_time']

                spend_mc = spend_df[spend_df['CampaignName'].str.contains(code_main, na=False, regex=False)]

                # Existing Google and Facebook spend logic stays unchanged here...

                # Now Taboola spend
                if pd.notna(code_taboola):
                    taboola_spend = spend_df[
                        (spend_df['Platform'].str.lower() == 'taboola') &
                        (spend_df['CampaignName'].str.contains(code_taboola, na=False, regex=False))
                    ]
                    if not taboola_spend.empty:
                        taboola_merged.loc[i, 'Taboola Spend'] = adjusted_spend(start_date, end_date, start_time, end_time, taboola_spend)
            taboola_merged = taboola_merged[['MasterClass', "Start_date", "end_date", 
                                            "Google Paid", "Google Free", 
                                            "Facebook Paid", "Facebook Free", 
                                            'Taboola Spend']]

            # Define the grouping columns (all except 'Taboola Spend')
            group_cols = ['MasterClass', "Start_date", "end_date", 
                        "Google Paid", "Google Free", "Facebook Paid", "Facebook Free"]

            # Group by those columns and sum Taboola Spend
            final_result = taboola_merged.groupby(group_cols, as_index=False)['Taboola Spend'].sum()
            # Second groupby: sum all spend columns by MasterClass and dates
            final_spend_summary = final_result.groupby(
                ['MasterClass', 'Start_date', 'end_date','Taboola Spend'],
                as_index=False
            ).agg({
                'Google Paid': 'sum',
                'Google Free': 'sum',
                'Facebook Paid': 'sum',
                'Facebook Free': 'sum',
            })
            final_spend_summary[['Google Paid', 'Google Free','Facebook Paid','Facebook Free','Taboola Spend']] = final_spend_summary[['Google Paid', 'Google Free','Facebook Paid','Facebook Free','Taboola Spend']].astype(int)
            st.table(final_spend_summary)
