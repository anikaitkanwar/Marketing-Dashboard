def ads_query(start_date,end_date):
    query2 = f'''
    select distinct platform, "campaignName","adsetName","adName","adId","clicks","impressions","reach","spend","reportDate" from "AdStats"
where "reportDate" between '{start_date}' and '{end_date}'
'''
    return query2
def ads_query2(start_date,end_date):
    query2 = f'''
    select distinct platform, "campaignName","adsetName","adName","adId","clicks","impressions","reach","spend","reportDate","landingUrl" from "AdStats"
where "reportDate" between '{start_date}' and '{end_date}'
'''
    return query2



finance_keywords = [
    'hjst', 'kpsst', 'psst', 'cjot', 'gvlrs', 'dsaii', 'psis',
    'bkhgt', 'jsnit', 'akiob', 'jhft', 'mbcwc', 'amct', 'hspct',
    'asatp', 'kjamat', 'tw'
]

def categorize(row):
    campaign = str(row.get('CampaignName', '')).lower()
    ad_name = str(row.get('AdName', '')).lower()
    platform = str(row.get('Platform', '')).lower()

    # Priority condition for Taboola platform with 'al-' in AdName
    # Case 1: Platform is Taboola
    if platform == 'taboola':
        if 'al-' in ad_name:
            return 'Spirituality'
        else:
            return 'Finance'

    # Case 2: Platform is NOT Taboola and finance keywords exist
    elif any(keyword in campaign for keyword in finance_keywords) or \
         any(keyword in ad_name for keyword in finance_keywords):
        return 'Finance'

    # Default
    return 'Spirituality'
