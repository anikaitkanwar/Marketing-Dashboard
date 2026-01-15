def ads_query(start_date,end_date):
    query2 = f'''
     SELECT
    a.platform,
    a."campaignName",
    a."adName",
    a."adsetName",
    "adId",
    a.spend AS spends,
    a.impressions AS impressions,

        CASE
            WHEN a.platform = 'facebook' THEN a."inlineLinkClicks"
            ELSE a.clicks
        END
     AS clicks
FROM "AdStats" a
WHERE a."reportDate" BETWEEN '{start_date}' AND '{end_date}'
GROUP BY
    a.platform,
    a."campaignName",
    a."adName",
    a."adsetName",
    a."adId",
    a.spend,
    a.impressions,
    a."inlineLinkClicks",
    a.clicks
'''
    return query2

def ads_query2(start_date,end_date):
    query2 = f'''
     SELECT
    a.platform,
    a."campaignName",
    a."adName",
    a."adsetName",
    a."landingUrl",
    "adId",
    a.spend AS spends,
    a.impressions AS impressions,

        CASE
            WHEN a.platform = 'facebook' THEN a."inlineLinkClicks"
            ELSE a.clicks
        END
     AS clicks
FROM "AdStats" a
WHERE a."reportDate" BETWEEN '{start_date}' AND '{end_date}'
GROUP BY
    a.platform,
    a."campaignName",
    a."adName",
    a."adsetName",
    a."landingUrl",
    a."adId",
    a.spend,
    a.impressions,
    a."inlineLinkClicks",
    a.clicks
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
