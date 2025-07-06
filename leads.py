def Leads_query(start_date,end_date):
    query2 = f'''
    SELECT distinct cat.name,l.source,l.comment,l."id",l."userId",L."createdAt",mc.title,u2."createdAt"  
FROM "Leads" l  
JOIN "User"u2 on u2."id" = l."userId"              
Join "MasterClassSlots" mcs on l."masterclassSlotId" = mcs."id"      
Join "MasterClass" mc on mc."id" = mcs."masterClassId"        
Join "Bootcamp" b ON mc."bootcampId" = b."id"        
Join "User" u ON b."teacherId" = u."id"        
Join "Categories" cat ON u."categoryId" = cat."id"        
WHERE  L."createdAt" BETWEEN '{start_date}' AND '{end_date}'
AND l."source" NOT IN  ('ret', 'LMS', 'MGID', 'calendar', 'maha-shiv-puja', 'pankajD', 'arvind.tech', 'retdm', 'null', 'cal', 'dm', 'email', 'push-notification', 'sms', 'freshdm', 'retp', 'api-ops', 'act', 'Zoom Reschedule','Livekit_Reschedule_Feedback','Livekit_Reschedule_Post_Joining','Livekit_Reschedule')
AND l."source" NOT LIKE 'act%'
and l.source not ilike '%youtube%'
'''
    return query2
