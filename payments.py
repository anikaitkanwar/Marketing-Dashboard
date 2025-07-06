def payment_query(start_date,end_date):
    query2 = f'''
   with bootcamp_payments AS (
    SELECT DISTINCT pi.id, pi.amount, pi."userId", pi."createdAt" as time,cat.name,b.title
    FROM "PaymentIntent" pi
    JOIN "Bootcamp" b ON b.id = pi."bootcampId"
    JOIN "User" u2 ON u2.id = pi."userId"
    JOIN "User" u ON b."teacherId" = u."id"        
    JOIN "Categories" cat ON u."categoryId" = cat."id"    
    WHERE pi."createdAt" BETWEEN '{start_date}' AND '{end_date}'
    AND pi.status = '1'
),
 
masterclass_payments AS (
    SELECT DISTINCT pi.id, pi.amount, pi."userId", pi."createdAt" as time,cat.name,mc.title
    FROM "PaymentIntent" pi
    JOIN "MasterClass" mc ON mc.id = pi."masterclassId"
    JOIN "User" u2 ON u2.id = pi."userId"
    JOIN "Bootcamp" b ON b.id = mc."bootcampId"
    JOIN "User" u ON b."teacherId" = u."id"        
    JOIN "Categories" cat ON u."categoryId" = cat."id"    
    WHERE pi."createdAt" BETWEEN '{start_date}' AND '{end_date}'
    AND pi.status = '1'
),

product_payments as (
        select
                distinct pi.id,
                pi.amount,
                pi."userId",
				pi."createdAt" as time,
                case 
                when
                cast(pi.amount as numeric) > 29900 then 'Spirituality'
                 else 'Finance' end as name,
				 pd."name" as title
				 
        from
                "PaymentIntent" pi
		Join "Products"pd on pd.id = pi."ProductId"
        where
                "bootcampId" is null and "masterclassId" is null
                and pi.status = '1'
                and  pi."createdAt" BETWEEN '{start_date}' AND '{end_date}'
),
unmapped_product_payments as (
        select
                distinct pi.id,
                pi.amount,
                pi."userId",
                pi."createdAt" as time,
                case
                when
                cast(pi.amount as numeric) > 29900 then 'Spirituality'
                 else 'Finance' end as name,  
                'Unmapped_product' as title

        from
                "PaymentIntent" pi
        where
                "bootcampId" is null and "masterclassId" is null
				and "ProductId" is null
                and pi.status = '1'
                and  pi."createdAt" BETWEEN '{start_date}' AND '{end_date}'
),


payments_filtered AS (
    SELECT DISTINCT pi.id, cast(pi.amount as numeric)/100,pi.name,pi."userId",pi.time,pi.title
    FROM bootcamp_payments pi
    UNION ALL
    SELECT DISTINCT mp.id, cast(mp.amount as numeric)/100,mp.name,mp."userId",mp.time,mp.title
    FROM masterclass_payments mp
    UNION ALL
    SELECT DISTINCT pd.id,cast(pd.amount as numeric)/100 ,pd.name,pd."userId",pd.time,pd.title
    from product_payments pd
    UNION ALL
    SELECT DISTINCT upd.id,cast(upd.amount as numeric)/100 ,upd.name,upd."userId",upd.time,upd.title
    from unmapped_product_payments upd

)


SELECT *
FROM payments_filtered;




'''
    return query2
