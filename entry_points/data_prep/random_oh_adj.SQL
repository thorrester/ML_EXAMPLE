CREATE OR REPLACE TABLE `{staging_project}.{staging_dataset}.RANDOM_OH_ADJ`
PARTITION BY FSCL_WK_END_DT
AS (

WITH MAX_DATE AS(
SELECT
MAX(PHY_INV_DT) MAX_DATE
FROM `{staging_project}.{staging_dataset}.PHYS_INV_ADJUSTED_OH_HIST`
),

TABLE_ AS(
SELECT
ID
, fscl_WK_END_DT
#, EXT_CLASS_NBR
, SKU_NBR
, STR_NBR
, CASE WHEN DATE_DIFF(B.MAX_DATE, PHY_INV_DT, DAY) <= 6 THEN '1_WEEK'
       WHEN DATE_DIFF(B.MAX_DATE, PHY_INV_DT, DAY) BETWEEN 7 AND 84 THEN '3_MONTH'
       WHEN DATE_DIFF(B.MAX_DATE, PHY_INV_DT, DAY) BETWEEN 85 AND 252 THEN '3-9_MONTH'
       ELSE '>9_MONTH'
       END WINDOW_
FROM `{staging_project}.{staging_dataset}.PHYS_INV_ADJUSTED_OH_HIST` A, MAX_DATE B
WHERE 1=1
    AND SUB_DEPT_NBR <> '023F'
    AND SLS_ON_PHYS_INV = 0
    AND RCVD_ON_PHYS_INV = 0
)


SELECT
  GENERATE_UUID() ID
, WINDOW_
, FSCL_WK_END_DT
, DATE_SUB(PHY_INV_DT, INTERVAL 1 DAY) PREDICTION_DATE
, SKU_NBR
, SKU_CRT_DT
, STR_NBR
, SKU_STR_ID
, CURR_RETL_AMT
#, AINV_OH_QTY
#, INV_COUNT_DIFF
#, INV_OH_POSITION
, LOG_INVENTORY_DOLLARS
, OOS_IND
, SHELF_OUT
, SHELF_LOW
, BINARY_OH_LABEL
, MULTI_CLASS_LABEL
FROM `{staging_project}.{staging_dataset}.PHYS_INV_ADJUSTED_OH_HIST` A
INNER JOIN (
          SELECT
          WINDOW_
          , ID
          , CASE WHEN WINDOW_ = '1_WEEK'    AND SEQ <= 1000000 THEN 1
                 WHEN WINDOW_ = '3_MONTH'   AND SEQ <= 3000000 THEN 1
                 WHEN WINDOW_ = '3-9_MONTH' AND SEQ <= 2000000 THEN 1
                 WHEN WINDOW_ = '>9_MONTH'  AND SEQ <= 1500000 THEN 1
                 ELSE 0 END KEEP_FLG
          FROM(
              SELECT
              WINDOW_
              , ID
              , SUM(1) OVER(PARTITION BY WINDOW_ ORDER BY RAND()) SEQ
              FROM(
                  SELECT
                  WINDOW_
                  , SKU_NBR
                  , ARRAY_AGG(ID ORDER BY RAND() LIMIT 50) IDS
                  FROM TABLE_
                  GROUP BY 1,2
               ), UNNEST(IDS) ID
               group by 1,2
              )
          ) B
   ON A. ID = B. ID
   WHERE 1=1
    AND KEEP_FLG = 1
);