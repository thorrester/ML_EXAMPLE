CREATE OR REPLACE TABLE `{staging_project}.{staging_dataset}.PHYS_INV_ADJUSTED_OH_HIST`
PARTITION BY PHY_INV_DT
CLUSTER BY SKU_NBR, STR_NBR
OPTIONS(expiration_timestamp=TIMESTAMP(DATE_ADD(CURRENT_DATE, INTERVAL 7 DAY)))
AS(

WITH STAGE01 AS(
SELECT
  FSCL_YR
, FSCL_WK_END_DT
, PHY_INV_DT
, PREDICTION_DATE
, SKU_NBR
, SKU_CRT_DT
, STR_NBR
, SKU_STR_ID
, CURR_RETL_AMT
, BINV_OH_QTY
, AINV_OH_QTY
, RTN_UNTS
, UNT_SLS
, SLS_ON_PHYS_INV
, RCVD_ON_PHYS_INV
, RCVD_QTY
, ADJUSTED_1_DAY_OH
, COALESCE((ADJUSTED_1_DAY_OH + UNT_SLS - RTN_UNTS - RCVD_QTY), AINV_OH_QTY) ADJUSTED_OH       -- IF PRESENT, USE 1 DAY POST ADJ WITH SLS ADDED AND RETURNS AND RECIEVED QTY SUBTRACTED
, CASE WHEN ADJUSTED_1_DAY_OH IS NOT NULL THEN 1 ELSE 0 END OH_ADJ_1_DAY_FLG
FROM(
    SELECT
    EXTRACT(YEAR FROM A. FSCL_WK_END_DT) FSCL_YR
    , A. FSCL_WK_END_DT
    , A. PHY_INV_DT
    , A. PREDICTION_DATE
    , A. SKU_NBR
    , A. SKU_CRT_DT
    , A. STR_NBR
    , A. SKU_STR_ID
    , A. CURR_RETL_AMT
    , A. BINV_OH_QTY --, CASE WHEN A. BINV_OH_QTY <= 0 THEN 0 ELSE A. BINV_OH_QTY END BINV_OH_QTY                 -- ADJUST NEGATIVE INVENTORY TO ZERO
    , AINV_OH_QTY --CASE WHEN A. AINV_OH_QTY <= 0 THEN 0 ELSE A. AINV_OH_QTY END AINV_OH_QTY    -- ADJUST NEGATIVE INVENTORY TO ZERO
    #, INV_OH_POSITION
    , COALESCE(SUM( D. END_OH_QTY), NULL) ADJUSTED_1_DAY_OH
    , COALESCE(SUM(ABS(B. GROSS_RTN_UNTS)),0) RTN_UNTS
    , COALESCE(SUM(B. GROSS_UNT_SLS),0) UNT_SLS  -- TRY TAKING OUT RETURN UNITS AT SOME POINT
    , COALESCE(SUM(C. RCVD_QTY), 0) RCVD_QTY
    , CASE WHEN SUM(CASE WHEN PHY_INV_DT = SLS_DT THEN  B. GROSS_UNT_SLS ELSE NULL END) > 0 THEN 1 ELSE 0 END SLS_ON_PHYS_INV
    , CASE WHEN SUM(CASE WHEN PHY_INV_DT = RCVD_DT THEN  c. RCVD_QTY ELSE NULL END) > 0 THEN 1 ELSE 0 END RCVD_ON_PHYS_INV
    FROM `{staging_project}.{staging_dataset}.PHYS_INV_HIST` A
    LEFT JOIN `{staging_project}.{staging_dataset}.SKU_STR_SALES_HIST` B
      ON B. SLS_DT BETWEEN A. PHY_INV_DT AND DATE_ADD(A. PHY_INV_DT, INTERVAL 1 DAY)
      AND A. SKU_STR_ID = B. SKU_STR_ID
    LEFT JOIN (
               SELECT
                 RCVD_DT
               , SKU_NBR
               , SKU_CRT_DT
               , STR_NBR
               , SKU_STR_ID
               , SUM(RCVD_QTY) RCVD_QTY
               FROM `{staging_project}.{staging_dataset}.RECEIPT_DATA`
               GROUP BY
                   RCVD_DT
                 , SKU_NBR
                 , SKU_CRT_DT
                 , STR_NBR
                 , SKU_STR_ID
              ) C
        ON C. RCVD_DT BETWEEN A. PHY_INV_DT AND DATE_ADD(A. PHY_INV_DT, INTERVAL 1 DAY)
        AND A. SKU_STR_ID = C. SKU_STR_ID

    LEFT JOIN `{staging_project}.{staging_dataset}.OH_CHGS` D
        ON D. OH_CHG_DT = DATE_ADD(A. PHY_INV_DT, INTERVAL 1 DAY)
        AND A. SKU_NBR = D. SKU_NBR
        AND A. SKU_CRT_DT = D. SKU_CRT_DT
        AND A. STR_NBR = D. STR_NBR

    WHERE 1=1
        --AND A. ZERO_FLG = 0
        AND NULL_FLG = 0
    GROUP BY 1,2,3,4,5,6,7,8,9,10,11
    )
  #WHERE SLS_ON_PHYS_INV = 0
 ),

STAGE02 AS(
SELECT
GENERATE_UUID() ID
, FSCL_YR
, FSCL_WK_END_DT
, PHY_INV_DT
, PREDICTION_DATE
, SKU_NBR
, SKU_CRT_DT
, STR_NBR
, SKU_STR_ID
, CURR_RETL_AMT
, RTN_UNTS
, UNT_SLS
, SLS_ON_PHYS_INV
, RCVD_ON_PHYS_INV
, RCVD_QTY
, BINV_OH_QTY
, AINV_OH_QTY PREV_AINV_OH_QTY
, ADJUSTED_AINV_OH_QTY AS AINV_OH_QTY
, CASE WHEN ADJUSTED_AINV_OH_QTY < BINV_OH_QTY THEN 'OVERSTATED'
       WHEN ADJUSTED_AINV_OH_QTY > BINV_OH_QTY THEN 'UNDERSTATED'
       WHEN ADJUSTED_AINV_OH_QTY = BINV_OH_QTY THEN 'CORRECT'
       END INV_OH_POSITION
, ADJUSTED_AINV_OH_QTY - BINV_OH_QTY INV_COUNT_DIFF
, OH_ADJ_1_DAY_FLG
, AINV_POS_NEG_FLG
FROM (
      SELECT
        FSCL_YR
      , FSCL_WK_END_DT
      , PHY_INV_DT
      , PREDICTION_DATE
      , SKU_NBR
      , SKU_CRT_DT
      , STR_NBR
      , SKU_STR_ID
      , CURR_RETL_AMT
      , BINV_OH_QTY
      , AINV_OH_QTY
      , RTN_UNTS
      , UNT_SLS
      , SLS_ON_PHYS_INV
      , RCVD_ON_PHYS_INV
      , RCVD_QTY
      , ADJUSTED_1_DAY_OH
      , ADJUSTED_OH
      , OH_ADJ_1_DAY_FLG
      , CASE WHEN                                                                                 -- default to physical inventory value if post adjustment is less than zero
            (CASE WHEN ADJUSTED_OH <= 0 THEN AINV_OH_QTY ELSE ADJUSTED_OH END) <= 0 THEN 0 ELSE   -- if both are less than or equal to zero then zero out
            (CASE WHEN ADJUSTED_OH <= 0 THEN AINV_OH_QTY ELSE ADJUSTED_OH END)
            END ADJUSTED_AINV_OH_QTY
      , CASE WHEN ADJUSTED_OH <= 0 AND AINV_OH_QTY > 0 THEN 1 ELSE 0 END AINV_POS_NEG_FLG
      FROM STAGE01
      )  A
),

INV_LOCATION AS(
    SELECT
      B. PHY_INV_DT
    , A. FSCL_YR
    , A. STR_NBR
    , SKU_NBR
    , SKU_CRT_DT
    , A. STR_NBR
    , SKU_CRT_DT
    , CONCAT(CAST(SKU_NBR AS STRING),'-', CAST(SKU_CRT_DT AS STRING), '-', A. STR_NBR) AS SKU_STR_ID
    , SUM(CASE WHEN IN_STR_LOC_TXT IS NULL THEN A.PHY_INV_UNT_CNT ELSE 0 END) UNKNOWN_LOC_INV
    , SUM(CASE WHEN SAFE_CAST(A.IN_STR_LOC_TXT AS INT64) >= 1000 AND SAFE_CAST(A.IN_STR_LOC_TXT AS INT64) <= 2999 THEN A.PHY_INV_UNT_CNT ELSE 0 END) OVERHEAD_INV
    , SUM(CASE WHEN SAFE_CAST(A.IN_STR_LOC_TXT AS INT64) >= 3000 AND SAFE_CAST(A.IN_STR_LOC_TXT AS INT64) <= 4599 THEN A.PHY_INV_UNT_CNT ELSE 0 END) SALES_FLR_INV
    , SUM(CASE WHEN SAFE_CAST(A.IN_STR_LOC_TXT AS INT64) >= 4600 AND SAFE_CAST(A.IN_STR_LOC_TXT AS INT64) <= 5999 THEN A.PHY_INV_UNT_CNT ELSE 0 END) SPECIALTY_AREA_INV
    , SUM(A.PHY_INV_UNT_CNT) AS UNT_CNT
    FROM `{edw_project}.{edw_dataset_schn_inv}.STR_SKU_INV_CNT` AS A
    INNER JOIN `{edw_project}.{edw_dataset_fin_standard}.PHY_INV_SCH` B
      ON A. PHY_INV_CYC_ID = B. PHY_INV_CYC_ID
      AND A. STR_NBR = B. STR_NBR
      AND A. FSCL_YR = B. FSCL_YR
    WHERE PHY_INV_DT >= DATE_SUB(CURRENT_DATE, INTERVAL 104 WEEK)
    --AND A. STR_NBR = '0111'
    GROUP BY 1,2,3,4,5
)

SELECT
 A.*
, SUB_DEPT_NBR
, EXT_CLASS_NBR
, CLASS_DESC
, EXT_SUB_CLASS_NBR
, SUB_CLASS_DESC
, EXT_SUB_SC_NBR
, CASE WHEN AINV_OH_QTY = 0 THEN 0 ELSE B. UNKNOWN_LOC_INV END UNKNOWN_LOC_INV
, CASE WHEN AINV_OH_QTY = 0 THEN 0 ELSE B. OVERHEAD_INV END    OVERHEAD_INV
, CASE WHEN (UNKNOWN_LOC_INV + OVERHEAD_INV + SPECIALTY_AREA_INV) = 0 AND AINV_OH_QTY < B. SALES_FLR_INV THEN AINV_OH_QTY
       WHEN AINV_OH_QTY = 0 THEN 0
       ELSE B. SALES_FLR_INV
       END SALES_FLR_INV
, CASE WHEN AINV_OH_QTY = 0 THEN 0 ELSE  B. SPECIALTY_AREA_INV END SPECIALTY_AREA_INV
, CASE WHEN AINV_OH_QTY = 0  THEN 1 ELSE 0 END OOS_IND
, CASE WHEN AINV_OH_QTY >0 AND SALES_FLR_INV = 0 THEN 1 ELSE 0 END SHELF_OUT
, CASE WHEN AINV_OH_QTY >0 AND SALES_FLR_INV > 0 AND SALES_FLR_INV <= 2 THEN 1 ELSE 0 END SHELF_LOW
, CASE WHEN INV_OH_POSITION = 'CORRECT' THEN 0
       WHEN INV_OH_POSITION = 'UNDERSTATED' THEN 1
       WHEN INV_OH_POSITION = 'OVERSTATED' THEN 1
       END BINARY_OH_LABEL
, CASE WHEN AINV_OH_QTY = 0 THEN 0
       WHEN AINV_OH_QTY >0 AND SALES_FLR_INV = 0 THEN 1
       WHEN AINV_OH_QTY >0 AND SALES_FLR_INV > 0 AND SALES_FLR_INV <= 2 THEN 2
       ELSE 3
       END MULTI_CLASS_LABEL
, CASE WHEN AINV_OH_QTY > 0 THEN LOG(AINV_OH_QTY*CURR_RETL_AMT) ELSE 0 END LOG_INVENTORY_DOLLARS
FROM STAGE02 A
LEFT JOIN INV_LOCATION B
  ON A. PHY_INV_DT = B. PHY_INV_DT
  AND A. SKU_STR_ID = B. SKU_STR_ID
INNER JOIN `{staging_project}.{staging_dataset}.THD_SKU_ACTIVE` C
  ON A. SKU_STR_ID = C. SKU_STR_ID
);