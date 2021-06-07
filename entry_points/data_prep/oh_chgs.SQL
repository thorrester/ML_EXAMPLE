CREATE OR REPLACE TABLE `{staging_project}.{staging_dataset}.OH_CHGS`
PARTITION BY OH_CHG_DT
CLUSTER BY SKU_NBR, STR_NBR
AS(
SELECT *
FROM
  (
     SELECT
      STR_NBR
      , SKU_NBR
      , SKU_CRT_DT
      , CONCAT(CAST(SKU_NBR AS STRING),'-', CAST(SKU_CRT_DT AS STRING), '-', STR_NBR) AS SKU_STR_ID
      , BGN_OH_QTY
      , END_OH_QTY
      , CNT_CMPLT_DT as OH_CHG_DT
      , rank() over(partition by str_nbr, sku_nbr, sku_crt_dt,CNT_CMPLT_DT order by AUD_CRT_TS desc) AS RNK
    FROM `{edw_project}.{edw_dataset_schn_inv}.WH_W_STRSK_RCNT`
    WHERE CNT_CMPLT_DT >= DATE_SUB(CURRENT_DATE, INTERVAL 104 WEEK)
  )
  WHERE RNK = 1
);