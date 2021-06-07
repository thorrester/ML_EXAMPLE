CREATE OR REPLACE TABLE `{staging_project}.{staging_dataset}.EMPTY_PKGS`
PARTITION BY fscl_WK_END_DT
OPTIONS(expiration_timestamp=TIMESTAMP(DATE_ADD(CURRENT_DATE, INTERVAL 3 DAY)))
AS(
SELECT
FSCL_WK_END_DT
, DTL. STR_NBR
, DTL. SKU_NBR
, DTL. SKU_CRT_DT
, CONCAT(CAST(DTL. SKU_NBR AS STRING),'-', CAST(DTL. SKU_CRT_DT AS STRING), '-', DTL. STR_NBR) AS SKU_STR_ID
, ROUND(AVG(skustr.curr_retl_amt),2) as avg_curr_retl_amt
, ROUND(SUM(dtl.unt_cnt),2) as emp_pkg_unt_qty
, ROUND(SUM(dtl.unt_cnt * skustr.curr_retl_amt),2) as emp_pkg_retl_amt
FROM `{edw_project}.{edw_dataset_shared}.STR_AUDTD_MUMD` as MUMD
INNER JOIN `{edw_project}.{edw_dataset_shared}.STR_AUDTD_MUMD_DTL` as DTL
		ON mumd.str_nbr = dtl.str_nbr
		AND mumd.fscl_yr_wk_key_val = dtl.fscl_yr_wk_key_val
		AND mumd.mkup_mkdn_ctrl_nbr = dtl.mkup_mkdn_ctrl_nbr
INNER JOIN `{edw_project}.{edw_dataset_shared}.STR_HIER` as STR
		ON mumd.str_nbr = str.str_nbr
INNER JOIN `{edw_project}.{edw_dataset_shared}.CAL_PRD_HIER` as CAL
		ON mumd.bus_dt = cal.cal_dt
LEFT JOIN `{edw_project}.{edw_dataset_shared}.SKU_HIER` as SKU
		ON dtl.sku_nbr = sku.sku_nbr
		AND dtl.sku_crt_dt = sku.sku_crt_dt
LEFT JOIN `{edw_project}.{edw_dataset_shared}.SKU_STR` as SKUSTR
		ON str.str_nbr = skustr.str_nbr
		AND sku.sku_nbr = skustr.sku_nbr
		AND sku.sku_crt_dt = skustr.sku_crt_dt
		AND cal.cal_dt >= skustr.eff_bgn_dt
		AND cal.cal_dt <= skustr.eff_end_dt
WHERE
	1=1
	AND SUBSTR(trim(dtl.mkdn_rsn_txt),1,2) = '04'
	AND str.str.str_cls_dt >= '2015-02-02'
	AND CAL_DT >= DATE_SUB(CURRENT_DATE, INTERVAL 104 WEEK)
GROUP BY
	1,2,3,4
);