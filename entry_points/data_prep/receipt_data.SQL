CREATE OR REPLACE TABLE `{staging_project}.{staging_dataset}.RECEIPT_DATA`
PARTITION BY RCVD_DT
CLUSTER BY SKU_NBR, STR_NBR
AS(
SELECT
  DISTINCT
  A.RCVD_DT
, E.SKU_NBR
, E. SKU_CRT_DT
, C.STR_NBR
, CONCAT(CAST(E. SKU_NBR AS STRING),'-', CAST(E. SKU_CRT_DT AS STRING), '-', C. STR_NBR) AS SKU_STR_ID
#, A.RECV_LOC_NBR
#, A.RECV_CTRL_NBR
#, A.ORD_CRT_DT
#, A.SHPG_DT
#, A.DTL_RCVD_DT
#, A.RECV_TYP_DESC
#, A.CNSGNMNT_FLG
#, A.PO_NBR
#, A.XFER_CTRL_NBR
, A.MVNDR_NBR
#, A.MER_DEPT_NBR
#, A.FRT_BILL_NBR
#, A.SND_LOC_NBR
#, A.SND_LOC_TYP_CD
#, A.MAN_KEY_PO_FLG
#, A.CERT_RECV_FLG
#, A.CERT_DTL_RECV_FLG
, A.AUDTD_FLG
#, A.RCVD_BY_USER_ID
, B.CURR_COST_AMT
#, B.CURR_RETL_AMT
, B.ORD_QTY
, B.SHPD_QTY
, B.RCVD_QTY
, CASE WHEN CAST(A. DSVC_TYP_CD AS INT64) = 1 THEN 'DTS'
       WHEN CAST(A. DSVC_TYP_CD AS INT64) = 2 THEN 'RDC'
       WHEN CAST(A. DSVC_TYP_CD AS INT64) = 3 THEN 'RDCXD'
       WHEN CAST(A. DSVC_TYP_CD AS INT64) = 4 THEN 'SDCD'
       WHEN CAST(A. DSVC_TYP_CD AS INT64) = 5 THEN 'BDC'
       WHEN CAST(A. DSVC_TYP_CD AS INT64) = 6 THEN 'SDC'
       ELSE 'UNKNOWN' END FUFILLMNT_PATH

#, A.ASN_DOC_TRC_ID
FROM `{edw_project}.{edw_dataset_schn_actvy}.STR_RECPT` A

INNER JOIN `{edw_project}.{edw_dataset_schn_actvy}.STR_RECPT_SKU` B
  ON A.recv_loc_nbr = B.recv_loc_nbr
  AND A.recv_ctrl_nbr = B.recv_ctrl_nbr
  AND A.rcvd_dt = B.rcvd_dt

INNER JOIN `{edw_project}.{edw_dataset_shared}.STR_HIER` as C
  ON A.recv_loc_nbr = C.str_nbr

INNER JOIN `{edw_project}.{edw_dataset_shared}.CAL_PRD_HIER` as D
  ON A.rcvd_dt = D.cal_dt

LEFT JOIN `{edw_project}.{edw_dataset_shared}.SKU_HIER` as E
  ON B.sku_nbr = E.sku_nbr
  AND B.sku_crt_dt = E.sku_crt_dt

WHERE  1=1
    AND A. RCVD_DT > DATE_SUB(CURRENT_DATE, INTERVAL 104 WEEK)
    AND FRT_BILL_NBR IS NOT NULL
    AND A. ORD_CRT_DT IS NOT NULL
    AND A. recv_typ_cd = 9
    AND C. div.div_nbr IN ('0001','0002','0004')
    AND C. str.str_typ_cd = 1
    AND SAFE_DIVIDE(B. RCVD_QTY, B. SHPD_QTY) < 5
);