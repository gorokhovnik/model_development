DROP TABLE Modeling.dbo.bki_var_CNTp_sum_TO_Overview_RATIO
SELECT TOP 0 ove.application_id,
CNTp_PFD01_PFP00_sum_CFO00_CFT00 * 1.0 / NULLIF(inquiries_total, 0) AS CNTp_PFD01_PFP00_sum_CFO00_CFT00_TO_inquiries_total_RATIO,
CNTp_PFD02_PFP00_sum_CFO00_CFT00 * 1.0 / NULLIF(inquiries_total, 0) AS CNTp_PFD02_PFP00_sum_CFO00_CFT00_TO_inquiries_total_RATIO,
CNTp_PFD03_PFP00_sum_CFO00_CFT00 * 1.0 / NULLIF(inquiries_total, 0) AS CNTp_PFD03_PFP00_sum_CFO00_CFT00_TO_inquiries_total_RATIO,
CNTp_PFD04_PFP00_sum_CFO00_CFT00 * 1.0 / NULLIF(inquiries_total, 0) AS CNTp_PFD04_PFP00_sum_CFO00_CFT00_TO_inquiries_total_RATIO,
CNTp_PFD00_PFP00_sum_CFO00_CFT00 * 1.0 / NULLIF(inquiries_total, 0) AS CNTp_PFD00_PFP00_sum_CFO00_CFT00_TO_inquiries_total_RATIO
INTO Modeling.dbo.bki_var_CNTp_sum_TO_Overview_RATIO
FROM Modeling.dbo.bki_var_overview AS ove
LEFT JOIN Modeling.dbo.bki_var_CNTp_sum AS CNTp_sum
ON ove.application_id = CNTp_sum.loan_application_id



INSERT INTO Modeling.dbo.bki_var_CNTp_sum_TO_Overview_RATIO
SELECT ove.application_id,
CNTp_PFD01_PFP00_sum_CFO00_CFT00 * 1.0 / NULLIF(inquiries_total, 0) AS CNTp_PFD01_PFP00_sum_CFO00_CFT00_TO_inquiries_total_RATIO,
CNTp_PFD02_PFP00_sum_CFO00_CFT00 * 1.0 / NULLIF(inquiries_total, 0) AS CNTp_PFD02_PFP00_sum_CFO00_CFT00_TO_inquiries_total_RATIO,
CNTp_PFD03_PFP00_sum_CFO00_CFT00 * 1.0 / NULLIF(inquiries_total, 0) AS CNTp_PFD03_PFP00_sum_CFO00_CFT00_TO_inquiries_total_RATIO,
CNTp_PFD04_PFP00_sum_CFO00_CFT00 * 1.0 / NULLIF(inquiries_total, 0) AS CNTp_PFD04_PFP00_sum_CFO00_CFT00_TO_inquiries_total_RATIO,
CNTp_PFD00_PFP00_sum_CFO00_CFT00 * 1.0 / NULLIF(inquiries_total, 0) AS CNTp_PFD00_PFP00_sum_CFO00_CFT00_TO_inquiries_total_RATIO
FROM Modeling.dbo.bki_var_overview AS ove
LEFT JOIN Modeling.dbo.bki_var_CNTp_sum AS CNTp_sum
ON ove.application_id = CNTp_sum.loan_application_id
ALTER TABLE Modeling.dbo.bki_var_CNTp_sum_TO_Overview_RATIO
REBUILD PARTITION = ALL WITH (DATA_COMPRESSION = PAGE)







DROP TABLE Modeling.dbo.bki_var_CNTp_sum_TO_CNT_RATIO
SELECT TOP 0 CNTp_sum.loan_application_id,
CNTp_PFD01_PFP00_sum_CFO00_CFT00 * 1.0 / NULLIF(CNT_CFO00, 0) AS CNTp_PFD01_PFP00_sum_CFO00_CFT00_TO_CNT_CFO00_RATIO,
CNTp_PFD02_PFP00_sum_CFO00_CFT00 * 1.0 / NULLIF(CNT_CFO00, 0) AS CNTp_PFD02_PFP00_sum_CFO00_CFT00_TO_CNT_CFO00_RATIO,
CNTp_PFD03_PFP00_sum_CFO00_CFT00 * 1.0 / NULLIF(CNT_CFO00, 0) AS CNTp_PFD03_PFP00_sum_CFO00_CFT00_TO_CNT_CFO00_RATIO,
CNTp_PFD04_PFP00_sum_CFO00_CFT00 * 1.0 / NULLIF(CNT_CFO00, 0) AS CNTp_PFD04_PFP00_sum_CFO00_CFT00_TO_CNT_CFO00_RATIO,
CNTp_PFD00_PFP00_sum_CFO00_CFT00 * 1.0 / NULLIF(CNT_CFO00, 0) AS CNTp_PFD00_PFP00_sum_CFO00_CFT00_TO_CNT_CFO00_RATIO
INTO Modeling.dbo.bki_var_CNTp_sum_TO_CNT_RATIO
FROM Modeling.dbo.bki_var_CNTp_sum AS CNTp_sum
LEFT JOIN Modeling.dbo.bki_var_CNT AS CNT
ON CNTp_sum.loan_application_id = CNT.loan_application_id



INSERT INTO Modeling.dbo.bki_var_CNTp_sum_TO_CNT_RATIO
SELECT CNTp_sum.loan_application_id,
CNTp_PFD01_PFP00_sum_CFO00_CFT00 * 1.0 / NULLIF(CNT_CFO00, 0) AS CNTp_PFD01_PFP00_sum_CFO00_CFT00_TO_CNT_CFO00_RATIO,
CNTp_PFD02_PFP00_sum_CFO00_CFT00 * 1.0 / NULLIF(CNT_CFO00, 0) AS CNTp_PFD02_PFP00_sum_CFO00_CFT00_TO_CNT_CFO00_RATIO,
CNTp_PFD03_PFP00_sum_CFO00_CFT00 * 1.0 / NULLIF(CNT_CFO00, 0) AS CNTp_PFD03_PFP00_sum_CFO00_CFT00_TO_CNT_CFO00_RATIO,
CNTp_PFD04_PFP00_sum_CFO00_CFT00 * 1.0 / NULLIF(CNT_CFO00, 0) AS CNTp_PFD04_PFP00_sum_CFO00_CFT00_TO_CNT_CFO00_RATIO,
CNTp_PFD00_PFP00_sum_CFO00_CFT00 * 1.0 / NULLIF(CNT_CFO00, 0) AS CNTp_PFD00_PFP00_sum_CFO00_CFT00_TO_CNT_CFO00_RATIO
FROM Modeling.dbo.bki_var_CNTp_sum AS CNTp_sum
LEFT JOIN Modeling.dbo.bki_var_CNT AS CNT
ON CNTp_sum.loan_application_id = CNT.loan_application_id
ALTER TABLE Modeling.dbo.bki_var_CNTp_sum_TO_CNT_RATIO
REBUILD PARTITION = ALL WITH (DATA_COMPRESSION = PAGE)







DROP TABLE Modeling.dbo.bki_var_CNT_TO_CNT_RATIO
SELECT TOP 0 loan_application_id,
CNT_cfd00_CFT01 * 1.0 / NULLIF(CNT_cfd00_CFT00, 0) AS CNT_cfd00_CFT01_TO_CNT_cfd00_CFT00_RATIO,
CNT_cfd00_CFT02 * 1.0 / NULLIF(CNT_cfd00_CFT00, 0) AS CNT_cfd00_CFT02_TO_CNT_cfd00_CFT00_RATIO,
CNT_cfd00_CFT03 * 1.0 / NULLIF(CNT_cfd00_CFT00, 0) AS CNT_cfd00_CFT03_TO_CNT_cfd00_CFT00_RATIO,
CNT_cfd00_CFT04 * 1.0 / NULLIF(CNT_cfd00_CFT00, 0) AS CNT_cfd00_CFT04_TO_CNT_cfd00_CFT00_RATIO,
CNT_cfd00_CFT05 * 1.0 / NULLIF(CNT_cfd00_CFT00, 0) AS CNT_cfd00_CFT05_TO_CNT_cfd00_CFT00_RATIO,
CNT_CFO02 * 1.0 / NULLIF(CNT_CFO03, 0) AS CNT_CFO02_TO_CNT_CFO03_RATIO,
CNT_CFO02 * 1.0 / NULLIF(CNT_CFO04, 0) AS CNT_CFO02_TO_CNT_CFO04_RATIO,
CNT_CFO03 * 1.0 / NULLIF(CNT_CFO04, 0) AS CNT_CFO03_TO_CNT_CFO04_RATIO
INTO Modeling.dbo.bki_var_CNT_TO_CNT_RATIO
FROM Modeling.dbo.bki_var_CNT



INSERT INTO Modeling.dbo.bki_var_CNT_TO_CNT_RATIO
SELECT loan_application_id,
CNT_cfd00_CFT01 * 1.0 / NULLIF(CNT_cfd00_CFT00, 0) AS CNT_cfd00_CFT01_TO_CNT_cfd00_CFT00_RATIO,
CNT_cfd00_CFT02 * 1.0 / NULLIF(CNT_cfd00_CFT00, 0) AS CNT_cfd00_CFT02_TO_CNT_cfd00_CFT00_RATIO,
CNT_cfd00_CFT03 * 1.0 / NULLIF(CNT_cfd00_CFT00, 0) AS CNT_cfd00_CFT03_TO_CNT_cfd00_CFT00_RATIO,
CNT_cfd00_CFT04 * 1.0 / NULLIF(CNT_cfd00_CFT00, 0) AS CNT_cfd00_CFT04_TO_CNT_cfd00_CFT00_RATIO,
CNT_cfd00_CFT05 * 1.0 / NULLIF(CNT_cfd00_CFT00, 0) AS CNT_cfd00_CFT05_TO_CNT_cfd00_CFT00_RATIO,
CNT_CFO02 * 1.0 / NULLIF(CNT_CFO03, 0) AS CNT_CFO02_TO_CNT_CFO03_RATIO,
CNT_CFO02 * 1.0 / NULLIF(CNT_CFO04, 0) AS CNT_CFO02_TO_CNT_CFO04_RATIO,
CNT_CFO03 * 1.0 / NULLIF(CNT_CFO04, 0) AS CNT_CFO03_TO_CNT_CFO04_RATIO
FROM Modeling.dbo.bki_var_CNT
ALTER TABLE Modeling.dbo.bki_var_CNT_TO_CNT_RATIO
REBUILD PARTITION = ALL WITH (DATA_COMPRESSION = PAGE)







DROP TABLE Modeling.dbo.bki_var_CNTp_sum_TO_CNTp_sum_RATIO
SELECT TOP 0 loan_application_id,
CNTp_PFD01_PFP03_sum_CFO00_CFT00 * 1.0 / NULLIF(CNTp_PFD00_PFP03_sum_CFO00_CFT00, 0) AS CNTp_PFD01_PFP03_sum_CFO00_CFT00_TO_CNTp_PFD00_PFP03_sum_CFO00_CFT00_RATIO
INTO Modeling.dbo.bki_var_CNTp_sum_TO_CNTp_sum_RATIO
FROM Modeling.dbo.bki_var_CNTp_sum



INSERT INTO Modeling.dbo.bki_var_CNTp_sum_TO_CNTp_sum_RATIO
SELECT loan_application_id,
CNTp_PFD01_PFP03_sum_CFO00_CFT00 * 1.0 / NULLIF(CNTp_PFD00_PFP03_sum_CFO00_CFT00, 0) AS CNTp_PFD01_PFP03_sum_CFO00_CFT00_TO_CNTp_PFD00_PFP03_sum_CFO00_CFT00_RATIO
FROM Modeling.dbo.bki_var_CNTp_sum
ALTER TABLE Modeling.dbo.bki_var_CNTp_sum_TO_CNTp_sum_RATIO
REBUILD PARTITION = ALL WITH (DATA_COMPRESSION = PAGE)







DROP TABLE Modeling.dbo.bki_var_Overview_TO_Overview_RATIO
SELECT TOP 0 application_id,
inquiries_1month * 1.0 / NULLIF(inquiries_3month, 0) AS inquiries_1month_TO_inquiries_3month_RATIO,
inquiries_1month * 1.0 / NULLIF(inquiries_6month, 0) AS inquiries_1month_TO_inquiries_6month_RATIO,
inquiries_1month * 1.0 / NULLIF(inquiries_9month, 0) AS inquiries_1month_TO_inquiries_9month_RATIO,
inquiries_1month * 1.0 / NULLIF(inquiries_12month, 0) AS inquiries_1month_TO_inquiries_12month_RATIO,
inquiries_3month * 1.0 / NULLIF(inquiries_6month, 0) AS inquiries_3month_TO_inquiries_6month_RATIO,
inquiries_3month * 1.0 / NULLIF(inquiries_9month, 0) AS inquiries_3month_TO_inquiries_9month_RATIO,
inquiries_3month * 1.0 / NULLIF(inquiries_12month, 0) AS inquiries_3month_TO_inquiries_12month_RATIO,
inquiries_6month * 1.0 / NULLIF(inquiries_9month, 0) AS inquiries_6month_TO_inquiries_9month_RATIO,
inquiries_6month * 1.0 / NULLIF(inquiries_12month, 0) AS inquiries_6month_TO_inquiries_12month_RATIO,
inquiries_9month * 1.0 / NULLIF(inquiries_12month, 0) AS inquiries_9month_TO_inquiries_12month_RATIO,
inquiries_1month * 1.0 / NULLIF(inquiries_total, 0) AS inquiries_1month_TO_inquiries_total_RATIO,
inquiries_3month * 1.0 / NULLIF(inquiries_total, 0) AS inquiries_3month_TO_inquiries_total_RATIO,
inquiries_6month * 1.0 / NULLIF(inquiries_total, 0) AS inquiries_6month_TO_inquiries_total_RATIO,
inquiries_9month * 1.0 / NULLIF(inquiries_total, 0) AS inquiries_9month_TO_inquiries_total_RATIO,
inquiries_12month * 1.0 / NULLIF(inquiries_total, 0) AS inquiries_12month_TO_inquiries_total_RATIO
INTO Modeling.dbo.bki_var_Overview_TO_Overview_RATIO
FROM Modeling.dbo.bki_var_overview



INSERT INTO Modeling.dbo.bki_var_Overview_TO_Overview_RATIO
SELECT application_id,
inquiries_1month * 1.0 / NULLIF(inquiries_3month, 0) AS inquiries_1month_TO_inquiries_3month_RATIO,
inquiries_1month * 1.0 / NULLIF(inquiries_6month, 0) AS inquiries_1month_TO_inquiries_6month_RATIO,
inquiries_1month * 1.0 / NULLIF(inquiries_9month, 0) AS inquiries_1month_TO_inquiries_9month_RATIO,
inquiries_1month * 1.0 / NULLIF(inquiries_12month, 0) AS inquiries_1month_TO_inquiries_12month_RATIO,
inquiries_3month * 1.0 / NULLIF(inquiries_6month, 0) AS inquiries_3month_TO_inquiries_6month_RATIO,
inquiries_3month * 1.0 / NULLIF(inquiries_9month, 0) AS inquiries_3month_TO_inquiries_9month_RATIO,
inquiries_3month * 1.0 / NULLIF(inquiries_12month, 0) AS inquiries_3month_TO_inquiries_12month_RATIO,
inquiries_6month * 1.0 / NULLIF(inquiries_9month, 0) AS inquiries_6month_TO_inquiries_9month_RATIO,
inquiries_6month * 1.0 / NULLIF(inquiries_12month, 0) AS inquiries_6month_TO_inquiries_12month_RATIO,
inquiries_9month * 1.0 / NULLIF(inquiries_12month, 0) AS inquiries_9month_TO_inquiries_12month_RATIO,
inquiries_1month * 1.0 / NULLIF(inquiries_total, 0) AS inquiries_1month_TO_inquiries_total_RATIO,
inquiries_3month * 1.0 / NULLIF(inquiries_total, 0) AS inquiries_3month_TO_inquiries_total_RATIO,
inquiries_6month * 1.0 / NULLIF(inquiries_total, 0) AS inquiries_6month_TO_inquiries_total_RATIO,
inquiries_9month * 1.0 / NULLIF(inquiries_total, 0) AS inquiries_9month_TO_inquiries_total_RATIO,
inquiries_12month * 1.0 / NULLIF(inquiries_total, 0) AS inquiries_12month_TO_inquiries_total_RATIO
FROM Modeling.dbo.bki_var_overview
ALTER TABLE Modeling.dbo.bki_var_Overview_TO_Overview_RATIO
REBUILD PARTITION = ALL WITH (DATA_COMPRESSION = PAGE)