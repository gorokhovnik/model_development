DROP TABLE #payments
SELECT TOP 0 pay_loan_id,
SUM(IIF(payment_status <> '0' and 1=1 AND 1=1, 1, 0)) AS CNTp_PFD00_PFP00,
SUM(IIF(payment_status <> '0' and 1=1 AND months_ago<=3, 1, 0)) AS CNTp_PFD00_PFP01,
SUM(IIF(payment_status <> '0' and 1=1 AND months_ago<=6, 1, 0)) AS CNTp_PFD00_PFP02,
SUM(IIF(payment_status <> '0' and 1=1 AND months_ago<=12, 1, 0)) AS CNTp_PFD00_PFP03,
SUM(IIF(payment_status <> '0' and payment_status not like '%[2-9]%' AND 1=1, 1, 0)) AS CNTp_PFD01_PFP00,
SUM(IIF(payment_status <> '0' and payment_status not like '%[2-9]%' AND months_ago<=3, 1, 0)) AS CNTp_PFD01_PFP01,
SUM(IIF(payment_status <> '0' and payment_status not like '%[2-9]%' AND months_ago<=6, 1, 0)) AS CNTp_PFD01_PFP02,
SUM(IIF(payment_status <> '0' and payment_status not like '%[2-9]%' AND months_ago<=12, 1, 0)) AS CNTp_PFD01_PFP03,
SUM(IIF(payment_status <> '0' and payment_status like '%[2-3]%' and payment_status not like '%[4-9]%' AND 1=1, 1, 0)) AS CNTp_PFD02_PFP00,
SUM(IIF(payment_status <> '0' and payment_status like '%[2-3]%' and payment_status not like '%[4-9]%' AND months_ago<=3, 1, 0)) AS CNTp_PFD02_PFP01,
SUM(IIF(payment_status <> '0' and payment_status like '%[2-3]%' and payment_status not like '%[4-9]%' AND months_ago<=6, 1, 0)) AS CNTp_PFD02_PFP02,
SUM(IIF(payment_status <> '0' and payment_status like '%[2-3]%' and payment_status not like '%[4-9]%' AND months_ago<=12, 1, 0)) AS CNTp_PFD02_PFP03,
SUM(IIF(payment_status <> '0' and payment_status like  '%[4-9]%' AND 1=1, 1, 0)) AS CNTp_PFD03_PFP00,
SUM(IIF(payment_status <> '0' and payment_status like  '%[4-9]%' AND months_ago<=3, 1, 0)) AS CNTp_PFD03_PFP01,
SUM(IIF(payment_status <> '0' and payment_status like  '%[4-9]%' AND months_ago<=6, 1, 0)) AS CNTp_PFD03_PFP02,
SUM(IIF(payment_status <> '0' and payment_status like  '%[4-9]%' AND months_ago<=12, 1, 0)) AS CNTp_PFD03_PFP03,
SUM(IIF(payment_status <> '0' and payment_status like '%[6-9]%' AND 1=1, 1, 0)) AS CNTp_PFD04_PFP00,
SUM(IIF(payment_status <> '0' and payment_status like '%[6-9]%' AND months_ago<=3, 1, 0)) AS CNTp_PFD04_PFP01,
SUM(IIF(payment_status <> '0' and payment_status like '%[6-9]%' AND months_ago<=6, 1, 0)) AS CNTp_PFD04_PFP02,
SUM(IIF(payment_status <> '0' and payment_status like '%[6-9]%' AND months_ago<=12, 1, 0)) AS CNTp_PFD04_PFP03,
MAX(months_ago) - MAX(IIF(1=1, months_ago, NULL)) AS Months_before_first_delay_PFD00,
MAX(months_ago) - MAX(IIF(payment_status not like '%[2-9]%', months_ago, NULL)) AS Months_before_first_delay_PFD01,
MAX(months_ago) - MAX(IIF(payment_status like '%[2-3]%' and payment_status not like '%[4-9]%', months_ago, NULL)) AS Months_before_first_delay_PFD02,
MAX(months_ago) - MAX(IIF(payment_status like  '%[4-9]%', months_ago, NULL)) AS Months_before_first_delay_PFD03,
MAX(months_ago) - MAX(IIF(payment_status like '%[6-9]%', months_ago, NULL)) AS Months_before_first_delay_PFD04
INTO #payments
FROM Modeling.dbo.BKI_payments
GROUP BY pay_loan_id



INSERT INTO #payments
SELECT pay_loan_id,
SUM(IIF(payment_status <> '0' and 1=1 AND 1=1, 1, 0)) AS CNTp_PFD00_PFP00,
SUM(IIF(payment_status <> '0' and 1=1 AND months_ago<=3, 1, 0)) AS CNTp_PFD00_PFP01,
SUM(IIF(payment_status <> '0' and 1=1 AND months_ago<=6, 1, 0)) AS CNTp_PFD00_PFP02,
SUM(IIF(payment_status <> '0' and 1=1 AND months_ago<=12, 1, 0)) AS CNTp_PFD00_PFP03,
SUM(IIF(payment_status <> '0' and payment_status not like '%[2-9]%' AND 1=1, 1, 0)) AS CNTp_PFD01_PFP00,
SUM(IIF(payment_status <> '0' and payment_status not like '%[2-9]%' AND months_ago<=3, 1, 0)) AS CNTp_PFD01_PFP01,
SUM(IIF(payment_status <> '0' and payment_status not like '%[2-9]%' AND months_ago<=6, 1, 0)) AS CNTp_PFD01_PFP02,
SUM(IIF(payment_status <> '0' and payment_status not like '%[2-9]%' AND months_ago<=12, 1, 0)) AS CNTp_PFD01_PFP03,
SUM(IIF(payment_status <> '0' and payment_status like '%[2-3]%' and payment_status not like '%[4-9]%' AND 1=1, 1, 0)) AS CNTp_PFD02_PFP00,
SUM(IIF(payment_status <> '0' and payment_status like '%[2-3]%' and payment_status not like '%[4-9]%' AND months_ago<=3, 1, 0)) AS CNTp_PFD02_PFP01,
SUM(IIF(payment_status <> '0' and payment_status like '%[2-3]%' and payment_status not like '%[4-9]%' AND months_ago<=6, 1, 0)) AS CNTp_PFD02_PFP02,
SUM(IIF(payment_status <> '0' and payment_status like '%[2-3]%' and payment_status not like '%[4-9]%' AND months_ago<=12, 1, 0)) AS CNTp_PFD02_PFP03,
SUM(IIF(payment_status <> '0' and payment_status like  '%[4-9]%' AND 1=1, 1, 0)) AS CNTp_PFD03_PFP00,
SUM(IIF(payment_status <> '0' and payment_status like  '%[4-9]%' AND months_ago<=3, 1, 0)) AS CNTp_PFD03_PFP01,
SUM(IIF(payment_status <> '0' and payment_status like  '%[4-9]%' AND months_ago<=6, 1, 0)) AS CNTp_PFD03_PFP02,
SUM(IIF(payment_status <> '0' and payment_status like  '%[4-9]%' AND months_ago<=12, 1, 0)) AS CNTp_PFD03_PFP03,
SUM(IIF(payment_status <> '0' and payment_status like '%[6-9]%' AND 1=1, 1, 0)) AS CNTp_PFD04_PFP00,
SUM(IIF(payment_status <> '0' and payment_status like '%[6-9]%' AND months_ago<=3, 1, 0)) AS CNTp_PFD04_PFP01,
SUM(IIF(payment_status <> '0' and payment_status like '%[6-9]%' AND months_ago<=6, 1, 0)) AS CNTp_PFD04_PFP02,
SUM(IIF(payment_status <> '0' and payment_status like '%[6-9]%' AND months_ago<=12, 1, 0)) AS CNTp_PFD04_PFP03,
MAX(months_ago) - MAX(IIF(1=1, months_ago, NULL)) AS Months_before_first_delay_PFD00,
MAX(months_ago) - MAX(IIF(payment_status not like '%[2-9]%', months_ago, NULL)) AS Months_before_first_delay_PFD01,
MAX(months_ago) - MAX(IIF(payment_status like '%[2-3]%' and payment_status not like '%[4-9]%', months_ago, NULL)) AS Months_before_first_delay_PFD02,
MAX(months_ago) - MAX(IIF(payment_status like  '%[4-9]%', months_ago, NULL)) AS Months_before_first_delay_PFD03,
MAX(months_ago) - MAX(IIF(payment_status like '%[6-9]%', months_ago, NULL)) AS Months_before_first_delay_PFD04
FROM Modeling.dbo.BKI_payments
GROUP BY pay_loan_id
ALTER TABLE #payments
REBUILD PARTITION = ALL WITH (DATA_COMPRESSION = PAGE)



DROP TABLE Modeling.dbo.bki_var_loans_and_payments
SELECT TOP 0 loans.*,
CNTp_PFD00_PFP00,
CNTp_PFD00_PFP01,
CNTp_PFD00_PFP02,
CNTp_PFD00_PFP03,
CNTp_PFD01_PFP00,
CNTp_PFD01_PFP01,
CNTp_PFD01_PFP02,
CNTp_PFD01_PFP03,
CNTp_PFD02_PFP00,
CNTp_PFD02_PFP01,
CNTp_PFD02_PFP02,
CNTp_PFD02_PFP03,
CNTp_PFD03_PFP00,
CNTp_PFD03_PFP01,
CNTp_PFD03_PFP02,
CNTp_PFD03_PFP03,
CNTp_PFD04_PFP00,
CNTp_PFD04_PFP01,
CNTp_PFD04_PFP02,
CNTp_PFD04_PFP03,
Months_before_first_delay_PFD00,
Months_before_first_delay_PFD01,
Months_before_first_delay_PFD02,
Months_before_first_delay_PFD03,
Months_before_first_delay_PFD04
INTO Modeling.dbo.bki_var_loans_and_payments
FROM Modeling.dbo.BKI_loans AS loans
LEFT JOIN #payments AS payments
ON loans.loan_id = payments.pay_loan_id



INSERT INTO Modeling.dbo.bki_var_loans_and_payments
SELECT loans.*,
CNTp_PFD00_PFP00,
CNTp_PFD00_PFP01,
CNTp_PFD00_PFP02,
CNTp_PFD00_PFP03,
CNTp_PFD01_PFP00,
CNTp_PFD01_PFP01,
CNTp_PFD01_PFP02,
CNTp_PFD01_PFP03,
CNTp_PFD02_PFP00,
CNTp_PFD02_PFP01,
CNTp_PFD02_PFP02,
CNTp_PFD02_PFP03,
CNTp_PFD03_PFP00,
CNTp_PFD03_PFP01,
CNTp_PFD03_PFP02,
CNTp_PFD03_PFP03,
CNTp_PFD04_PFP00,
CNTp_PFD04_PFP01,
CNTp_PFD04_PFP02,
CNTp_PFD04_PFP03,
Months_before_first_delay_PFD00,
Months_before_first_delay_PFD01,
Months_before_first_delay_PFD02,
Months_before_first_delay_PFD03,
Months_before_first_delay_PFD04
FROM Modeling.dbo.BKI_loans AS loans
LEFT JOIN #payments AS payments
ON loans.loan_id = payments.pay_loan_id
WHERE (loan_id IS NOT NULL) AND (infosource IS NOT NULL)
ALTER TABLE Modeling.dbo.bki_var_loans_and_payments
REBUILD PARTITION = ALL WITH (DATA_COMPRESSION = PAGE)