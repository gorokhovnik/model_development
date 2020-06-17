DROP TABLE Modeling.dbo.bki_var_overview
SELECT TOP 0 appl.application_id,
inquiries_1month AS inquiries_1month,
inquiries_3month AS inquiries_3month,
inquiries_6month AS inquiries_6month,
inquiries_9month AS inquiries_9month,
inquiries_12month AS inquiries_12month,
inquiries_total AS inquiries_total,
payload AS payload,
IIF(segment = '¿ ', 1, 0) AS type_auto,
IIF(segment = '» ', 1, 0) AS type_mort,
IIF(segment = 'œ ', 1, 0) AS type_pil
INTO Modeling.dbo.bki_var_overview
FROM Modeling.dbo.application AS appl
LEFT JOIN Modeling.dbo.BKI_overview AS ove
ON appl.application_id = ove.over_application_id



INSERT INTO Modeling.dbo.bki_var_overview
SELECT appl.application_id,
inquiries_1month AS inquiries_1month,
inquiries_3month AS inquiries_3month,
inquiries_6month AS inquiries_6month,
inquiries_9month AS inquiries_9month,
inquiries_12month AS inquiries_12month,
inquiries_total AS inquiries_total,
payload AS payload,
IIF(segment = '¿ ', 1, 0) AS type_auto,
IIF(segment = '» ', 1, 0) AS type_mort,
IIF(segment = 'œ ', 1, 0) AS type_pil
FROM Modeling.dbo.application AS appl
LEFT JOIN Modeling.dbo.BKI_overview AS ove
ON appl.application_id = ove.over_application_id
WHERE 1=1 AND 1=1
ALTER TABLE Modeling.dbo.bki_var_overview
REBUILD PARTITION = ALL WITH (DATA_COMPRESSION = PAGE)