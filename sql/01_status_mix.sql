-- Loan status mix by issue year for the filtered sample (36-month loans, development + OOT
-- windows), BEFORE indeterminate loans are dropped. Shows how much the target definition excludes.
SELECT
    year(issue_d)                                        AS issue_year,
    loan_status,
    count(*)                                             AS n_loans,
    round(100.0 * count(*) / sum(count(*)) OVER (PARTITION BY year(issue_d)), 2) AS pct_of_year
FROM read_parquet('{loans_filtered}')
GROUP BY 1, 2
ORDER BY 1, 3 DESC;
