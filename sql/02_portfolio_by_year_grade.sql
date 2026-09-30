-- Modelling sample (target known): volume, bad rate and average loan by issue year and LC grade.
SELECT
    year(issue_d)              AS issue_year,
    grade,
    count(*)                   AS n_loans,
    round(avg(bad), 4)         AS bad_rate,
    round(avg(loan_amnt), 0)   AS avg_loan_amnt
FROM read_parquet('{modelling}')
GROUP BY 1, 2
ORDER BY 1, 2;
