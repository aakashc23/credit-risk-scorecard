-- Out-of-time population in 50-point score bands: volume, observed vs predicted bad rate, losses.
SELECT
    CAST(floor(score / 50) * 50 AS INTEGER)  AS score_band_low,
    CAST(floor(score / 50) * 50 + 49 AS INTEGER) AS score_band_high,
    count(*)                                 AS n_loans,
    round(avg(bad), 4)                       AS bad_rate,
    round(avg(pd), 4)                        AS avg_pd,
    round(sum(el), 0)                        AS expected_loss,
    round(sum(realised_loss), 0)             AS realised_loss
FROM read_parquet('{scored_oot}')
GROUP BY 1, 2
ORDER BY 1;
