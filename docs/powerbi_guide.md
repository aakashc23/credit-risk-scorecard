# Power BI guide (optional)

The Streamlit app is the main deliverable. This guide shows how to rebuild the cut-off
simulator as a one-page Power BI report from the CSVs the pipeline exports.

> **Honest status:** no `.pbix` file is produced or committed. The measures below are
> written by hand from the exported columns and have not been validated inside Power BI
> Desktop by the pipeline's tests. Numbers in any report you build come from the CSVs, which
> come from `artifacts/`.

## 1. Data

`python scripts/run_pipeline.py` writes three tidy CSVs to `artifacts/powerbi/` (copies of
tables the pipeline already produces; no extra analytics):

| File | One row per | Key columns |
|---|---|---|
| `cutoff_table.csv` | score cut-off (300 to 850, step 5), out-of-time loans | `cutoff`, `n_total`, `n_approved`, `n_rejected`, `approval_rate`, `rejection_rate`, `expected_bad_rate`, `observed_bad_rate`, `approved_exposure`, `expected_loss`, `expected_loss_rate`, `realised_loss` |
| `score_band_summary.csv` | 50-point score band | `score_band_low`, `score_band_high`, `n_loans`, `bad_rate`, `avg_pd`, `expected_loss`, `realised_loss` |
| `decile_table_oot.csv` | score decile (1 = riskiest) | `decile`, `n`, `bads`, `bad_rate`, `mean_pd`, `min_score`, `max_score`, `cum_bad_capture` |

Load them with **Home > Get data > Text/CSV**. Set `cutoff` and the `score_band_*` columns to
*Whole number* or *Decimal number*; rates are fractions (format them as percentages).
Do not create relationships; the tables are independent.

## 2. The cut-off What-If parameter

**Modeling > New parameter > Numeric range**: name `Cutoff`, minimum 300, maximum 850,
increment 5, default 585 (any value; it is only the starting position), *Add slicer to this
page* ticked. Power BI creates a table `Cutoff` and a measure `Cutoff Value`
(`SELECTEDVALUE(Cutoff[Cutoff])`).

## 3. DAX measures

`cutoff_table` already holds the outcome at every cut-off, so each measure looks up the row
for the slider value. Create them on the `cutoff_table` table.

```dax
Selected Cutoff = SELECTEDVALUE ( Cutoff[Cutoff], 300 )

Approval Rate =
CALCULATE (
    MAX ( cutoff_table[approval_rate] ),
    FILTER ( ALL ( cutoff_table ), cutoff_table[cutoff] = [Selected Cutoff] )
)

Rejection Rate = 1 - [Approval Rate]

Approved Loans =
CALCULATE (
    MAX ( cutoff_table[n_approved] ),
    FILTER ( ALL ( cutoff_table ), cutoff_table[cutoff] = [Selected Cutoff] )
)

Rejected Loans =
CALCULATE (
    MAX ( cutoff_table[n_rejected] ),
    FILTER ( ALL ( cutoff_table ), cutoff_table[cutoff] = [Selected Cutoff] )
)

Expected Bad Rate =
CALCULATE (
    MAX ( cutoff_table[expected_bad_rate] ),
    FILTER ( ALL ( cutoff_table ), cutoff_table[cutoff] = [Selected Cutoff] )
)

Observed Bad Rate =
CALCULATE (
    MAX ( cutoff_table[observed_bad_rate] ),
    FILTER ( ALL ( cutoff_table ), cutoff_table[cutoff] = [Selected Cutoff] )
)

Expected Loss =
CALCULATE (
    MAX ( cutoff_table[expected_loss] ),
    FILTER ( ALL ( cutoff_table ), cutoff_table[cutoff] = [Selected Cutoff] )
)

Expected Loss % of Exposure =
CALCULATE (
    MAX ( cutoff_table[expected_loss_rate] ),
    FILTER ( ALL ( cutoff_table ), cutoff_table[cutoff] = [Selected Cutoff] )
)

Realised Loss =
CALCULATE (
    MAX ( cutoff_table[realised_loss] ),
    FILTER ( ALL ( cutoff_table ), cutoff_table[cutoff] = [Selected Cutoff] )
)

-- Baseline: approve everyone (cut-off 300)
Expected Bad Rate (Approve All) =
CALCULATE (
    MAX ( cutoff_table[expected_bad_rate] ),
    FILTER ( ALL ( cutoff_table ), cutoff_table[cutoff] = 300 )
)

Bad Rate Reduction (pts) = [Expected Bad Rate (Approve All)] - [Expected Bad Rate]
```

`MAX` is only a way to pick the single matching value; there is exactly one row per cut-off.

## 4. One-page layout

1. **Top left:** the `Cutoff` slicer (slider style).
2. **KPI cards row 1:** Approval Rate, Rejection Rate, Approved Loans, Rejected Loans.
3. **KPI cards row 2:** Expected Bad Rate, Observed Bad Rate, Expected Loss, Expected Loss %
   of Exposure, Realised Loss.
4. **Line chart (trade-off):** X = `cutoff_table[approval_rate]`, Y = `expected_bad_rate` and
   `observed_bad_rate`. Because the measures read the whole table with `ALL`, the lines do not
   move with the slider, which is what you want; add the Selected Cutoff as a reference line
   or a one-point series.
5. **Column chart:** `score_band_summary[score_band_low]` on X, `n_loans` on columns and
   `bad_rate` as a line (secondary axis), showing volume and risk by score band.
6. **Column chart:** `decile_table_oot[decile]` on X, `bad_rate` and `mean_pd` on Y.
7. **Text box:** state the caveat from the app (accepted loans only; portfolio project; not a
   lending system).

Tip: turn off the default slicer interactions for the line chart (Format > Edit interactions)
so the slider only changes the cards.

## 5. Refreshing

Re-run the pipeline, then **Home > Refresh** in Power BI Desktop (or point the CSV
sources at the new folder).
