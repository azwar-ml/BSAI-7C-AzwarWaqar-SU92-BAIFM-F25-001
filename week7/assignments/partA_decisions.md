| What | How detected | Rule applied | Rows affected |
|---|---|---|---:|
| Register join keys | Case and internal-space mismatch | Uppercase and remove spaces | 2 |
| Outside federation | Key absent from all 24 boundary features | Exclude D00 overlapping capital territory from federation sum | 1 |
| Boarding scale | Under one boarding per resident, isolated thousand-unit entries | Multiply three thousand-unit entries by 1000; rounding uncertainty retained | 3 |
| Concession fraction | Fraction among percentage entries | Multiply fractional entries by 100 | 4 |
| Road sentinel | Nonpositive road length | Replace with missing | 2 |
| Mixed date formats | Slash vs ISO strings | Parse slash dates as DD/MM/YYYY and ISO as YYYY-MM-DD explicitly | 385 |
| Exact re-submissions | All columns identical | Keep one identical submission | 30 |
| Impossible daily counts | Negative counts or on-time > operated or operated > scheduled | Preserve filed slot; mask all measurement columns on impossible row (no invented repairs) | 7 |
| Route mode labels | Six spellings for four modes | Normalize brt and Bus Rapid Transit to BRT | 3 |
| punctuality_pct sentinel | Out-of-domain operator code | Mask code as missing; do not impute | 3 |
| cost_recovery_pct sentinel | Out-of-domain operator code | Mask code as missing; do not impute | 2 |
| Fare summary rows | All tickets equals three fare-product children | Exclude summaries from parts; retain summary separately | 72 |
