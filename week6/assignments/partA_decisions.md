| What | How detected | Rule applied | Rows affected |
|---|---|---|---:|
| Conflicting district duplicate | Name differs from GeoJSON canonical name | Drop noncanonical duplicate; retain canonical Kotla | 1 |
| Household scale | Persons per household exceeds 20; value recorded in thousands | Multiply household entry by 1000, consistent with register unit | 1 |
| Percent suffix | Literal percent sign | Strip suffix and parse numeric; preserve genuine blanks | 2 |
| Duration sentinel | Negative minutes | Replace -1 with missing, never zero | 2 |
| Mixed date formats | Slash vs ISO strings | Parse slash dates as DD/MM/YYYY and ISO as YYYY-MM-DD explicitly | 174 |
| Exact re-submissions | All columns identical | Keep one identical submission | 12 |
| Conflicting daily submissions | Same date-zone and meter, different peak and duration | One zone-day: retain common meter and average disputed measurements; not a known true value | 3 |
| Peak sentinel | Nonpositive peak MW | Replace with missing | 46 |
| Counter reset | Negative consecutive meter difference | Flag reset interval as unknown; do not sum meter levels or bridge gaps | 1 |
| Absent zone-days | Full date-zone calendar minus submissions | Keep absent slots missing; invalidate energy differences spanning gaps | 128 |
| Published generation totals | All sources / Total duplicates sum of children | Exclude total rows from composition; retain separate summaries | 18 |
