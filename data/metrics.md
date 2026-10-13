# Rezultate pe setul de test (4578 wafere, neîntâlnite la antrenare)

| Model | Acuratețe | Macro F1 |
|---|---|---|
| Random Forest (features simple) | 0.852 | 0.803 |
| CNN | 0.879 | 0.830 |

## F1 pe clasă

| Clasă | RF | CNN | Nr. test |
|---|---|---|---|
| Center | 0.92 | 0.93 | 644 |
| Donut | 0.75 | 0.79 | 83 |
| Edge-Loc | 0.79 | 0.83 | 779 |
| Edge-Ring | 0.96 | 0.97 | 1452 |
| Loc | 0.66 | 0.73 | 539 |
| Near-full | 0.98 | 0.84 | 22 |
| Random | 0.87 | 0.89 | 130 |
| Scratch | 0.42 | 0.60 | 179 |
| none | 0.88 | 0.88 | 750 |