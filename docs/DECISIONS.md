# Decisions

One line each: date, decision, who.

- 2026-10-04: GCC output contract between bearing.py and srp.py locked (spec 5.2). Avik, Srihaas.
- 2026-10-05: Added MAX_LAG_SAMPLES = 4.4 to the Bearing section of config. Avik.
- 2026-10-05: Added empty root conftest.py so plain `pytest` can import dac. Avik.
- 2026-10-08: Finding: provisional tracker thresholds do not flag an approach before 3 m on the synthetic 6 m→1 m walk (LEVEL_MARGIN_DB=10 binds; flag never fires before 1 m). Test T4 marked xfail until S7 tuning. Srihaas.
