# Job to resume category mapping

The LinkedIn postings do not use the 24 resume categories, so each posting is
mapped with fixed rules based on its title, skills and industries. The rules
are written down here so the mapping can be checked and repeated.
`scripts/clean_data.py` (`map_job_category`) follows this table.

| Resume category | Rule (title keywords, skills, industries) |
|---|---|
| | |

Postings that match no rule are dropped from the job pool.
