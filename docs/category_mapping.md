# Job to resume category mapping

Status: draft for team review. `scripts/clean_data.py` (`map_job_category`) should follow these rules once they are agreed.

The LinkedIn postings do not use the 24 resume categories, so each posting is mapped with the fixed rules below. The rules were written after reading what each resume category really contains.

## How a posting is mapped

1. Drop postings with under 50 words of description or a duplicate description. This takes 123,849 postings down to 106,596.
2. If the title matches the exclusion list, leave the posting unmapped.
3. Try the title rules in order. The first match wins, so specific domains come before general ones. For example, "Automotive Sales" is AUTOMOBILE, not SALES.
4. If no title rule matches, use the industry rule. It only applies to `Retail Apparel and Fashion` jobs whose title also looks like a retail or apparel role.
5. Everything else is left out of the job pool. That is about 49% of postings, mostly roles with no matching category, such as administrative assistants, project managers, warehouse staff and lawyers.

### Why titles and not the LinkedIn skills

The proposal says job skills and industries would be mapped. Testing showed that does not work well:

- The LinkedIn "skills" are only 35 broad job functions, such as "Sales" or "Management".
- Using industry alone gave results like "Custodial Handyman" in BANKING and "Overnight Cleaner" in FITNESS.

Mention this change in the final report.

## What each category covers

Some categories do not mean what their name suggests, because the resume labels look keyword-based. See `docs/data_notes.md`.

| Category | Jobs it covers | Jobs found | In pool |
|---|---|---:|---:|
| ACCOUNTANT | Accountants, bookkeepers, audit, tax, controllers | 3,704 | 505 |
| ADVOCATE | Patient, customer and victim advocates, social workers. Not lawyers. | 328 | 328 |
| AGRICULTURE | Farming, agronomy, horticulture, irrigation | 84 | 84 |
| APPAREL | Fashion, garment, footwear and apparel retail | 376 | 376 |
| ARTS | Artists, art directors, museums, theatre, photographers | 132 | 132 |
| AUTOMOBILE | Auto and diesel mechanics, body shops, service advisors | 519 | 505 |
| AVIATION | Aircraft mechanics, pilots, flight crew, avionics | 159 | 159 |
| BANKING | Tellers, bankers, loans, mortgages | 1,090 | 505 |
| BPO | Customer service, call and contact centres | 1,305 | 505 |
| BUSINESS-DEVELOPMENT | Business development, partnerships | 760 | 505 |
| CHEF | Chefs, cooks, bakers | 537 | 505 |
| CONSTRUCTION | Construction, skilled trades, estimators, superintendents | 1,850 | 505 |
| CONSULTANT | Consultants other than sales and leasing | 1,272 | 505 |
| DESIGNER | Graphic, UX, interior and technical designers | 1,003 | 505 |
| DIGITAL-MEDIA | Social media, digital marketing, SEO, video | 551 | 505 |
| ENGINEERING | Mechanical, electrical, civil and other non-IT engineers | 6,597 | 505 |
| FINANCE | Financial analysts and advisors, treasury, investment | 2,092 | 505 |
| FITNESS | Personal trainers, fitness instructors, yoga, gyms | 101 | 101 |
| HEALTHCARE | Nurses, therapists, physicians, medical staff | 13,200 | 505 |
| HR | Human resources, recruiters, talent acquisition | 1,505 | 505 |
| INFORMATION-TECHNOLOGY | Software, data, networks, IT support, security | 8,511 | 505 |
| PUBLIC-RELATIONS | Public relations, media relations, communications | 251 | 251 |
| SALES | Sales staff, account executives and managers, leasing | 7,236 | 505 |
| TEACHER | Teachers, tutors, professors | 793 | 505 |
| **Total** | | **53,956** | **10,016** |

"In pool" caps each category at 505, the cap that gets the pool to about 10,000. Seven categories have fewer than 505 jobs and keep all of them, so the pool is not perfectly balanced.

## Exact rules

Patterns are Python regular expressions, matched without regard to case. `(?-i:...)` is case-sensitive, so "IT" does not match the word "it".

Exclusion list:

```text
sandwich artist|state farm|highlands ranch|health (and|&) safety|environmental health|pilot flying|pilot travel|pilot company|wind farm|solar farm|farm reporter|night auditor|crop insurance|agriculture derivatives|fancy farm|\.net ui|ui tax|materials developer|fueler|pizza|store associate|executive assistant|medical drivers
```

Title rules, in order:

```text
 1 INFORMATION-TECHNOLOGY  \bsoftware (engineer|developer|development engineer)|\bsalesforce\b|\bcitrix\b|\bvoip\b|\bcisco\b|\bcmdb\b|exchange engineer|system support|microsoft systems|machine learning|\bml engineer|data engineering
 2 CHEF                    \b(chef|sous|cook|culinary|pastry|kitchen manager|baker)\b(?! hughes| tilly)
 3 AVIATION                \b(aviation|aircraft|airframe|avionics?|a&p|airline|flight attendant|helicopter|pilot)\b(?! plant| program| project)
 4 AUTOMOBILE              \b(automotive|automobile|auto (body|technician|mechanic|sales|service|parts|detailer)|collision|service advisor|diesel (technician|mechanic)|car (sales|dealer)|vehicle technician|lube tech)\b
 5 AGRICULTURE             \b(agricultur\w*|agronom\w*|farm|farmer|crop|horticultur\w*|greenhouse|irrigation|soil|livestock|extension agent)\b
 6 APPAREL                 \b(apparel|fashion|garment|textile|footwear|sewing|sewist|seamstress|tailor)\b
 7 FITNESS                 \b(fitness|personal trainer|yoga|pilates|gym|strength (and|&) conditioning|group exercise|wellness coach)\b
 8 ADVOCATE                \b(advocate|social worker|case worker|caseworker|community outreach)\b
 9 BPO                     \b(call cent(er|re)|contact cent(er|re)|customer (service|support|care)|client support|workforce management)\b|(?-i:\bBPO\b)
10 PUBLIC-RELATIONS        \b(public relations|media relations|publicist|communications? (specialist|manager|director|coordinator|officer|associate)|corporate communications?)\b|(?-i:\bPR (professional|assistant|manager|specialist|coordinator|director|intern)\b)
11 DIGITAL-MEDIA           \b(digital media|social media|digital marketing|media buyer|paid media|programmatic|content (creator|strategist|producer)|video (producer|editor)|digital content)\b|(?-i:\b(SEO|PPC)\b)
12 ARTS                    \b(artist|art (director|teacher|supervisor)|museum|curator|gallery|theat(er|re)|musician|illustrator|choreographer|photographer|dance (teacher|instructor)|(performing|visual|fine|language) arts)\b
13 DESIGNER                \b(designer|user experience|creative director)\b|(?-i:\b(UX|UI)\b)
14 TEACHER                 (?<!clinical )(?<!nurse )(?<!informatics )\b(teachers?|educator|teaching|tutor|paraprofessional|professor|lecturer|preschool)\b
15 HR                      \b(human resources?|hrbp|recruiter|recruiting|talent acquisition|people operations|benefits (specialist|administrator)|compensation analyst)\b|(?-i:(?<![/\d$])\bHR\b)
16 BUSINESS-DEVELOPMENT    \b(business development|bdr|partnerships? manager|strategic partnerships?)\b
17 BANKING                 \b(bank|banker|banking|teller|loan|mortgage|credit analyst)\b
18 ACCOUNTANT              \b(accountant|accounting|bookkeep\w*|accounts? (payable|receivable)|auditor|audit|tax|controller)\b|(?-i:\bCPA\b)
19 FINANCE                 \b(financ\w*|fp&a|treasury|investment|wealth|portfolio manager)\b|(?-i:\bCFO\b)
20 SALES                   \b(sales|salesperson|account executive|account manager|retail associate|merchandiser|leasing consultant)\b
21 CONSULTANT              \bconsult\w*\b
22 INFORMATION-TECHNOLOGY  \b(software|developer|programmer|sysadmin|systems? administrator|network|devops|cyber\w*|cloud|help ?desk|database|data (engineer|scientist|analyst)|analytics|information technology|information security|full stack|(front|back) ?end (developer|engineer)|java|python|sql)\b|(?-i:\b(IT|SRE)\b)
23 CONSTRUCTION            \b(construction|carpenter|electrician|plumber|superintendent|foreman|estimator|hvac|laborer|mason|roofer|welder|pipefitter)\b
24 HEALTHCARE              \b(nurses?|nursing|physician|medical|clinical|health ?care|(mental|behavioral|home|public) health|therapist|pharmacist|pharmacy|dental|dentist|caregiver|surgeon|radiolog\w*|hospital|patient)\b|(?-i:\b(RN|LPN|CNA)\b)
25 ENGINEERING             \b(engineer|engineering)\b
```

Industry rule, used only when no title rule matched and the title is not excluded:

```text
Industry is "Retail Apparel and Fashion" AND title matches
\b(?:store|retail|sales|selling|merchandis\w*|stylist|buyer|pattern|keyholder|shop|showroom)\b
-> APPAREL
```

## How well the rules work

Checked by reading job titles, so this is one reader's judgement on small samples, not a measured accuracy.

- **Full check:** 20 random titles per category (480 in total), on the version before the last fixes. About 90% looked right (433 of 480). The weakest were AVIATION (13 of 20), PUBLIC-RELATIONS (14 of 20) and AGRICULTURE (15 of 20), mostly because of a wider industry fallback that has since been removed.
- **Recheck:** fresh samples of the weak categories after those fixes. AVIATION, PUBLIC-RELATIONS and APPAREL then read mostly correct.
- **Not re-measured:** the final rules have not had a new full 480-title check, so the 90% figure is not a score for this exact version.

## Known limits

- ENGINEERING still contains some IT-like titles, such as "System Engineer" and "Sr. Netsuite Engineer".
- BPO is really "customer service". The BPO resumes are too mixed to define it better.
- DESIGNER includes technical roles like piping and CAD designers. The resumes include mechanical and technical designers too, so these are kept.
- Lawyers and paralegals are left out. "Advocate" in this dataset does not mean lawyer: only 14% of ADVOCATE resumes mention attorney, lawyer or paralegal.
- The small categories have few jobs: AGRICULTURE 84, FITNESS 101, ARTS 132, AVIATION 159.
