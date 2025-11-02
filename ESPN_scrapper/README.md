# ESPNcricinfo Ball-by-Ball Commentary Scraper

This repository contains code to scrape **ball-by-ball cricket commentary** from ESPNcricinfo for a list of matches.

The scraping logic is based on ESPNcricinfo match index pages such as:  
https://stats.espncricinfo.com/ci/engine/stats/index.html?class=3;spanmin1=17+Feb+2022;spanval1=span;template=results;type=aggregate;view=results

These pages help extract match IDs and build commentary URLs automatically.

---

## 📌 Features

- Fetches ball-by-ball text commentary from ESPNcricinfo
- Handles dynamic page scroll / pagination to load all overs
- Saves commentary with match metadata for analysis
- Ideal for cricket analytics & NLP research on commentary text

---

## 🧠 How It Works

1. **Match Discovery**
   - Identify match list using ESPN stats filters
   - Extract match IDs from links on the results page

2. **URL Construction**
   - Convert match IDs into official commentary page links

3. **Dynamic Scrolling / Pagination**
   - ESPN loads commentary for a few overs at a time
   - Script scrolls gradually to load the full innings feed

4. **Data Extraction**
   - Parse ball number & commentary text
   - Store output in structured CSV/DataFrame format



---

## 🖼️ Output Preview
> <img width="1344" height="667" alt="image" src="https://github.com/user-attachments/assets/98f41cf8-751f-4f9c-ab28-0b3f97c193e8" />


