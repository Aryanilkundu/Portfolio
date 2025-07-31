import requests
import pandas as pd


url = "https://stats.espncricinfo.com/ci/engine/stats/index.html?class=2;filter=advanced;orderby=runs;result=1;size=200;template=results;type=batting"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.espncricinfo.com/"
}
response = requests.get(url, headers=headers)
tables = pd.read_html(response.text)
for i, table in enumerate(tables):
    print(f"Table {i}:\n", table.head()) 
df = tables[2] 
print(df)