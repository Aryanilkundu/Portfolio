import time
from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
#update this with the location of chromedriver
CHROME_DRIVER_PATH = r"C:\Users\ARYANIL TANISHA\Downloads\chromedriver-win64\chromedriver-win64\chromedriver.exe"
URL = "https://www.espncricinfo.com/series/india-in-england-2025-1445348/england-vs-india-2nd-test-1448350/ball-by-ball-commentary"

options = Options()
options.add_argument("--window-size=1920,1080")
service = Service(CHROME_DRIVER_PATH)
driver = webdriver.Chrome(service=service, options=options)

driver.get(URL)
time.sleep(8)
prev_count = 0
same_count_repeats = 0
max_repeats = 8

while True:
    # Scroll down by 50 pixels
    driver.execute_script("window.scrollBy(0, 600);")
    time.sleep(1.5)  # Initial wait after scroll

    # Double check after a short pause
    curr_count = len(driver.find_elements(By.CSS_SELECTOR, "div.ds-text-tight-m"))
    if curr_count == prev_count:
        # Wait a bit longer and check again
        time.sleep(2)
        curr_count = len(driver.find_elements(By.CSS_SELECTOR, "div.ds-text-tight-m"))

        if curr_count == prev_count:
            # Try scrolling up slightly and then down again to re-trigger loading
            driver.execute_script("window.scrollBy(0, -100);")
            time.sleep(0.5)
            driver.execute_script("window.scrollBy(0, 150);")
            time.sleep(2)
            curr_count = len(driver.find_elements(By.CSS_SELECTOR, "div.ds-text-tight-m"))

            if curr_count == prev_count:
                same_count_repeats += 1
            else:
                same_count_repeats = 0
        else:
            same_count_repeats = 0
    else:
        same_count_repeats = 0

    if same_count_repeats >= max_repeats:
        break

    prev_count = curr_count

# After loop, collect all commentary
elements = driver.find_elements(By.CSS_SELECTOR, "div.ds-text-tight-m")
texts = [e.text.strip() for e in elements if e.text.strip()]
print(f"Total entries collected: {len(texts)}")
driver.quit()
# Optionally save to CSV
# import pandas as pd
# pd.Series(texts).to_csv("full_commentary.csv", index=False, header=False)
