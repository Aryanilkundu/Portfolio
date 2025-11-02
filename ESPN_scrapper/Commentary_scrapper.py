import time
import pandas as pd
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from urllib.parse import urljoin
from selenium.webdriver.common.keys import Keys

# --- Configuration ---
RESULTS_TABLE_URL = "https://stats.espncricinfo.com/ci/engine/stats/index.html?class=3;spanmin1=17+Feb+2022;spanval1=span;template=results;type=aggregate;view=results"
BASE_URL = "https://stats.espncricinfo.com"

# --- Selectors for Scraping ---
COMMENTARY_CONTAINER_SELECTOR = "div.ds-relative"
OVER_SELECTOR = "span.ds-text-tight-s"
RUNS_INDICATOR_SELECTOR = "div.ds-w-10"
BALL_INFO_SELECTOR = "div.ds-text-tight-m"
COMMENTARY_TEXT_SELECTOR = "p.ci-html-content"


def initialize_driver():
    """Sets up and returns a Selenium WebDriver instance."""
    print("Initializing WebDriver...")
    chrome_options = Options()
    # chrome_options.add_argument("--headless") # Comment out to see the browser actions
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--log-level=3")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def get_scorecard_links(driver, url):
    """
    Visits the results table and extracts the href for each match scorecard
    by finding links that contain the unique match URL pattern and a numeric ID.
    """
    print(f"Navigating to a table: {url}")
    driver.get(url)
    time.sleep(5)
    links = set()
    try:
        scorecard_link_selector = "a[href*='/engine/match/']"
        WebDriverWait(driver, 20).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, scorecard_link_selector))
        )
        link_elements = driver.find_elements(By.CSS_SELECTOR, scorecard_link_selector)
        for link_element in link_elements:
            href = link_element.get_attribute('href')
            if href:
                if re.search(r'/engine/match/\d+\.html', href):
                    full_link = urljoin(BASE_URL, href)
                    links.add(full_link)
        print(f"Found {len(links)} unique match scorecard links.")
        return list(links)
    except Exception as e:
        print(f"Could not find or process scorecard links. Error: {e}")
        return []

def get_second_innings_team(driver):
    """
    On a scorecard page, returns the name of the team batting second.
    """
    try:
        team_name_selector = "a[class*='ds-inline-flex'] .ds-text-tight-l"
        WebDriverWait(driver, 15).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, team_name_selector))
        )
        team_elements = driver.find_elements(By.CSS_SELECTOR, team_name_selector)
        
        if len(team_elements) >= 2:
            second_team = team_elements[1].text
            print(f"  -> Found second innings team: {second_team}")
            return second_team
        return "N/A"
    except Exception as e:
        print(f"  -> Could not extract second innings team. Error: {e}")
        return "N/A"

def get_winner_info(driver):
    """
    On a scorecard page, returns the match result summary string (e.g., 'Australia won by 44 runs').
    """
    try:
        winner_info_selector = "p[class*='ds-text-tight-s'] > span"
        winner_element = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, winner_info_selector))
        )
        winner_text = winner_element.text.strip()
        print(f"  -> Found winner info: {winner_text}")
        return winner_text
    except Exception as e:
        print(f"  -> Could not extract winner info. Error: {e}")
        return "N/A"

def parse_commentary_data(raw_data, match_id, second_innings_team, winner_info):
    """
    Parses the raw scraped text into a structured DataFrame and adds match-level info.
    """
    parsed_records = []
    output_value = 1 if second_innings_team != "N/A" and second_innings_team in winner_info else 0

    for entry in raw_data:
        over = entry.get('over')
        runs_indicator = entry.get('runs_indicator', '').strip()
        ball_info = entry.get('ball_info')
        commentary = entry.get('commentary_text')

        if not all([over, ball_info, commentary]):
            continue

        is_wicket = "W" in runs_indicator
        runs = 0
        numeric_part = re.search(r'(\d+)', runs_indicator)
        if numeric_part:
            runs = int(numeric_part.group(1))
            
        parsed_records.append({
            "match_id": match_id,
            "second_innings_batting_team": second_innings_team,
            "match_winner_info": winner_info,
            "output": output_value, 
            "Over_number": over,
            "ball_info": ball_info,
            "Runs": runs,
            "Wicket": is_wicket,
            "Commentary": commentary
        })
    
    return pd.DataFrame(parsed_records)

def scroll_and_scrape_data(driver):
    """
    Handles infinite scrolling on the commentary page to load all data.
    """
    print("    Scrolling to load all commentary...")
    stale_count = 0
    try:
        body = driver.find_element(By.TAG_NAME, 'body')
    except NoSuchElementException:
        print("    Could not find body element to scroll.")
        return []

    while stale_count < 5: 
        count_before = len(driver.find_elements(By.CSS_SELECTOR, COMMENTARY_CONTAINER_SELECTOR))
        body.send_keys(Keys.PAGE_DOWN)
        time.sleep(5)
        count_after = len(driver.find_elements(By.CSS_SELECTOR, COMMENTARY_CONTAINER_SELECTOR))

        if count_after > count_before:
            print(f"    Loaded new content. Total entries now: {count_after}")
            stale_count = 0
        else:
            stale_count += 1
            print(f"    Scroll did not load new content. Stale count: {stale_count}")

    print("    Scrolling complete. Extracting commentary data...")
    all_commentary_entries = driver.find_elements(By.CSS_SELECTOR, COMMENTARY_CONTAINER_SELECTOR)
    raw_data = []
    for entry in all_commentary_entries:
        try:
            raw_data.append({
                'over': entry.find_element(By.CSS_SELECTOR, OVER_SELECTOR).text,
                'runs_indicator': entry.find_element(By.CSS_SELECTOR, RUNS_INDICATOR_SELECTOR).text,
                'ball_info': entry.find_element(By.CSS_SELECTOR, BALL_INFO_SELECTOR).text,
                'commentary_text': entry.find_element(By.CSS_SELECTOR, COMMENTARY_TEXT_SELECTOR).text
            })
        except NoSuchElementException:
            continue
    print(f"    Extracted {len(raw_data)} commentary entries.")
    return raw_data

def navigate_to_commentary_and_scrape(driver, scorecard_url):
    """
    Orchestrates the scraping for a single match: gets winner info, team info,
    then navigates to commentary and scrapes ball-by-ball data.
    """
    try:
        print(f"\n  Processing Match: {scorecard_url}")
        driver.get(scorecard_url)

        second_team = get_second_innings_team(driver)
        winner_info = get_winner_info(driver)
        
        
        if 'won' not in winner_info:
            print(f"  -> Match result does not contain 'won' ('{winner_info}'). Skipping this scorecard.")
            return None
        commentary_tab_selector = "a[href*='/ball-by-ball-commentary']"
        commentary_tab = WebDriverWait(driver, 15).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, commentary_tab_selector))
        )
        commentary_tab.click()
        print("  -> Clicked 'Commentary' tab.")
        
        WebDriverWait(driver, 15).until(EC.url_contains("ball-by-ball-commentary"))
        time.sleep(2)

        try:
            new_button_selector = "//span[contains(@class, 'ds-cursor-pointer') and .//span[text()='New']]"
            new_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, new_button_selector))
            )
            new_button.click()
            print("  -> Clicked 'New' to sort commentary chronologically.")
            time.sleep(3)
        except TimeoutException:
            print("  -> Could not find 'New' button or commentary is already sorted.")

        match_id = int(re.search(r'/match/(\d+)(?:\.html)?', scorecard_url).group(1))
        raw_commentary = scroll_and_scrape_data(driver)
        match_df = parse_commentary_data(raw_commentary, match_id, second_team, winner_info)
        
        print(f"  -> Successfully created DataFrame for match {match_id} with {len(match_df)} rows.")
        return match_df
        
    except Exception as e:
        print(f"  -> An error occurred for {scorecard_url}. Error: {e}")
        return None

if __name__ == "__main__":
    driver = initialize_driver()
    all_matches_dfs = []
    try:
        scorecard_urls = get_scorecard_links(driver, RESULTS_TABLE_URL)

        for url in scorecard_urls:  
            match_dataframe = navigate_to_commentary_and_scrape(driver, url)
            if match_dataframe is not None and not match_dataframe.empty:
                all_matches_dfs.append(match_dataframe)

        print("\n--- Scraping Workflow Complete ---")
        if all_matches_dfs:
            final_df = pd.concat(all_matches_dfs, ignore_index=True)
            print(f"Successfully combined data for {len(all_matches_dfs)} matches.")
            print("Final DataFrame shape:", final_df.shape)
            print("\nFinal DataFrame head (with new columns):")
            print(final_df.head())
            print("\nFinal DataFrame columns:")
            print(final_df.columns)
            
            # To save the final result to a CSV file:
            # final_df.to_csv("all_matches_commentary_enhanced.csv", index=False)
            # print("\nData saved to all_matches_commentary_enhanced.csv")
        else:
            print("No data was scraped.")
            
    finally:
        driver.quit()
        print("\nWebDriver closed.")

