# ESPNcricinfo Commentary Scraper

This Python script uses Selenium to scrape ball-by-ball commentary from an ESPNcricinfo match page. It automatically scrolls down the page to load the complete commentary and then extracts the text.

---

## 📋 Prerequisites

Before you run this script, you need to have the following installed:

* **Python 3**
* **Google Chrome** browser
* **ChromeDriver**: The version must match your installed Google Chrome version. You can download it from the [Chrome for Testing availability dashboard](https://googlechromelabs.github.io/chrome-for-testing/).
* **Required Python Libraries**:
    ```bash
    pip install selenium
    ```

---

## ⚙️ Setup and Configuration

1.  **Download ChromeDriver**: Download the correct version of ChromeDriver for your operating system and Chrome browser version.
2.  **Update the Script**: Open the Python script and update the `CHROME_DRIVER_PATH` variable with the absolute path to the `chromedriver.exe` file you just downloaded.

    ```python
    # update this with the location of your chromedriver
    CHROME_DRIVER_PATH = r"C:\path\to\your\chromedriver-win64\chromedriver.exe"
    ```

3.  **Set the URL**: You can change the `URL` variable to the ball-by-ball commentary page of any other match you wish to scrape.

---

## ▶️ How to Run

1.  Make sure you have completed the setup steps above.
2.  Run the script from your terminal:
    ```bash
    python your_script_name.py
    ```
3.  The script will open a Chrome window, navigate to the URL, and begin scrolling. Once it reaches the end of the commentary, it will print the total number of entries collected and close the browser.

---

## 📄 Output

* The script prints the total count of commentary entries collected to the console.
* The collected text can be optionally saved to a CSV file by uncommenting the last two lines of the script.
