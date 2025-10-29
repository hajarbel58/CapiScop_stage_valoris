from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def extract_tickers(url, driver_path=None, headless=True, timeout=15):
    # Configuration du navigateur
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-dev-shm-usage')

    service = ChromeService(driver_path) if driver_path else ChromeService()
    driver = webdriver.Chrome(service=service, options=options)

    try:
        driver.get(url)

        # Attendre que le container du ticker soit présent
        ticker_container = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((
                By.CSS_SELECTOR,
                "div.flex.reverse.w-fit.md\\:hover\\:pause-animation.scrollbar-hide"
            ))
        )

        # Les éléments 'div > div' contiennent le contenu texte (tickers)
        inner_divs = ticker_container.find_elements(By.CSS_SELECTOR, "div > div")
        tickers = [d.text.strip() for d in inner_divs if d.text.strip()]
        return tickers

    finally:
        driver.quit()

if __name__ == "__main__":
    url = "https://www.casablanca-bourse.com/fr/live-market/marche-cash/indices"
    # Remplacez par le chemin vers votre ChromeDriver, si nécessaire
    driver_path = r"C:\Users\HP\Documents\ffmpeg\chromedriver\chromedriver-win64\chromedriver.exe"
    tickers = extract_tickers(url, driver_path=driver_path, headless=True)
    print("Tickers présents dans le scroller :", tickers)
