import streamlit as st
import pandas as pd
import plotly.express as px
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Set page configuration
st.set_page_config(page_title="AI Investment Advisor", layout="wide")

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load AI model
@st.cache_resource()
def load_model():
    model_name = "google/gemma-2b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)

ai_pipeline = load_model()

# Function to scrape top cryptocurrencies using Selenium
def get_top_cryptos():
    try:
        url = "https://coinmarketcap.com/"

        # Setup Selenium WebDriver
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in headless mode
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("start-maximized")
        chrome_options.add_argument("disable-infobars")

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.get(url)

        # Wait for the table to load
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))

        # Extract cryptocurrency data
        cryptos = []
        rows = driver.find_elements(By.XPATH, "//table/tbody/tr")[:10]  # Top 10 cryptos
        for row in rows:
            try:
                name = row.find_element(By.XPATH, ".//td[3]//p").text  # Crypto Name
                price = row.find_element(By.XPATH, ".//td[4]//span").text  # Price
                change_24h = row.find_element(By.XPATH, ".//td[5]").text  # 24H Change
                cryptos.append([name, price, change_24h])
            except Exception as e:
                continue  # Skip rows with missing data

        driver.quit()

        # Convert data to DataFrame
        df = pd.DataFrame(cryptos, columns=["Name", "Price", "24H Change"])
        return df
    except Exception as e:
        st.error(f"Error fetching cryptocurrency data: {e}")
        return None

# UI Design
st.title("💰 AI Investment Advisor")
st.write("Enter your **monthly savings** and get AI-driven investment suggestions!")

# User input for monthly savings
monthly_savings = st.slider("💵 Monthly Savings (INR)", min_value=1000, max_value=100000, step=1000, value=5000)

# Investment categories
st.subheader("📊 Investment Breakdown")
investment_options = {
    "Mutual Funds": 0.5,
    "Fixed Deposits": 0.3,
    "Crypto": 0.2
}

# Calculate investment amounts
investment_breakdown = {key: monthly_savings * value for key, value in investment_options.items()}

# Display investment recommendations
st.write("💡 **Suggested Investment Distribution**")
for investment, amount in investment_breakdown.items():
    st.write(f"**{investment}**: ₹{amount:.2f}")

# Fetch and display top cryptocurrencies
st.subheader("🚀 Top Cryptocurrencies to Consider")
crypto_data = get_top_cryptos()
if crypto_data is not None:
    st.dataframe(crypto_data)

# Get AI-based investment advice
if st.button("📢 Get AI Financial Advice"):
    with st.spinner("🤖 AI is analyzing your financial strategy..."):
        user_query = f"My monthly savings are ₹{monthly_savings}. Suggest an investment strategy considering risk and long-term gains."
        response = ai_pipeline(user_query, max_length=300, do_sample=True)
        st.success("💡 AI Investment Advice:")
        st.write(response[0]["generated_text"])
