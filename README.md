# 🏥 Hospital Inventory AI Agent

An AI-powered hospital inventory decision support system that monitors medication stock, predicts future demand, simulates seasonal illness scenarios, and recommends proactive reorder actions before shortages happen.

---

## 🌐 Live Demo

[Add Live Demo Link Here](https://hospital-inventory-ai-agent-dyhkdfpiiicgbwnahemrb9.streamlit.app/)

> Replace the link above with your deployed app URL once it is live.

---

## 📌 Overview

Hospitals often have inventory systems that track medications and supplies, but many systems are mainly reactive. This project adds an intelligent decision-support layer on top of inventory data to help answer questions such as:

- What medications are at risk of shortage?
- What should be reordered now?
- How does seasonal illness affect demand?
- What happens if supplier delays increase?
- Which medication categories are most impacted under different scenarios?

This system combines data analysis, forecasting, simulation, and agent-based recommendations in one interactive application.

---

## 🎯 Problem Statement

Medication shortages and delayed reorders can affect patient care and hospital operations. Traditional inventory systems may show current stock levels, but they often do not:

- forecast upcoming demand clearly  
- simulate future scenarios  
- explain why a medication is high risk  
- provide interactive decision support  

This project was built to make hospital inventory planning more proactive, explainable, and data-driven.

---

## 🚀 Features

### 📦 Inventory Monitoring
- Tracks current stock levels  
- Calculates days remaining  
- Flags low-stock and at-risk medications  
- Compares stock against reorder thresholds  

### 📈 Demand Forecasting
- Predicts future monthly demand  
- Uses data-driven seasonality learned from the dataset  
- Adjusts predictions based on patient volume and supplier delay  

### 🧪 Seasonal Scenario Simulation
- Simulates Winter, Spring, Summer, and Fall conditions  
- Adjustable seasonal illness severity  
- Models how demand changes under different scenarios  

### 🧠 AI Agent Insights
- Assigns HIGH / MEDIUM / LOW priority  
- Explains why items are risky  
- Recommends reorder actions  
- Estimates reorder quantities  

### 💬 Ask the Agent
Users can ask:
- What should I reorder now?  
- Which medication has the highest predicted demand?  
- Which department uses the most medication?  
- What season scenario is active?  
- Show me high priority items  

### 📊 Visual Dashboard
- Interactive charts  
- Color-coded stock status  
- Department usage breakdown  
- Demand by medication and category  

---

## 🛠️ Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Streamlit  
- Plotly  

---


---

## ⚙️ How It Works

### 1. Data Processing
Reads inventory-style data including:
- stock levels  
- usage rates  
- reorder thresholds  
- supplier lead times  
- seasonal/month data  

### 2. Inventory Risk Logic
Calculates:
- days remaining  
- stock risk  
- future shortage risk  
- reorder urgency  

### 3. Data-Driven Seasonality
Learns seasonal demand patterns by:
- grouping usage by season  
- calculating averages  
- comparing against baseline demand  

### 4. Forecasting
Uses a machine learning model to predict:
- monthly demand  
- projected stock duration  

### 5. Agent Recommendations
The AI agent:
- prioritizes medications  
- explains risk  
- recommends actions  
- answers user questions interactively  

---

## 📊 Example Use Cases

- Prepare inventory for winter illness season  
- Identify medications at risk of shortage  
- Simulate increased patient load  
- Simulate supplier delays  
- Support hospital decision-making  

---

## 💬 Example Questions

- What should I reorder now?  
- Which medication has the highest predicted demand?  
- Which category has the highest demand?  
- Which department uses the most medication?  
- Give me a summary  
- What does the season multiplier mean?  

---

## ▶️ How to Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/your-username/hospital-inventory-ai-agent.git
cd hospital-inventory-ai-agent

python3 -m venv venv
source venv/bin/activate
install all requirments
To run:
streamlit run app.py
