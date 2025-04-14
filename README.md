# 📊 Hotel Booking RAG-Based Query System

This project is an end-to-end **Retrieval-Augmented Generation (RAG)** system designed for analytical and natural language querying over hotel booking data. It combines **SQLite**, **ChromaDB**, **custom analytics**, and **LLMs** (LLaMA2/Gemini) to give contextual and intelligent responses from both structured and unstructured data.

---

## 🚀 Features

- 📂 **CSV Ingestion** → Structured storage in **SQLite**
- 📈 **Custom Analytics** → Generated from raw hotel booking data
- 🧠 **ChromaDB Vector Indexing** → Indexing of analytics descriptions, titles, and questions
- 🔍 **Semantic Query Search** → Find relevant analytics from vector DB
- 💬 **LLM-powered Query Response** → Uses **LLaMA2 locally** or **Gemini API fallback**
- 📦 **Two API Endpoints**:
  - `POST /ask`: Get insights using LLM and table context
  - `POST /analytics`: Return tables + metadata as JSON for analytics UI

---


## 🛠️ Architecture Overview

                     ┌────────────────────┐
                     │  CSV Raw Data      │
                     └────────┬───────────┘
                              │
                              ▼
                     ┌────────────────────┐
                     │   SQLite Database   │
                     └────────┬───────────┘
                              │
                 ┌────────────▼────────────┐
                 │  Custom Analytics Logic │
                 └────────────┬────────────┘
                              ▼
                  ┌────────────────────────┐
                  │  ChromaDB Vector Store  │
                  └────────┬───────────────┘
                           ▼
    ┌────────────────────────────────────────────┐
    │      /ask (Query → Tables → LLM Answer)    │
    │      /analytics (Query → Tables + Metadata)│
    └────────────────────────────────────────────┘


## 🧪 APIs

### `GET /health_check`
Returns system status only if all major services (SQLite, ChromaDB, LLM) are initialized properly.

### `POST /ask`

**Request:**
```json
{
  "req_id": "123",
  "query": "What is the average revenue by market segment?"
  "descriptive": true
}
```
**Response:**

```json
{
  "response": {
    "result": "...",
    "analysis": "...",
    "insights": "..."
  }
}
```
### `POST /analytics`
**Request:**
```json
{
  "query": "Show me revenue trends over time",
  "top_k": 2
}
```
**Response:**

```json
{
  "analytics": [
    {
      "table_name": "monthly_revenue_trend",
      "analytics_title": "Revenue Trends",
      "description": "Monthly revenue patterns showing hotel earnings...",
      "key_metrics": "Total revenue, bookings, ADR",
      "data": {...},
    }
  ]
}
```

## 🧠 LLM Strategy
✅ **Primary**: LLaMA2 from HuggingFace (custom local loading)

🔁 **Fallback**: Gemini Pro API via REST call

🔍 **Based on table context + user query**

## 📁 Project Structure
```
.
├── API.py                    # FastAPI application
├── utils/
│   ├── sqlite_helper.py      # SQLite data interaction
│   ├── vector_db_helper.py   # ChromaDB functions
│   ├── get_analytics.py      # Analytics table logic
│   └── local_LLM.py          # LLaMA2/Gemini handler
├── preprocessed_data.csv     # Final cleaned CSV
├── analytics_description.txt # Text for vector indexing
├── requirements.txt          # All dependencies
```

## 📦 Installation
git clone https://github.com/your-username/hotel-booking-rag.git
cd hotel-booking-rag
conda create -n rag_env python=3.10
conda activate rag_env
pip install -r requirements.txt

## ⚙️ Environment Setup

Before running the application, create a `.env` file in the root directory with the following variables:

```env
GEMINI_API_KEY=your_gemini_api_key_here
chroma_db_collection=your_chroma_collection_name
data_db=hotel_bookings.db
data_db_table=hotel_booking_data
```

## 📍 TODO / Improvements
### ⏳ To make this for more generalized dataset and use more generalized data analytics

### ⏳ Plot and send visualizations for /analytics

### ⏳ Add support for CSV upload via API

### ⏳ Streamed response with chunked LLM answers

### ⏳ Srore databased in S3 and fetch from there

### ⏳ Real time Data support

### ⏳ UI integration (React/Vue)
