from fastapi import FastAPI
from fastapi.responses import JSONResponse
from utils.local_LLM import LLM
from utils.vector_db_helper import VecDBHelper
from utils.get_analytics import BookingAnalytics
from utils.sqlite_helper import SQLHelper
from pydantic import BaseModel
import uvicorn
import re
import json
app = FastAPI()

class query(BaseModel):
    req_id: str
    query: str
    descriptive:bool

class AnalyticsQuery(BaseModel):
    query: str
    top_k: int = 2 

preprocessed_csv = "/home/vaibhav/Downloads/Hotel_Booking_RAG_QnA/utils/preprocessed_data.csv"

health_status = {
    "sqlhelper": False,
    "vecdb": False,
    "analytics": False,
    "llm": False
}

try:
    sqlhelper = SQLHelper()
    if not sqlhelper.check_data_table():
        sqlhelper.create_data_table(csv_path=preprocessed_csv)
        print("Initial Data sent to DB!!")
    else:
        print("Database is up!!")
    health_status["sqlhelper"] = True
except Exception as e:
    print(f"[ERROR] SQLHelper init failed: {e}")

try:
    vecDB = VecDBHelper()
    health_status["vecdb"] = True
except Exception as e:
    print(f"[ERROR] VecDBHelper init failed: {e}")

try:
    analytics = BookingAnalytics()
    analytics.create_all_analytics()
    health_status["analytics"] = True
except Exception as e:
    print(f"[ERROR] BookingAnalytics init failed: {e}")

try:
    llm = LLM(use_custom_llm=False) #Custom_llm is not tested yet due to resource restriction, It will be updated soon after testing on Colab/Kaggle. 
    health_status["llm"] = True
except Exception as e:
    print(f"[ERROR] LLM init failed: {e}")

# === Routes ===
@app.get("/health_check")
def give_status():
    if all(health_status.values()):
        return {"status": "ok", "details": health_status}
    return {"status": "fail", "details": health_status}


@app.post("/ask")
def generate_result(query: query):
    q = query.query
    tables = vecDB.get_query_results(query=q)
    print("\n[INFO] Tables retrieved from vector DB:\n", tables, "\n")
    tables_data = sqlhelper.analytical_data_fetch(tables=tables['ids'][0])
    print("[INFO] Analytical data fetched:\n", tables_data)
    
    # Get the response and debug its type and content
    response = llm.get_answer(contexts=tables_data, question=q)
    print(f"[DEBUG] Response type: {type(response)}")
    print(f"[DEBUG] Response content: {response}")
    
    # If response is a string that looks like a dictionary, try to parse it
    if isinstance(response, str) and response.startswith("{") and response.endswith("}"):
        try:
            response_dict = json.loads(response)
            if 'response' in response_dict:
                json_str = response_dict['response']
            else:
                json_str = response  
        except json.JSONDecodeError:
            json_str = response
    elif isinstance(response, dict) and 'response' in response:
        json_str = response['response']
    else:
        json_str = str(response)
    
    try:
        data = json.loads(json_str)
        result = data.get('result')
        analysis = data.get('analysis')
        insights = data.get('insights')
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse JSON: {e}")
        # If we can't parse as JSON, just use the raw text
        result = json_str
        analysis = ""
        insights = ""

    if query.descriptive:
        return {"response": f"{result}\n{analysis}\n{insights}"}
    else:
        return {"response": result}
    
    
@app.post("/analytics")
def get_analytics(query: AnalyticsQuery):
    try:
        # Step 1: Vector Search
        result = vecDB.collection.query(
            query_texts=[query.query],
            n_results=query.top_k,
            include=["documents", "metadatas"]
        )

        table_ids = result['ids'][0]  # List of table names
        metadatas = result['metadatas'][0]  # Corresponding metadata list

        # Step 2: Fetch table data from SQLite
        table_data = sqlhelper.analytical_data_fetch(tables=table_ids)

        # Step 3: Combine with metadata
        response_payload = []
        for table_name, metadata in zip(table_ids, metadatas):
            matching_table = next((t[table_name] for t in table_data if table_name in t), {})
            response_payload.append({
                "table_name": table_name,
                "analytics_title": metadata.get("analytics_title"),
                "description": metadata.get("description", ""),
                "key_metrics": metadata.get("key_metrics"),
                "data": matching_table
            })

        return JSONResponse(content={"analytics": response_payload}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    

if __name__ == "__main__":
    uvicorn.run("API:app", port=5000, log_level="info")
