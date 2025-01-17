# pip install fastapi uvicorn python-multipart pydantic

# run : uvicorn main_test:app --reload
from typing import Union
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from datetime import datetime
import io

app = FastAPI()

class Item(BaseModel):
    model: str
    timestamp : datetime 
    value: float
    valve: int
# data
# [
#     {"model": "FGRWGS0.780-344-LEI-8023.F_CV", "timestamp": "2023-10-21 20:43", "value": "5.028375", "valve": "100"},
#     {"model": "FGRWGS0.780-344-LEI-8023.F_CV", "timestamp": "2023-10-21 20:44", "value": "5.027", "valve": "100"},
#     {"model": "FGRWGS0.780-344-LEI-8023.F_CV", "timestamp": "2023-10-21 20:45", "value": "5.02425", "valve": "100"},
#     {"model": "FGRWGS0.780-344-LEI-8023.F_CV", "timestamp": "2023-10-21 20:46", "value": "5.020125", "valve": "100"},
#     {"model": "FGRWGS0.780-344-LEI-8023.F_CV", "timestamp": "2023-10-21 20:47", "value": "5.025625", "valve": "100"},
#     {"model": "FGRWGS0.780-344-LEI-8023.F_CV", "timestamp": "2023-10-21 20:48", "value": "5.02975", "valve": "100"}
# ]

@app.post("/save-csv/")
async def save_csv(items: list[Item]):
    # 데이터를 Pandas DataFrame으로 변환
    data = [item.dict() for item in items]
    df = pd.DataFrame(data)
    
    # CSV 파일로 저장
    file_name = "test.csv"
    df.to_csv(file_name, index=False, encoding="utf-8")
    
    return {"message": f"CSV 파일이 '{file_name}'로 저장되었습니다.", "row_count": len(df)}

@app.post("/upload-and-save/")
async def upload_and_save(file: UploadFile = File(...)):
    file_name = f"uploaded_{file.filename}"
    with open(file_name, "wb") as f:
        f.write(await file.read())
    return {"message": f"'{file_name}'로 파일이 저장되었습니다."}