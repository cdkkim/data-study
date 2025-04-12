# Simple Web Service

Build a simple web service that reads data from [orders database](https://drive.google.com/drive/folders/16sNdHEv76dJShhucqcSVDcZ4sYWgiae9) in csv format.

## Steps

### Step 0: Create git branch

### Step 1: Install and run API server with [FastAPI](https://fastapi.tiangolo.com/)

### Step 2: Add Restful APIs

- ex1) orders count by month
- ex2) orders count by country

### Step 3: Visualize the data read from the server with [Streamlit](https://streamlit.io/) or any visualization library of your choice

For example

```python
# visualize.py
import requests
import streamlit as st

# fastapi 서버에서 데이터를 가져옵니다.
count_orders_by_date = requests.get('http://127.0.0.1:8000/count-orders-by-date').json()

# 그래프 타이틀을 "Line Chart"라고 설정합니다.
st.title("Line Chart")
# count_orders_by_date 데이터로 선 그래프를 그립니다.
st.line_chart(data=count_orders_by_date)
```

### (optional) Step 4: Install MySQL or PostgreSQL and migrate csv data into the database

### (optional) Step 5: fix web server to read data from Step 4 instead of csv files

### (optional) Step 6: Install pyspark and perform ETL


## 참고
- [fastapi-pytorch-mnist](https://github.com/cdkkim/data-study/tree/main/fastapi-pytorch-mnist)
- [dxcodingcamp-python](https://github.com/cdkkim/dxcodingcamp-python)
