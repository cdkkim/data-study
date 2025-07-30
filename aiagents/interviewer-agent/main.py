import pdb

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from company_summarizer_mcp import get_graph, CompanyState

app = FastAPI(
    title="Company Summarizer MCP API",
    description="Summarizes companies using LangGraph and RAG.",
    version="1.0.0"
)

# Request body schema
class SummarizeCompanyRequest(BaseModel):
    company_name: str


# Response schema
class SummarizeResponse(BaseModel):
    company_name: str
    summary: str


@app.post("/company-summarize", response_model=SummarizeResponse)
def summarize_company(request: SummarizeCompanyRequest):
    try:
        graph = get_graph()
        state = CompanyState(company_name=request.company_name)
        result = graph.invoke(state)
        pdb.set_trace()
        return SummarizeResponse(
            company_name=request.company_name,
            summary=result.get('summary'),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

