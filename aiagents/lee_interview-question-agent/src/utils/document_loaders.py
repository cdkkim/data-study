import httpx
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from typing import List, Union, Any, Optional
import os
import io
from unstructured.partition.auto import partition # pyhwp가 안됨 임시 
from langchain_unstructured import UnstructuredLoader

async def _load_pdf_document(source: Union[str, bytes], file_type_hint: str = "pdf", file_name: str="file_name") -> List[Document]:
    """PDF 파일 (경로 또는 bytes)에서 문서를 로드한다."""
    print(f"PDF 파일 로드 중 : {file_type_hint} (Source Type : {type(source)})")

    elements: List[Any] = []
    # 파일 확장자를 이용하여 unstructured가 파일 타입을 추론하도록 돕는다.

    try:
        # unstructured.partition 함수에 io.BytesIO 객체를 직접 전달한다.
        # file_filename 인자를 통해 원본 파일명을 알려주면 unstructured가 타입 추론에 도움을 받는다.
        elements = partition( # partition 함수가 비동기적으로 동작하는 경우 await 필요
            file=io.BytesIO(source),
            file_filename=file_name, # 원본 파일명 전달
            # content_type=f"application/{file_type_hint}" if file_type_hint != "unknown" else None, # 선택 사항
            # strategy='hi_res', # 필요시 더 정교한 파싱 전략 사용
            # timeout=60.0 # 파싱 타임아웃
        )

        text_content = "\n\n".join([str(el) for el in elements if hasattr(el, 'text') and el.text is not None and el.text.strip()])

        if not text_content.strip():
            print(f"경고: unstructured로 바이트 데이터에서 유의미한 텍스트를 추출하지 못했습니다: {file_name}")
            return []

        doc = Document(page_content=text_content, metadata={"source": file_name, "type": f"bytes_{file_type_hint}", "file_type_hint": file_type_hint})
        print(f"바이트 데이터에서 텍스트 로드 완료 (unstructured 기반, 길이: {len(text_content)}).")
        return [doc]

    except Exception as e:
        print(f"바이트 데이터 문서 파싱 중 오류 발생 (unstructured, 파일명: {file_name}): {e}")
        return []



def _load_text_document(source: Union[str, bytes], file_type_hint: str = "txt", file_name: str="file_name") -> List[Document]:
    """일반 텍스트/기타 파일(경로 또는 bytes)에서 문서를 로드한다."""
    print(f"일반 텍스트/기타 파일 로드 중 {file_type_hint}")
    
    loader = None
    if isinstance(source, str):
        loader = UnstructuredLoader(source) 
    elif isinstance(source, bytes): 
        # UnstructuredFileLoader는 file_contents 인자에 bytes를 직접 받을 수 있다.
        # 또는 file_obj 인자에 io.BytesIO 객체를 받을 수도 있다.
        loader = UnstructuredLoader(file_contents=source, mode="elements",
                                        file_filename=f"uploaded_file.{file_type_hint}") # 파일명 힌트 제공
    else:
        raise ValueError("Unsupported source type for text/other: Must be str (path) or bytes.")
    return loader.load()


def _load_hwp_document(source: Union[str, bytes], file_type_hint: str = "hwp", file_name: str="file_name") -> List[Document]:
    """
    HWP 파일 (경로 또는 bytes)에서 문서를 로드한다 (unstructured 사용).
    """
    print(f"HWP 파일 로드 중 (unstructured): {file_type_hint} (Source Type: {type(source)})")
    
    elements: List[Any] = [] # unstructured 파싱 결과는 element 객체 리스트이다.

    try:
        if isinstance(source, str): # 파일 경로인 경우
            # partition 함수는 파일 경로를 직접 받을 수 있다.
            elements = partition(filename=source, content_type="application/x-hwp")
            source_identifier = source
        elif isinstance(source, bytes): # bytes 데이터인 경우
            # partition 함수는 io.BytesIO 객체를 받을 수 있다.
            # file_filename은 원본 파일명을 알려주어 unstructured가 파일 타입을 추론하는 데 도움을 준다.
            elements = partition(file=io.BytesIO(source), content_type="application/x-hwp")
            source_identifier = "in_memory_bytes"
        else:
            raise ValueError("Unsupported source type for HWP: Must be str (path) or bytes.")

        # unstructured의 elements에서 텍스트 내용을 추출하고 Document 객체로 변환
        full_text = "\n\n".join([str(el) for el in elements if hasattr(el, 'text') and el.text is not None])
        # hasattr : 속성을 가지고 있는지 여부를 확인하는 메서드

        if not full_text.strip(): # 추출된 텍스트가 비어있는 경우
            print(f"경고: HWP 파일에서 유효한 텍스트를 추출하지 못했습니다. ({file_type_hint}, Source Type: {type(source)})")
            return [] # 빈 리스트 반환

        # 메타데이터에 unstructured가 제공하는 정보를 추가할 수도 있다. (선택 사항)
        metadata = {
            "source": source_identifier,
            "type": "hwp",
            "file_type_hint": file_type_hint,
            # "elements_count": len(elements) # 디버깅용
        }
        
        doc = Document(page_content=full_text, metadata=metadata)
        print(f"HWP 파일 로드 완료 (길이: {len(full_text)}).")
        return [doc] # Document 리스트로 반환

    except Exception as e:
        print(f"HWP 파일 로드 중 오류 발생 (unstructured, {file_type_hint}, Source Type: {type(source)}): {e}")
        # 오류 발생 시 빈 리스트 반환
        return []

async def load_document_from_bytes(file_bytes: bytes, file_name: str) -> List[Document]:
    """
    파일의 raw bytes 데이터와 파일 유형 힌트를 받아 문서를 로드한다.
    FastAPI UploadFile.read() 결과를 처리하는 데 사용된다.
    """
    print(f"바이트 데이터에서 문서 로드 시작. 유형: {file_name}")

    file_type = file_name.split(".")[-1]
    try:
        if file_type.lower() == "pdf":
            return await _load_pdf_document(file_bytes, file_type, file_name)
        elif file_type.lower() == "hwp":
            return await _load_hwp_document(file_bytes, file_type, file_name)
        # UnstructuredFileLoader가 처리할 수 있는 다른 일반 텍스트/파일 형식
        elif file_type.lower() in ["txt", "doc", "docx", "ppt", "pptx", "xls", "xlsx", "csv"]:
            return await _load_text_document(file_bytes, file_type, file_name)
        else:
            print(f"경고: 알 수 없는 파일 유형 '{file_type}'. UnstructuredFileLoader로 시도합니다.")
            return await _load_text_document(file_bytes, file_type, file_name)
        
    except Exception as e:
        print(f"바이트 데이터에서 문서 로드 중 오류 발생 (Type: {file_type}): {e}")
        return []


def load_document_from_path(file_path: str) -> List[Document]:
    """
    주어진 파일 경로에서 문서를 로드한다.
    지원 형식: PDF, HWP, TXT 등 (UnstructuredFileLoader 지원 형식).
    """
    if not os.path.exists(file_path):
        print(f"오류: 파일이 존재하지 않습니다: {file_path}")
        return []

    try:
        if file_path.lower().endswith(".pdf"):
            return _load_pdf_document(file_path, "pdf")
        elif file_path.lower().endswith(".hwp"):
            return _load_hwp_document(file_path, "hwp")
        else:
            return _load_text_document(file_path, os.path.splitext(file_path)[1].lstrip('.')) # 파일 확장자로 힌트 제공
    except Exception as e:
        print(f"파일 경로에서 문서 로드 중 치명적인 오류 발생 ({file_path}): {e}")
        return []
    

async def load_document_from_url(url: str) -> List[Document]:
    """
    주어진 URL에서 웹 페이지 내용을 로드하고 unstructured + Playwright를 사용하여 핵심 정보를 파싱한다.
    실무 팁: JavaScript 렌더링 페이지는 requests만으로는 처리 불가능하며, Playwright와 같은 헤드리스 브라우저가 필수이다.
             unstructured 라이브러리가 Playwright 연동을 지원하여 동적 페이지 파싱을 간소화한다.
    """
    elements: List[Any] = []

    try:
        # httpx를 사용하여 초기 HTTP 요청을 보내는 것은 여전히 유효하지만,
        # unstructured.partition(url=url)을 사용한다면 unstructured가 내부적으로
        # 페이지 로딩 및 JavaScript 렌더링을 처리하므로, 여기서 직접 response.text를 가져올 필요는 없다.
        # 이 부분을 제거하고 바로 unstructured.partition(url=url)을 호출하는 것이 더 간결하고 명확하다.

        # unstructured의 partition 함수를 사용하여 HTML 내용 파싱
        # unstructured는 'unstructured[playwright]'가 설치되어 있으면
        # 내부적으로 Playwright를 사용하여 URL을 로드하고 JavaScript를 실행한다.
        # 이 함수가 코루틴 객체를 반환하는 경우를 대비하여 await를 붙인다.
        elements = partition( # <--- await partition(url=url)
            url=url, # URL 인자를 직접 전달
            # timeout: 파싱 작업에 대한 타임아웃 (requests.get의 timeout과 다름)
            # strategy: 'fast' (빠르게), 'hi_res' (고해상도, 더 정확하지만 느림)
            # 예를 들어, partition(url=url, strategy='hi_res', hi_res_model_name="yolox")
        )
        # unstructured elements에서 유의미한 텍스트 추출
        # unstructured는 제목, 단락, 목록 등 구조화된 elements를 반환하므로,
        # 이들을 합치면 보다 정제된 텍스트를 얻을 수 있다.
        text_content = "\n\n".join([str(el) for el in elements if hasattr(el, 'text') and el.text is not None and el.text.strip()])

        if not text_content.strip():
            print(f"경고: unstructured + Playwright로 URL에서 유의미한 텍스트를 추출하지 못했습니다: {url}")
            # 이 지점까지 왔다면 동적 콘텐츠 파싱이 실패한 것이므로,
            # BeautifulSoup을 이용한 폴백은 동적 페이지에서는 거의 의미가 없다.
            # 따라서 이 부분을 제거하거나, 매우 제한적인 경우에만 사용해야 한다.
            # 여기서는 제거하는 것을 권장한다.
            return [] # 여전히 내용이 없으면 빈 리스트 반환

        doc = Document(page_content=text_content, metadata={"source": url, "type": "web_page_unstructured_playwright"})
        print(f"URL에서 텍스트 로드 완료 (unstructured + Playwright 기반, 길이: {len(text_content)}).")
        return [doc]

    except Exception as e:
        # Playwright 관련 오류(브라우저 시작 실패 등)도 여기에 포함된다.
        print(f"URL 문서 파싱 중 오류 발생 (unstructured + Playwright, {url}): {e}")
        return []

    
    

def load_document(source: Union[str, bytes], file_type_hint: Optional[str] = None) -> List[Document]:
    """
    입력된 소스(파일 경로, URL, 또는 bytes)에 따라 적절한 로더를 호출하여 문서를 로드한다.
    """
    if isinstance(source, bytes):
        if not file_type_hint:
            raise ValueError("Bytes 데이터 처리 시 'file_type_hint'를 반드시 제공해야 합니다 (예: 'pdf', 'hwp').")
        return load_document_from_bytes(source, file_type_hint)
    elif isinstance(source, str):
        if source.startswith("http://") or source.startswith("https://"):
            return load_document_from_url(source)
        else: # 파일 경로
            return load_document_from_path(source)
    else:
        raise ValueError(f"지원되지 않는 문서 소스 유형: {type(source)}")
