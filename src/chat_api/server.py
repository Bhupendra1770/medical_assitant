import os, time, logging
from contextlib import asynccontextmanager
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from src.stt_tts_loop.response_generator.medical_response import get_generator, initialize_generator
from src.rag.rag_service import get_rag_service, initialize_rag
import io
import pypdf


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
API_KEY = os.getenv("CHAT_API_KEY", "dev-key")

def check_auth(r): return r.headers.get("x-api-key") == API_KEY

@asynccontextmanager
async def lifespan(app):
    from dotenv import load_dotenv; load_dotenv()
    groq_key = os.getenv("GROQ_API_KEY", "")
    db_url   = os.getenv("DATABASE_URL", "")
    rag_service = None
    if db_url:
        ok = await initialize_rag(db_url)
        rag_service = get_rag_service() if ok else None
    initialize_generator(groq_key, rag_service)
    logger.info("MediAssist HTTP server ready")
    yield
    logger.info("Shutdown")

async def health_handler(request):
    rag = get_rag_service()
    return JSONResponse({"status":"ok","rag_enabled":rag is not None,"document_count":rag.get_document_count() if rag else 0})

async def chat_handler(request):
    if not check_auth(request): return JSONResponse({"error":"Unauthorized"},status_code=401)
    try: body = await request.json()
    except: return JSONResponse({"error":"Invalid JSON"},status_code=400)
    msg = body.get("message","").strip() if isinstance(body,dict) else ""
    if not msg: return JSONResponse({"error":"message required"},status_code=400)
    gen = get_generator()
    if not gen: return JSONResponse({"error":"not ready"},status_code=503)
    reply = await gen.generate(msg, is_voice=False)
    return JSONResponse({"reply":reply})

async def upload_document_handler(request):
    if not check_auth(request): return JSONResponse({"error":"Unauthorized"},status_code=401)
    rag = get_rag_service()
    if not rag: return JSONResponse({"error":"RAG not available"},status_code=503)
    
    form  = await request.form()
    title = str(form.get("title","Medical Document"))
    
    if "file" in form:
        file_obj  = form["file"]
        raw_bytes = await file_obj.read()
        filename  = file_obj.filename.lower()

        if filename.endswith(".pdf"):
            try:
                reader  = pypdf.PdfReader(io.BytesIO(raw_bytes))
                content = "\n".join(
                    page.extract_text() or "" for page in reader.pages
                )
                if not content.strip():
                    return JSONResponse({"error": "Could not extract text from PDF — it may be scanned or image-based"}, status_code=400)
            except Exception as e:
                return JSONResponse({"error": f"PDF read error: {str(e)}"}, status_code=400)
        else:
            content = raw_bytes.decode("utf-8", errors="ignore")

    else:
        content = str(form.get("content", ""))

    if not content.strip(): return JSONResponse({"error":"No content"},status_code=400)
    ok = rag.add_document(content, title=title)
    return JSONResponse({"success":ok,"message":f"'{title}' added","total_documents":rag.get_document_count()})


async def list_documents_handler(request):
    if not check_auth(request): return JSONResponse({"error":"Unauthorized"},status_code=401)
    rag = get_rag_service()
    docs = rag.list_documents() if rag else []
    return JSONResponse({"documents":docs,"total":len(docs)})

async def delete_document_handler(request):
    if not check_auth(request): return JSONResponse({"error":"Unauthorized"},status_code=401)
    rag = get_rag_service()
    if not rag: return JSONResponse({"error":"RAG not available"},status_code=503)
    return JSONResponse({"success":rag.delete_document(int(request.path_params.get("doc_id",0)))})

routes=[
    Route("/health",health_handler,methods=["GET"]),
    Route("/chat",chat_handler,methods=["POST"]),
    Route("/upload-document",upload_document_handler,methods=["POST"]),
    Route("/documents",list_documents_handler,methods=["GET"]),
    Route("/documents/{doc_id:int}",delete_document_handler,methods=["DELETE"]),
]
app = Starlette(lifespan=lifespan, routes=routes)
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=True,allow_methods=["*"],allow_headers=["*"])