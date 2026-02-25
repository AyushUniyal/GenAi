"""
YouTube Transcript Chatbot
──────────────────────────
Fetches a YouTube transcript, chunks it, stores it in a FAISS vector
store, and answers questions grounded strictly in that transcript.
"""

import sys
import re
from urllib.parse import urlparse, parse_qs

from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Config ──────────────────────────────────────────────────────────────────

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 200
TOP_K         = 10

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant that answers questions based ONLY on the
YouTube transcript provided below. If the answer cannot be found in the
transcript, say "I don't know based on the transcript."

Transcript context:
{context}

Question: {question}

Answer:""",
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_video_id(url_or_id: str) -> str:
    """Accept a full YouTube URL or a bare video ID and return the video ID."""
    # Already a plain ID (11 alphanumeric/dash/underscore chars)
    if re.fullmatch(r"[\w-]{11}", url_or_id):
        return url_or_id

    parsed = urlparse(url_or_id)

    # youtu.be/<id>
    if parsed.netloc in ("youtu.be", "www.youtu.be"):
        return parsed.path.lstrip("/")

    # youtube.com/watch?v=<id>
    qs = parse_qs(parsed.query)
    if "v" in qs:
        return qs["v"][0]

    raise ValueError(f"Cannot extract video ID from: {url_or_id!r}")


def fetch_transcript(video_id: str, languages: list[str] = ["en"]) -> str:
    """Download and join the transcript for a given video ID."""
    api  = YouTubeTranscriptApi()
    data = api.fetch(video_id, languages=languages)
    return "  ".join(chunk.text for chunk in data)


def build_retriever(transcript: str):
    """Chunk the transcript, embed it, and return a FAISS retriever."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.create_documents([transcript])
    print(f"  ✓ {len(chunks)} chunks created")

    store = FAISS.from_documents(documents=chunks, embedding=OpenAIEmbeddings())
    return store.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})


def build_chain(retriever):
    """Compose the RAG chain: retrieve → format → prompt → model → parse."""
    format_context = RunnableLambda(
        lambda docs: "\n\n".join(doc.page_content for doc in docs)
    )

    parallel = RunnableParallel(
        context=retriever | format_context,
        question=RunnablePassthrough(),
    )

    return parallel | PROMPT | ChatOpenAI() | StrOutputParser()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Get video URL / ID ------------------------------------------------
    raw = input("Enter YouTube URL or video ID: ").strip()
    if not raw:
        raw = "T-D1OfcDW1M"   # fallback demo video

    print("\n📥 Fetching transcript…")
    video_id   = extract_video_id(raw)
    transcript = fetch_transcript(video_id)
    print(f"  ✓ Transcript fetched ({len(transcript):,} characters)")

    # ── 2. Build the RAG pipeline --------------------------------------------
    print("\n🔧 Building vector store…")
    retriever = build_retriever(transcript)
    chain     = build_chain(retriever)
    print("  ✓ Ready!\n")

    # ── 3. Interactive Q&A loop -----------------------------------------------
    print("💬 Ask questions about the video (type 'exit' to quit)\n")
    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        answer = chain.invoke(question)
        print(f"\nBot: {answer}\n")


if __name__ == "__main__":
    main()