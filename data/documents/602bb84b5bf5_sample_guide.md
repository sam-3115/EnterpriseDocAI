# Developer Onboarding Guide

## Architecture Overview
The EnterpriseDoc AI platform uses a dual-engine RAG architecture combining dense semantic retrieval with sparse keyword matching. Documents are indexed into ChromaDB vector database and SQLite metadata store.

## Getting Started
To run the platform locally, install dependencies using Python 3.11 and configure your Gemini API key in the `.env` file. The backend runs on FastAPI at port 8000, and the frontend dashboard runs on Streamlit at port 8501.

## Security and Role-Based Access Control
The system enforces strict RBAC separation:
- Admin users can upload, inspect, and delete enterprise documents.
- Standard users have search and read-only query capabilities across the knowledge repository.

