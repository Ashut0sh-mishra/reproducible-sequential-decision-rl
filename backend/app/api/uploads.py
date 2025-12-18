# from fastapi import APIRouter, UploadFile, HTTPException
# import os
# import pandas as pd
# import io
# from datetime import datetime

# router = APIRouter(prefix="/upload", tags=["Upload"])

# UPLOAD_PATH = "uploaded_data"
# os.makedirs(UPLOAD_PATH, exist_ok=True)

# @router.post("/")
# async def upload_file(file: UploadFile):
#     contents = await file.read()

#     # Parse CSV/XLSX
#     try:
#         if file.filename.endswith(".csv"):
#             df = pd.read_csv(io.BytesIO(contents))
#         elif file.filename.endswith(".xlsx"):
#             df = pd.read_excel(io.BytesIO(contents))
#         else:
#             raise HTTPException(400, "Only CSV/XLSX allowed")
#     except Exception as e:
#         raise HTTPException(400, f"Error parsing file: {str(e)}")

#     # Save file
#     file_path = os.path.join(UPLOAD_PATH, file.filename)
#     with open(file_path, "wb") as f:
#         f.write(contents)

#     return {
#         "id": hash(file.filename),
#         "filename": file.filename,
#         "size": os.path.getsize(file_path),
#         "uploaded_at": datetime.now().isoformat(),
#         "rows": len(df),
#         "columns": df.columns.tolist()
#     }

# @router.get("/list")
# def list_files():
#     files = []
#     for name in os.listdir(UPLOAD_PATH):
#         full = os.path.join(UPLOAD_PATH, name)
#         files.append({
#             "id": hash(name),
#             "filename": name,
#             "size": os.path.getsize(full),
#             "uploaded_at": datetime.fromtimestamp(os.path.getmtime(full)).isoformat()
#         })
#     return files
from fastapi import APIRouter, UploadFile, File, HTTPException
import os
import pandas as pd
import io
from datetime import datetime

router = APIRouter(prefix="/upload", tags=["Upload"])

UPLOAD_PATH = "uploaded_data"
os.makedirs(UPLOAD_PATH, exist_ok=True)

@router.post("/")
async def upload_file(file: UploadFile = File(...)):  # ✅ FIX
    contents = await file.read()

    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Only CSV/XLSX allowed")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing file: {str(e)}")

    file_path = os.path.join(UPLOAD_PATH, file.filename)
    with open(file_path, "wb") as f:
        f.write(contents)

    return {
        "id": hash(file.filename),
        "filename": file.filename,
        "size": os.path.getsize(file_path),
        "uploaded_at": datetime.now().isoformat(),
        "rows": len(df),
        "columns": df.columns.tolist()
    }

@router.get("/list")
def list_files():
    files = []
    for name in os.listdir(UPLOAD_PATH):
        full = os.path.join(UPLOAD_PATH, name)
        files.append({
            "id": hash(name),
            "filename": name,
            "size": os.path.getsize(full),
            "uploaded_at": datetime.fromtimestamp(os.path.getmtime(full)).isoformat()
        })
    return files
