# app/api/pdf_translation.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from app.services.pdf_translation_service import translate_pdf_bytes

router = APIRouter(tags=["pdf-translation"])

@router.post("/pdf/translate", response_class=StreamingResponse)
async def translate_pdf(
    file: UploadFile = File(...),
    target_lang: str = Form(...),
    src_lang: str | None = Form(None)
):
    """
    接收一个 PDF，翻译后返回一个新的 PDF（二进制）。
    Java 那边就是 multipart/form-data 调这个接口。
    """
    if file.content_type not in ("application/pdf",):
        raise HTTPException(status_code=400, detail="只能上传 PDF 文件")

    original_bytes = await file.read()

    try:
        translated_bytes = translate_pdf_bytes(
            original_bytes,
            target_lang=target_lang,
            src_lang=src_lang
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF 翻译失败: {e}")

    return StreamingResponse(
        content=translated_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=translated.pdf"}
    )
