# app/services/pdf_translation_service.py
import io
import fitz  # PyMuPDF

def dummy_translate(text: str, target_lang: str, src_lang: str | None = None) -> str:
    """
    占位的翻译函数，先直接返回原文。
    你要接大模型/DeepL/你已有的 transcribe/chatgpt 服务，改这里就行。
    """
    return text

def translate_pdf_bytes(pdf_bytes: bytes, target_lang: str, src_lang: str | None = None) -> io.BytesIO:
    """
    把 PDF 的二进制翻成目标语言，返回一个 BytesIO（给 FastAPI StreamingResponse 用）
    """
    # 打开原 PDF
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    # 我们要在内存里写一个新的 PDF
    out_pdf = fitz.open()

    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        # 新建一页，大小跟原来一致
        new_page = out_pdf.new_page(width=page.rect.width, height=page.rect.height)

        # 先把原页面画进来（背景/图片等都保留）
        # 这里用一个 trick：把原页面作为一个 PDF 内容插入
        new_page.show_pdf_page(new_page.rect, doc, page_index)

        # 然后提取文本块，再覆盖写文本
        blocks = page.get_text("blocks")
        for block in blocks:
            x0, y0, x1, y1, text, *_ = block
            if not text.strip():
                continue

            translated = dummy_translate(text, target_lang, src_lang)

            # 覆盖写文字：为了简单这里直接写到左上角
            # 想更精准就先画一个白矩形再写
            new_page.insert_text(
                (x0, y0),
                translated,
                fontsize=10,  # 你可以根据原 block 字号估一下，这里先定死
            )

    # 输出成 bytes
    out_stream = io.BytesIO()
    out_pdf.save(out_stream)
    out_pdf.close()
    doc.close()
    out_stream.seek(0)
    return out_stream
