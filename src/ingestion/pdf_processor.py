from pathlib import Path
import pymupdf as fitz


def extract_pdf_content(pdf_path: str, image_output_dir: str = "data/extracted_images"):
    """
    Extract page-wise text and embedded images from a research PDF.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    image_dir = Path(image_output_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    document = fitz.open(pdf_path)
    pages = []
    extracted_images = []

    try:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            pages.append({
                "page_number": page_number,
                "text": text,
                "word_count": len(text.split())
            })

            for image_index, image_info in enumerate(page.get_images(full=True), start=1):
                xref = image_info[0]
                image_data = document.extract_image(xref)

                if not image_data:
                    continue

                extension = image_data.get("ext", "png")
                image_path = image_dir / (
                    f"{pdf_path.stem}_page_{page_number}_image_{image_index}.{extension}"
                )

                image_path.write_bytes(image_data["image"])

                extracted_images.append({
                    "page_number": page_number,
                    "image_path": str(image_path),
                    "extension": extension
                })

    finally:
        document.close()

    return {
        "pdf_name": pdf_path.name,
        "total_pages": len(pages),
        "total_words": sum(page["word_count"] for page in pages),
        "pages": pages,
        "images": extracted_images
    }

