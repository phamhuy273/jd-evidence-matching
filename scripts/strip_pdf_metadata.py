import pypdf

def strip_metadata(pdf_path):
    reader = pypdf.PdfReader(pdf_path)
    writer = pypdf.PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    
    # Strip metadata entirely
    writer._info = None
    
    # Save back
    with open(pdf_path, 'wb') as f:
        writer.write(f)
    
    # Verify
    verify_reader = pypdf.PdfReader(pdf_path)
    print(f"Stripped {pdf_path}: metadata = {verify_reader.metadata}")

if __name__ == '__main__':
    strip_metadata('paper/main.pdf')
