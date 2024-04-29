
from PyPDF2 import PdfReader
import pdfplumber

# from langchain_community.document_loaders import PyPDFLoader

from langchain.schema.document import Document

class PDFManager:

    @staticmethod
    def get_docs_from_text(pdf_docs):
      docs = []
      for pdf in pdf_docs:
          reader = PdfReader(pdf)
          for i, page in enumerate(reader.pages):

              text = page.extract_text()

              #TODO: process equations

              metadata = {'page_number': i + 1, 'source': pdf.name, 'type': 'text', 'format': 'pdf'}
              doc = Document(page_content=text, metadata=metadata)

              docs.append(doc)

      return docs


    @staticmethod
    def extract_images(pdf_docs):
        images = []

        for pdf in pdf_docs:
            reader = PdfReader(pdf)

            #FIXME: handle non-rectangular images
            
            count = 0
            for page in reader.pages:

                for image_file_object in page.images:
                    images.append(image_file_object.data)

                    # with open(str(count) + image_file_object.name, "wb") as fp:
                    #     fp.write(image_file_object.data)
                    #     count += 1 

        return images

    @staticmethod
    def extract_tables(pdf_docs):
        tables = []
        for pdf in pdf_docs:
            with pdfplumber.open(pdf) as pdf:
                for page in pdf.pages:
                    tables.extend(page.extract_tables())

        return tables

