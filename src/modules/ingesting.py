from langchain_community.document_loaders import PyMuPDFLoader
# # from langchain_community.document_loaders import OnlinePDFloader                  # To read from the web

def ingesting_PDF(pdfs):

    if pdfs:        
        loader = PyMuPDFLoader(pdfs)
        docs = loader.load()
        print('Document loaded successfully!!!')
        return docs  # Return the loaded document
    else:
        print('Upload a PDF file')
        return None
