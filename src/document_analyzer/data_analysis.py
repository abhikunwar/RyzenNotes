import os
import sys
# from utils.model_loader import ModelLoader
from utils.phi3onnx import Phi3ONNX_LLM
import fitz
# from logger.custom_logger import CustomLogger
# from exception.custom_exception import DocumentPortalException
# from model.models import *
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
# from prompt.prompt_library import PROMPT_REGISTRY # type: ignore

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

document_analysis_prompt = ChatPromptTemplate.from_template("""
You are a highly capable assistant trained to analyze and summarize documents.
Return ONLY valid JSON matching the exact schema below.

{format_instructions}

Analyze this document:
{document_text}
""")

from pydantic import BaseModel, RootModel
from typing import List, Union
from enum import Enum

class Metadata(BaseModel):
    Summary: List[str]
    Title: str
    Author: List[str]
    DateCreated: str
    LastModifiedDate: str
    Publisher: str
    Language: str
    PageCount: Union[int, str]  # Can be "Not Available"
    SentimentTone: str

class DocumentAnalyzer:
    """
    Analyzes documents using a pre-trained model.
    Automatically logs all actions and supports session-based organization.
    """
    def __init__(self):
        # self.log = CustomLogger().get_logger(__name__)
        try:
            # self.loader=ModelLoader()
            self.llm= Phi3ONNX_LLM()
            
            # Prepare parsers
            self.parser = JsonOutputParser(pydantic_object=Metadata)
            self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)
            
            self.prompt = document_analysis_prompt
            
            # self.log.info("DocumentAnalyzer initialized successfully")
            
            
        except Exception as e:
            print("Error initializing DocumentAnalyzer")
            # raise DocumentPortalException("Error in DocumentAnalyzer initialization", sys)
        
        
    
    def analyze_document(self, document_text:str)-> dict:
        """
        Analyze a document's text and extract structured metadata & summary.
        """
        try:
            chain = self.prompt | self.llm | self.fixing_parser
            
            print("Meta-data analysis chain initialized")

            response = chain.invoke({
                "format_instructions": self.parser.get_format_instructions(),
                "document_text": document_text[0:]
            })

            print("Metadata extraction successful", keys=list(response.keys()))
            
            return response

        except Exception as e:
            print("Metadata analysis failed",e)
            # raise DocumentPortalException("Metadata extraction failed",sys)
        
if __name__ == "__main__":
    pdf_path = r"F:\Langchain-GenAi-Projects\project1-class-documentPortal\data\document_analyzer\sample.pdf"
    text_chunks = []
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1): # type: ignore
            text_chunks.append(f"\n--- Page {page_num} ---\n{page.get_text()}")
    text = "\n".join(text_chunks)
    short_text = text[:2000]
    print("Starting metadata analysis...")
    analyzer = DocumentAnalyzer()
    analysis_result = analyzer.analyze_document(short_text)