# objective is to create a class that can be used to get two pdf files from stramlit like interface
# and compare the changes between those two pdfs.
# need to create one pydantic class that will check the return type
from datetime import datetime
from pathlib import Path
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
import sys
class DocumentIngestion:
    def __init__(self):
        # will read config file,data_dir,
        self.log = CustomLogger.get_logger(__name__)
        self.base_data_dir = Path("data/document_compare")
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_path = self.base_data_dir / self.session_id
        self.session_path.makedir(parents =True,exist_ok = True )
        self.log.info("session path created successfully")
        

    def save_uploaded_file(self,refrence_file,actual_file):
        try:
            ref_file_path = self.session_path / refrence_file.name
            act_file_path = self.session_path / actual_file.name
            if not refrence_file.lower().endswith(".pdf") or not (actual_file.lower().endswith(".pdf")):
                raise ValueError("Only PDF files are allowed for now")
            with open(ref_file_path,'rb') as r_file:
                r_file.write(refrence_file.get_buffer())

            with open(act_file_path,'rb') as a_file:
                a_file.write(actual_file.get_buffer())    
            self.log.info("refrence file and actual file saved successfully") 


        except Exception as e:
            self.log.error("error occured while reading the file from thirs party class or UI tool")
            raise DocumentPortalException("Error saving files", sys)
        
    def read_file(self):
           



            




