import traceback
import sys
from logger.custom_logger import CustomLogger
logger=CustomLogger().get_logger(__file__)

class DocumentPortalException(Exception):
    """Custom exception for Document Portal"""
    def __init__(self,error_message):
        _,_,exc_tb= sys.exc_info()
        self.file_name=exc_tb.tb_frame.f_code.co_filename
        self.lineno=exc_tb.tb_lineno
        self.error_message=str(error_message)
        self.traceback_str = ''.join(traceback.format_exception(type(error_message), error_message, exc_tb)) 
        
    def __str__(self):
       return f"""
        Error in [{self.file_name}] at line [{self.lineno}]
        Message: {self.error_message}
        Traceback:
        {self.traceback_str}
        """
    
if __name__ == "__main__":
    try:
        # Simulate an error
        a = 1 / 0
        print(a)
    except Exception as e:
        app_exc=DocumentPortalException(e)
        # print(app_exc)
        logger.error(app_exc)
        raise app_exc