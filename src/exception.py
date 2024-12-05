import logging
import sys
from types import TracebackType
from src.logger import logging


def error_message_detail(error, exc_tb: TracebackType):
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message=f"Error occured in python script name[{file_name}] line number [{exc_tb.tb_lineno}] error message [{str(error)}]"

    return error_message

class CustomException(Exception):
    def __init__(self, error_messge, error_detail:TracebackType) -> None:
        super().__init__(error_messge)
        self.error_message = error_message_detail(error_messge,error_detail)
    
    def __str__(self):
        return self.error_message

    
if __name__ == '__main__':
        try:
             a = 1/0
        except Exception as e:
            logging.info('division by zero')
            raise CustomException(e, sys.exc_info()[2])