# src/utils/exception.py

import sys

class CustomException(Exception):
    """Custom Exception class for handling exceptions in the project.
    
    try:
        x = 10/0
    except Exception as e:
        raise CustomException(e, sys)
    
    """



    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message

    def __str__(self):
        return self.error_message