class UnrecognizedDataFormat(Exception):
     def __init__(self, value, accepted_formats):
         self.value = value
     def __str__(self):
         return repr(self.value)