class UnrecognizedDataFormat(Exception):
     def __init__(self, value, accepted_formats):
         self.value = value
         self.accepted_formats = accepted_formats
     def __str__(self):
         return 'You indicated that the data file came from a',value,'format. The only accepted formats are',accepted_formats.values()