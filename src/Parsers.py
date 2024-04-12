
from langchain_core.pydantic_v1 import BaseModel, Field

# Define your desired data structure.
class CoT(BaseModel):
    CoT: str = Field(...,description='''
        Return your complete thought process by following the format: Step 1: xxx
         Step 2: xxx
         ...
        ''')
    Final_answer: str = Field(...,description='''Output your final answer based on the question. 
    If options are provided in the original question, return the option index only. For instance, a) something should just return 'a' in lower 
    case. Otherwise just return the final answer''')
