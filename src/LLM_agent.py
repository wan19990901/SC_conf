from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from typing import List, Optional
import re
from langchain_community.llms import Ollama
from Parsers import *

class LLM_agent:
    def __init__(self,api_key = None, llm_type = 'openai',model="gpt-3.5-turbo-0125",temperature=0):
        self.api_key = api_key
        if llm_type == 'openai':
            self.llm = ChatOpenAI(openai_api_key=self.api_key, model_name=model, temperature=temperature)
        elif llm_type == 'anthropic':
            self.llm = ChatAnthropic(model=model,
                                  anthropic_api_key=self.api_key)
        elif llm_type == 'ollama':
            self.llm = Ollama(model=model)
        self.llm_type = llm_type
        self.chat_prompt = None

    def involk(self,var_dict):
        if self.llm_type == 'openai':
            output_parser = self.parser
        elif self.llm_type == 'anthropic':
            output_parser = extract_json
        elif self.llm_type == 'ollama':
            output_parser = extract_json
        chain = self.chat_prompt | self.llm | output_parser
        success = False
        attempts = 1
        response = None
        while (not success) and attempts < 5:
            print(attempts)
            attempts+= 1
            try:
                response = chain.invoke(var_dict)
                if len(response) == self.num_of_llm_output:
                    success = True
            except:
                success = False
        return response
    def get_llm(self):
        return self.llm
    def set_prompt(self,prompt_json_link):
        with open(prompt_json_link) as f:
            message = []
            for key,val in json.load(f).items():
                if key == 'system':
                    val += '\n{format_instructions}'
                message.append((key,val))

        chat_prompt = ChatPromptTemplate.from_messages(message)
        self.chat_prompt = chat_prompt

    def get_prompt(self):
        return self.chat_prompt

    def set_parser(self, parser_obj):
        self.parser = JsonOutputParser(pydantic_object=parser_obj)
        self.num_of_llm_output = len(parser_obj.__fields__)
        self.chat_prompt = self.chat_prompt.partial(format_instructions=self.parser.get_format_instructions())
    def get_parser(self):
        return self.parser

def extract_json(message: AIMessage) -> List[dict]:
    """Extracts JSON content from a string where JSON is embedded between ```json and ``` tags.

    Parameters:
        text (str): The text containing the JSON content.

    Returns:
        list: A list of extracted JSON strings.
    """
    text = message.content
    # Define the regular expression pattern to match JSON blocks
    pattern = r'"([^"]+)":\s*"([^"]+)"'

    # Find all non-overlapping matches of the pattern in the string
    matches = re.findall(pattern, text)

    # Return the list of matched JSON strings, stripping any leading or trailing whitespace
    try:
        json_string = json.dumps(dict(matches), indent=4)
        return dict(matches)
    except Exception:
        raise ValueError(f"Failed to parse: {message}")

if __name__ == '__main__':
    print(1)