from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI
import re
import os
from typing import List, Optional, Dict, Any, Tuple, Set
from langchain_community.chat_models import ChatOllama
from Parsers import *
from utils import *


class LLM_agent:
    def __init__(self,api_key = None, llm_type = 'openai',model="gpt-4o-mini",temperature=0.3):
        self.api_key = api_key
        self.llm_type = llm_type
        self.parser = None
        self.num_of_llm_output = None
        if llm_type == 'openai':
            os.environ["OPENAI_API_KEY"] = self.api_key
            self.llm = ChatOpenAI(model_name=model, temperature=temperature)
        elif llm_type == 'azure':
            os.environ["AZURE_OPENAI_API_KEY"] = self.api_key
            os.environ["AZURE_OPENAI_ENDPOINT"] = "https://rtp2-shared.openai.azure.com/"
            os.environ["AZURE_OPENAI_API_VERSION"] = "2024-10-21"
            os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = model
            self.llm = AzureChatOpenAI(
                openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
            )
        elif llm_type == 'anthropic':
            os.environ["ANTHROPIC_API_KEY"] = self.api_key
            self.llm = ChatAnthropic(model=model, temperature=temperature)
        elif llm_type == 'gemini':
            os.environ["GOOGLE_API_KEY"] = self.api_key
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/home/guangya/.config/gcloud/application_default_credentials.json'
            self.llm = ChatGoogleGenerativeAI(model=model, temperature=temperature, gemini_api_key=self.api_key)
        elif self.llm_type == 'lambda':
            self.llm = ChatOpenAI(
                api_key=self.api_key,
                base_url="https://api.lambdalabs.com/v1",  # Add this to your configuration
                model= model
            )
        elif llm_type == 'ollama':
            self.llm = ChatOllama(model=model, temperature=temperature)
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")

    def invoke(self, arg_dict: Dict[str, Any], max_attempts: int = 3) -> Dict[str, Any]:
        """
        Invoke the LLM with the provided arguments.
        Returns the response and number of attempts made.
        """
        if not self.chat_prompt:
            raise ValueError("Prompt template not set. Call setup_prompt first.")

        missing_vars = set(self.chat_prompt.input_variables) - set(arg_dict.keys())
        if missing_vars:
            raise ValueError(f"Missing required input variables: {missing_vars}")

        chain = self.chat_prompt | self.llm
        output = chain.invoke(arg_dict)
        output_text = extract_json(output.content)
        print(0)
        print(output_text)
        formatted_response = self.parser.invoke(output_text)
        return formatted_response

    def setup_prompt(self, prompt_json_path: str, parser_obj: BaseModel) -> None:
        """Set up the prompt template and parser"""
        # Load and process the prompt template
        with open(prompt_json_path) as f:
            prompt_data = json.load(f)
            messages = []
            for key, val in prompt_data.items():
                if key == 'system' and self.llm_type != 'ollama':
                    val += '\n{format_instructions}'
                messages.append((key, val))

        # Set up the parser and prompt
        self.parser = JsonOutputParser(pydantic_object=parser_obj)
        self.num_of_llm_output = len(parser_obj.__fields__)
        self.chat_prompt = ChatPromptTemplate(messages,partial_variables = {"format_instructions": self.parser.get_format_instructions()})

    def get_prompt(self):
        return self.chat_prompt
    def get_parser(self):
        return self.parser
    
if __name__ == '__main__':
    print(1)