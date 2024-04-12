from LLM_agent import *

if __name__ == '__main__':
    llm_config = {
        # change these three together
        'llm_type': 'ollama',  # openai, ollama, anthropic
        'api_key_link': 'api_key_claude_yw.txt',
        'model': "llama2:13b",  # see llm_model.txt
        # change these two together
        'prompt_link': 'prompt_template.json',
        'parser_template': CoT,
        # change as needed
        'temperature': 0,
    }
    arguments_dict = {
        'subject': 'math',
        'question': '''
        if - 3 x + 5 y = 48 and 3 x - 2 y = 6 , what is the product of x and y ? The options are: 
        a ) 252 , b ) 428 , c ) 464 . , d ) 200 , e ) 642
        '''
    }

    with open(llm_config['api_key_link'], 'r') as f:
        api_key = f.read()
    cot_agent = LLM_agent(llm_type=llm_config['llm_type'], api_key=api_key, model=llm_config['model'],
                          temperature=llm_config['temperature'])
    cot_agent.set_prompt(llm_config['prompt_link'])
    cot_agent.set_parser(llm_config['parser_template'])
    print(cot_agent.involk(arguments_dict))
