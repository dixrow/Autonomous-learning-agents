import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class StudentModel:
    def __init__(self, access_token, model_name):
        self.access_token = access_token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tokenizer = self.load_model_and_tokenizer(model_name)
        self.prompt = ""

    def load_model_and_tokenizer(self, model_name):
        trust_remote_code = False
        if model_name == "microsoft/Phi-3-vision-128k-instruct":
            trust_remote_code = True
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", torch_dtype=torch.bfloat16, token=self.access_token,trust_remote_code=trust_remote_code, attn_implementation="flash_attention_2")
        tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="cuda", token=self.access_token,trust_remote_code=trust_remote_code)
        return model.to(self.device), tokenizer

    def prepare_first_prompt(self, question, system_prompt = False):
        system_prompt_student = """
            You are a student ready to answer some questions that a teacher would ask you. You must think step by step to find the answer. There is only one correct answer, so please answer with only one choice. Please answer in the following JSON format and nothing else. Please do not give 2 or JSON objects, only one:
            {
    "Reasoning": "{Your Reasoning}",
    "Answer_choice": "{Choice letter}",
    "Answer":"{answer_text}"
            }
            
            """
        prompt = []
        if system_prompt:
            prompt = [
                {"role": "system", "content": system_prompt_student},
                {"role": "user", "content": question}
            ]
        else:
            prompt = [
                {"role": "user", "content": system_prompt_student + "\n\n" + "The teacher example and question are: " + "\n\n" + question}
            ]
        return prompt

    def prepare_second_prompt(self, question, previous_answer, system_prompt = False):
        system_prompt_student = """
           You are a student ready to answer some questions that a teacher would ask you. You must think step by step to find the answer. There is only one correct answer, so please answer with only one choice. Please answer in the following JSON format and nothing else. Please do not give 2 or JSON objects, only one:
            {
    "Reasoning": "{Your Reasoning}",
    "Answer_choice": "{A,B,C,D or E}",
    "Answer":"{Choice_text}"
            }
            """
        prompt = []
        if system_prompt:
            chat = [
                {"role": "system", "content": system_prompt_student},
                {"role": "user", "content": question}
            ]
        else:
            chat = [
                {"role": "assistant","content":previous_answer},
                {"role": "user", "content": system_prompt_student + "\n\n" + "A teacher is triying to teach you how to reasonate, explaining why you have a mistake, and said the following : " + """ " """  + "\n\n" + question + """ " """}
            ]
        return chat
    def generate_response(self, prompt):
        tokenized_chat = self.tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.device)
        tokenized_chat_decoded = self.tokenizer.decode(tokenized_chat[0])
        self.prompt = tokenized_chat_decoded
        outputs = self.model.generate(tokenized_chat, max_new_tokens=512)
        decoded_output = self.tokenizer.decode(outputs[0])
        assistant_response = decoded_output[len(tokenized_chat_decoded):]
        return assistant_response

