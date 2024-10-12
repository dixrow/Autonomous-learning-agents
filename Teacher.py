import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class TeacherModel:
    def __init__(self, access_token, model_name, attn_implementation):
        self.access_token = access_token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attn_implementation = attn_implementation
        self.model, self.tokenizer = self.load_model_and_tokenizer(model_name)
        self.prompt = ""
        

    def load_model_and_tokenizer(self, model_name):
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", torch_dtype=torch.bfloat16, token=self.access_token, attn_implementation = self.attn_implementation, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="cuda", token=self.access_token, trust_remote_code=True)
        return model.to(self.device), tokenizer

    def prepare_first_example(self, question_1, options_1, answer_1, system_prompt = True):
        system_prompt_teacher = """
        You are a teacher willing to explain to your students step by step the reasoning to answer some questions. 
        You will will show a student a question with an explanation step by step of the reasoning to get the correct answer. 
        Your response has to be in the following format, remember that the choices, answer and the question will be provided to you but you have to create the reasoning. 
            Here is an example question and answer along with the reasoning:

            Question: {question}
            Choices: {choices}
            Answer: {answer}
            Reasoning: {reasoning}
        """
        
        chat = []
        if system_prompt:
           chat=  [
            {"role": "system", "content": system_prompt_teacher},
            {"role": "user", "content": f"Explain the reasoning to the student using the following example with this question{question_1}, options {options_1} and the answer {answer_1}."}
        ]
        else:
            chat = [
                {"role": "user", "content": system_prompt_teacher + "\n\n" + f"Explain the reasoning to the student using the following example with this question{question_1}, options {options_1} and the answer {answer_1}."}
            ]
        return chat


    def generate_response(self, chat, question_2, choices_2, interaction = 1):
        tokenized_chat = self.tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.device)
        tokenized_chat_decoded = self.tokenizer.decode(tokenized_chat[0])
        self.prompt = tokenized_chat_decoded
        outputs = self.model.generate(tokenized_chat, max_new_tokens=512)
        decoded_output = self.tokenizer.decode(outputs[0])
        assistant_response = decoded_output[len(tokenized_chat_decoded):]

        question_text_1 = f"\n Now, the question that you have to answer is: {question_2}. \n The choices are: {choices_2}"
        question_text_2 = f"\n With that said, now think it again and answer the question: {question_2}. \n The choices are: {choices_2}"
        question=""
        if interaction == 1:
            question = assistant_response + question_text_1
        else:
            question = assistant_response + question_text_2
        return question

    def prepare_second_interation(self, question_text, answer_text, correct_answer, system_prompt = True):
        system_prompt_teacher = """
    You are a teacher who wants to give your students a step-by-step explanation of the reasoning behind the answers to some questions. 
        You want to explain to a student how to follow the reasoning to answer a question he answered incorrectly. This user will give you the conversation between you as a professor and the student, and you will have to give him another way to reason the question. You will never give him a clue or directly the correct answer, you will only explain to him why his reasoning was bad in general, without explicitly telling him anything about a particular choice.
            Here is how you need to talk to the student:
            First, repeat the reason he gave you, but summarize it.
            Then explain a reason why that reason was not correct.
            

        """
        chat = []

        if system_prompt:
           chat = [
            {"role": "system", "content": system_prompt_teacher},
            {"role": "user", "content": f"""Explain to the student why he got the wrong answer. The context of the conversation was this: " {question_text} ". \n You will only give the reason for the question, not the example given to him. The student's answer was {answer_text}. The correct answer is {correct_answer}."""}
        ]
        else:
            chat = [
                {"role": "user", "content": system_prompt_teacher + "\n\n" + f"""Explain to the student why he got the wrong answer. The context of the conversation was this: " {question_text} ". \n You will only give the reason for the question, not the example given to him. The student's answer was {answer_text}. The correct answer is {correct_answer}."""}
            ]
        return chat
    