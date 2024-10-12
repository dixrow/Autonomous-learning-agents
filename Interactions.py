from Teacher import TeacherModel
from Student import StudentModel
import re
import json
import pandas as pd

class Interaction:
    def __init__(self, teacher_model, student_model,system_prompt_teacher = False, system_prompt_student = False ):
        self.teacher = teacher_model
        self.student = student_model
        self.teacher_prompt = ""
        self.teacher_output = ""
        self.student_prompt = ""
        self.student_output = ""
        self.last_answer=""
        self.system_prompt_teacher = system_prompt_teacher
        self.system_prompt_student = system_prompt_student
        

    def make_first_interaction(self, question_1, options_1, answer_1, question_2, options_2,answer_2):
        prompt_teacher = self.teacher.prepare_first_example(question_1, options_1, answer_1, self.system_prompt_teacher)
        teacher_output= self.teacher.generate_response(prompt_teacher, question_2, options_2)
        prompt_student = self.student.prepare_first_prompt(teacher_output, self.system_prompt_student)
        student_output = self.student.generate_response(prompt_student)

        self.teacher_output = teacher_output
        self.student_output = student_output
        
        print("Teacher:" + self.teacher_output + "\n\n")
        print("Student Answer:" + self.student_output + "\n\n")
        
        is_correct = self.evaluate_answer(answer_2)
        return is_correct

    def extract_student_answer(self):
        json_match = re.search(r'{[^{}]*}', self.student_output, re.DOTALL)
        try:
            if json_match:
                json_match = json_match.group()
                response_dict = json.loads(json_match)
                reasoning = response_dict.get("Reasoning", "")
                answer = response_dict.get("Answer_choice", "")
                self.last_answer = answer
            else:
                # Handle case where no JSON-like structure is found
                print("No JSON structure found in the input.")
                self.last_answer = ""
        except json.JSONDecodeError:
            print("Failed to decode JSON. Please check the format.")
            self.last_answer = ""
    
    def evaluate_answer(self,answer_2):
        self.extract_student_answer()
        is_correct = self.last_answer == answer_2
        print(f'Student Answer: {self.last_answer}')
        print(f'Correct Answer: {answer_2}')
        print(f'Is Correct: {is_correct}')
        return is_correct


    def make_second_interaction(self,question_2, choices_2, answer_2):
        prompt_teacher = self.teacher.prepare_second_interation(self.teacher_output, self.student_output, answer_2, self.system_prompt_teacher)
        teacher_output= self.teacher.generate_response(prompt_teacher,question_2, choices_2, 2)
        prompt_student = self.student.prepare_second_prompt(teacher_output, self.student_output, self.system_prompt_student)
        student_output = self.student.generate_response(prompt_student)

        self.teacher_output = teacher_output
        self.student_output = student_output

        """
        print("Teacher:" + self.teacher_output + "\n\n")
        print("Student Answer:" + self.student_output + "\n\n")
        """
        is_correct = self.evaluate_answer(answer_2)
        return is_correct
        
###########################################################################

class Interaction_single_turn:
    def __init__(self, student_model,system_prompt = False):
        self.student = student_model
        self.teacher_output = ""
        self.student_prompt = ""
        self.student_output = ""
        self.last_answer=""
        self.system_prompt = system_prompt

    def make_interaction(self, question_1, options_1, answer_1, reasoning_1, question_2, options_2,answer_2):
        teacher_output = f"""
        Question: {question_1}
    Choices: {options_1}
    Answer: {answer_1}
    Reasoning: {reasoning_1}

    Now, the question that you have to answer is: {question_2}. 
 The choices are:
     {options_2}
    """
        prompt_student = self.student.prepare_first_prompt(teacher_output, self.system_prompt )
        student_output = self.student.generate_response(prompt_student)

        self.teacher_output = teacher_output
        self.student_output = student_output
        
        print("Teacher:" + self.teacher_output + "\n\n")
        print("Student Answer:" + self.student_output + "\n\n")
        
        is_correct = self.evaluate_answer(answer_2)
        return is_correct

    def make_interaction_without_reasoning(self, question_2, options_2,answer_2):
        teacher_output = f"""
    The question that you have to answer is: {question_2}. 
 The choices are:
     {options_2}
    """
        prompt_student = self.student.prepare_first_prompt(teacher_output, self.system_prompt)
        student_output = self.student.generate_response(prompt_student)

        self.teacher_output = teacher_output
        self.student_output = student_output
        
        print("Teacher:" + self.teacher_output + "\n\n")
        print("Student Answer:" + self.student_output + "\n\n")
        
        is_correct = self.evaluate_answer(answer_2)
        return is_correct

    def extract_student_answer(self):
        json_match = re.search(r'{[^{}]*}', self.student_output, re.DOTALL)
        json_match = json_match.group()
        response_dict = json.loads(json_match)
        reasoning = response_dict.get("Reasoning", "")
        answer = response_dict.get("Answer_choice", "")
        self.last_answer = answer

    def evaluate_answer(self,answer_2):
        self.extract_student_answer()
        is_correct = self.last_answer == answer_2
        print(f'Student Answer: {self.last_answer}')
        print(f'Correct Answer: {answer_2}')
        print(f'Is Correct: {is_correct}')
        return is_correct
        
class Experiment_multiturn:
    def __init__(self, teacher_model, student_model, df_qa, system_prompt_teacher=False, system_prompt_student=False):
        self.teacher = teacher_model
        self.student = student_model
        self.df_qa = df_qa
        self
        columns = {
        'Question': 'int64',
        'Correct_Ans': 'string',
        'Answer_1': 'string',
        'Answer_2': 'string'
        }
        
        self.df_results = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in columns.items()})
        self.system_prompt_teacher = system_prompt_teacher
        self.system_prompt_student = system_prompt_student

    def multi_turn(self, i):
        question_1 = self.df_qa['Question'][i-1]
        options_1 = self.df_qa['Options'][i-1]
        answer_1 = self.df_qa['Answer'][i-1]
        question_2 = self.df_qa['Question'][i]
        options_2 = self.df_qa['Options'][i]
        answer_2 = self.df_qa['Answer'][i]

        self.df_results.loc[i-1,'Question'] = i
        self.df_results.loc[i-1,'Correct_Ans']= answer_2
        interaction = Interaction(self.teacher,self.student,self.system_prompt_teacher,self.system_prompt_student)
        is_correct = interaction.make_first_interaction( question_1, options_1, answer_1, question_2, options_2, answer_2)
        self.df_results.loc[i-1,'Answer_1']= interaction.last_answer
        if not is_correct:
            interaction.make_second_interaction(  question_2, options_2, answer_2)
            self.df_results.loc[i-1,'Answer_2'] = interaction.last_answer
            
    def complete_experiment_multiturn(self):
        for i in range(1, self.df_qa.shape[0]):
            self.multi_turn(i)


class Experiment_single_turn:
    def __init__(self,  student_model, df_qa,system_prompt = False):
        self.student = student_model
        self.df_qa = df_qa
        columns = {
        'Question': 'int64',
        'Correct_Ans': 'string',
        'Answer_1': 'string'
        }
        
        self.df_results = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in columns.items()})
        self.system_prompt = system_prompt

    def one_turn(self, i):
        question_1 = self.df_qa['Question'][i-1]
        options_1 = self.df_qa['Options'][i-1]
        answer_1 = self.df_qa['Answer'][i-1]
        reasoning_1 = self.df_qa['Reasoning'][i-1]
        question_2 = self.df_qa['Question'][i]
        options_2 = self.df_qa['Options'][i]
        answer_2 = self.df_qa['Answer'][i]

        self.df_results.loc[i-1,'Question'] = i
        self.df_results.loc[i-1,'Correct_Ans']= answer_2
        interaction = Interaction_single_turn(self.student, self.system_prompt)
        is_correct = interaction.make_interaction( question_1, options_1, answer_1,reasoning_1, question_2, options_2, answer_2)
        self.df_results.loc[i-1,'Answer_1']= interaction.last_answer

    def one_turn_without_reasoning(self, i):
        question_1 = self.df_qa['Question'][i-1]
        options_1 = self.df_qa['Options'][i-1]
        answer_1 = self.df_qa['Answer'][i-1]
        question_2 = self.df_qa['Question'][i]
        options_2 = self.df_qa['Options'][i]
        answer_2 = self.df_qa['Answer'][i]

        self.df_results.loc[i-1,'Question'] = i
        self.df_results.loc[i-1,'Correct_Ans']= answer_2
        interaction = Interaction_single_turn(self.student,self.system_prompt)
        is_correct = interaction.make_interaction_without_reasoning( question_2, options_2, answer_2)
        self.df_results.loc[i-1,'Answer_1']= interaction.last_answer
            
    def complete_experiment_single_turn(self):
        for i in range(1, self.df_qa.shape[0]):
            self.one_turn(i)

    def complete_experiment_without_reasoning(self):
        for i in range(1, self.df_qa.shape[0]):
            self.one_turn_without_reasoning(i)

    
    
        
    
    