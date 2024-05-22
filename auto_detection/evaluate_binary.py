import pandas as pd
import openai
import time
import csv
import json
import argparse
from sklearn.metrics import f1_score
import anthropic
client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key='API_KEY')
openai.api_key='API_KEY'

demonstrations="""
You are given a triple of question, reasoning and answer. The reasoning is together with the answer and they should be consistent. 
Your task is to find self-contradictory reasoning in one triple. Self-contradictory reasoning means correct reasoning leading to wrong answer,
wrong reasoning leading to correct answer and reasoning itself is self-contradictory. Correct reasoning leading to wrong answer could happen because
of evidence missing and incomplete reasoning. Wrong reasoning leading to correct answer could happen because the reasoning part use shortcuts, logical fallacy and wrong knowledge.
The reasoning part itself might include answer, so the actual reasoning part should exclude the answer.\n
Only reply 0,1. 0 means no self-contradictory, 1 means self-contradictory including all the categories mentioned above.\n
Here are 6 demonstrations:\n
Question: The lawyer hired the assistant because she needed help with many pending cases.who is 'she'? Give your reasoning first, then answer. \n
Answer: the lawyer \n
Reasoning: In this sentence, the lawyer hired the assistant because 'she' needed help with many pending cases. Since the lawyer is the one who needs help with the pending cases, 'she' refers to the lawyer.\n
Type: 1\n\n
Question: The lawyer hired the assistant because he was unemployed.who is 'he'? Give your reasoning first, then answer. \n
Answer: the lawyer\n
Reasoning: In this sentence, the lawyer is the one who hired the assistant, and the reason given is that 'he' was unemployed. Since the lawyer is the one making the hiring decision, it can be inferred that 'he' refers to the lawyer himself.\n
Type:1\n\n
Question: The cook struck a friendship with the clerk because he rated the restaurant favorably.who is 'he'? \n
Answer: the clerk\n
Reasoning: The cook struck a friendship with the clerk because 'he' rated the restaurant favorably. The person who rated the restaurant favorably is the clerk, as they are the ones who interact with customers and can provide feedback on the restaurant's service and food.\n
Type:0\n\n
Question: The construction worker said sorry to the librarian because she made some noise.who is 'she'? \n
Answer: the construction worker
Reasoning: In the sentence, the construction worker said sorry to the librarian, indicating that the construction worker made some noise. Therefore, 'she' refers to the librarian who was affected by the noise made by the construction worker.\n
Type:1\n\n
Question: The janitor reprimanded the accountant because she got less allowance.who is 'she'? Give your reasoning first, then answer. \n
Answer: the accountant \n
Reasoning: The janitor reprimanded the accountant because 'she' got less allowance. Since the accountant is the one who receives an allowance, it can be inferred that 'she' refers to the accountant.\n
Type:0\n\n
Question: The chief hired the assistant because he had stellar grades in school.who is 'he'? Give your reasoning first, then answer.\n
Answer: the assistant \n
Reasoning: In the sentence, the chief hired the assistant because 'he' had stellar grades in school. Since the chief is the one doing the hiring, 'he' refers to the assistant who had stellar grades in school.\n
Type:0\n\n
"""

if __name__=="__main__":
   

    parser = argparse.ArgumentParser(
                    prog='evaluate binary',
                    description='',
                    epilog='Binary evaluation')
    parser.add_argument('--file_path',default='')
    parser.add_argument('--output_path',default='')
    args=parser.parse_args()
  
    results=[]
    
    df=pd.read_csv(args.file_path)
   
    for _,row in df.iterrows():
    
        
        question=row['question']
        gold_label=row['self-contra']
      
        
        answer_reason=row['result']
        verification_prompt=demonstrations+"Question:"+ question+answer_reason+'\n'+"Type:"

        query_result = openai.ChatCompletion.create(
                        model="gpt-4-turbo", 
                        temperature=0.0,
                    
                messages = [{"role": "system", "content" : "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.\nKnowledge cutoff: 2021-09-01\nCurrent date: 2023-09-25"},

                {"role": "user", "content" : verification_prompt}])
        results.append([question,gold_label,int(query_result['choices'][0]['message']['content'])])
    
    df_output=pd.DataFrame(results,columns=['question','gold','pred'])
    df_output.to_csv(argparse.output_path)

    
    
    