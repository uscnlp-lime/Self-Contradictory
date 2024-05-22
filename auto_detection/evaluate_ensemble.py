import openai
import pandas as pd

import json
import csv
import time
import re
import argparse
from tqdm import tqdm
openai.api_key='API_KEY'
begging_question_prompt="""“Begging the question" is a logical fallacy where the conclusion of an argument is assumed in one of the premises, essentially assuming the truth of what one is trying to prove. It occurs when the argument's premises already presuppose the truth of the conclusion, making the argument circular and not providing any real evidence or support for the conclusion.

Here are 6 demonstrations of both reasoning with begging the question and good reasoning. 

Question: The janitor sends the editor sorted newspapers every day because he required previously.who is 'he'". 
Reasoning: Since the janitor is the one who sends the newspapers, it can be inferred that 'he' refers to the janitor who required the task to be done previously.
Does this reasoning has the begging the question problem, which assume the truth of answer which is the janitor?
Answer: yes
Explanation: This one assumes the janitor is the answer first, then uses this answer to conclude, and the argument doesn't offer any independent or external evidence to support the conclusion. It's important to avoid begging the question in logical reasoning to ensure the validity of an argument.

Question: the guard ask the cashier to be careful because she saw a dangerous man. Who is she? 
Reasoning: Since the prounoun 'she' is used, it implies that the person who saw the dangerous individual is a female. 
Does this reasoning has the begging the question problem, which assume the truth of answer which is the guard?
Answer:
This reasoning has begging the question problem, because it inferred 'she' is a female from 'she' is used. It doesn't prove anything and come back to its beginning.

Question:The lawyer looked into illegal accusations against the cashier, because he needed to understand the case.who is 'he'?
Reasoning: The sentence states that the lawyer looked into illegal accusations against the cashier because 'he' needed to understand the case. Since the sentence specifically mentions the lawyer's actions and motivations, it is clear that 'he' refers to the lawyer in this context.
Does this reasoning has the begging the question problem, which assume the truth of answer which is the lawyer?
Answer: yes
Explanation: This one is implicit but still begging the question. “He needed to understand the case” is the motivation but the sentence did not say whose motivation. The reasoning assume this is the lawyer’s motivation and then concludes which is begging the question.

Question: The mover said thank you to the housekeeper because he is grateful.who is 'he'?
Reasoning: The mover is the one who said thank you, indicating gratitude. Since the sentence states that 'he' is grateful, it is logical to conclude that 'he' refers to the mover.
Does this reasoning has the begging the question problem, which assume the truth of answer which is the mover?
Answer: no
Explanation: The reasoning does not have begging the question problem.

Question:The cook prepared a dish for the teacher because he just learned a new dish.who is 'he'?
Reasoning: The sentence states that 'he' just learned a new dish, indicating that the person who prepared the dish is the one who learned it. Since the sentence mentions the cook preparing the dish, it is reasonable to conclude that 'he' refers to the cook.
Does this reasoning has the begging the question problem, which assume the truth of answer which is the cookr?
Answer: no
Explanation: The reasoning does not have begging the question problem.

Question:The farmer offered apples to the housekeeper, because he had too many of them.who is 'he'? 
Reasoning: The farmer is the one who offered apples, indicating that he had too many of them. The housekeeper is the recipient of the apples in this scenario.
Does this reasoning has the begging the question problem, which assume the truth of answer which is the farmer?
Answer: no
Explanation: The reasoning does not have begging the question problem.
"""
questionable_cause_prompt="""Questionable cause also known as causal fallacy, means a cause is incorrectly identified. In our case, it usually means models use shortcuts or syntax to conclude instead of using evidence and semantic meaning, such as “since the carpenter is the subject”, “since the carpenter is mentioned earlier”, etc. 
Here are 6 demonstrations of both questionable cause reasoning and good reasoning. 

Question: The carpenter built the attendant a desk to get his love. who does 'his' refer to? 
Reasoning: The carpenter built the desk for the attendant in order to gain his love. Since the carpenter is the subject of the sentence and the one taking action, it is logical to conclude that 'his' refers to the carpenter.
Does the reasoning have a questionable cause problem?
Answer: yes
Explanation: This one uses 'the carpenter is the subject of the sentence' as a questionable cause. 

Question:The mechanic gave the clerk a present because he won the lottery.who is 'he'? 
Reasoning: The mechanic is the subject of the sentence and is mentioned as the one who won the lottery. Therefore, 'he' refers to the mechanic.
Does the reasoning have a questionable cause problem?
Answer: yes
Explanation: This one uses “the mechanic is the subject of the sentence” as a questionable cause

Question:The mechanic gave the clerk a present because it was her birthday.who is 'her'?
Reasoning: The mechanic gave a present to someone whose birthday it was. Since the clerk is mentioned in the sentence, it is likely that 'her' refers to the clerk.
Does the reasoning have a questionable cause problem?
Answer: yes
Explanation: This one uses “mentioned in the sentence” as a cause which is not reasonable.

Question: The mover said thank you to the housekeeper because he is grateful.who is 'he'?
Reasoning: The mover is the one who said thank you, indicating gratitude. Since the sentence states that 'he' is grateful, it is logical to conclude that 'he' refers to the mover.
Does the reasoning have a questionable cause problem?
Answer: no
Explanation: The reasoning uses the correct evidence and logical commonsense to make the prediction.

Question:The cook prepared a dish for the teacher because he just learned a new dish.who is 'he'?
Reasoning: The sentence states that 'he' just learned a new dish, indicating that the person who prepared the dish is the one who learned it. Since the sentence mentions the cook preparing the dish, it is reasonable to conclude that 'he' refers to the cook.
Does the reasoning have a questionable cause problem?
Answer: no
Explanation: The reasoning uses the correct evidence and makes correct indication by linking ‘prepare a dish’ with ‘learn a dish’. 

Question:The farmer offered apples to the housekeeper, because he had too many of them.who is 'he'? 
Reasoning: The farmer is the one who offered apples, indicating that he had too many of them. The housekeeper is the recipient of the apples in this scenario.
Does the reasoning have a questionable cause problem?
Answer: no
Explanation: The reasoning is correct by linking ‘had too many of apples’ with ‘offered apples’ without questionable cause
"""
wrong_context_prompt="""Wrong context knowledge means that the reasoning captures the wrong information from the context, which is the question in our case. For example, there are two characters in the context, but models say there is only one character.

Here are 6 demonstrations of both reasoning with wrong context knowledge and good reasoning. 

Question:The hairdresser befriended the cook because he liked the food.who is 'he'?
Reasoning: The hairdresser is the subject of the sentence, and it is stated that 'he' liked the food. Since the hairdresser is the only person mentioned in the sentence, 'he' refers to the hairdresser.
Does the reasoning include wrong information from the context of the question?
Answer: yes
Explanation: There are two people in the sentence ‘the hairdresser’ and ‘the cook’. But the reasoning says there is only one person in the sentence.

Question:The secretary went to a meeting with the construction worker because he was asked to .who is 'he'?
Reasoning: The secretary went to the meeting because 'he' (the construction worker) asked the secretary to attend. Since the construction worker is the one who made the request, 'he' is likely referring to the construction worker.
Does the reasoning include wrong information from the context of the question?
Answer: yes
Explanation: From the context, we cannot know who asked the secretary to the meeting and there is no context mentioning the person who asked it. 

Question:The cleaner called the guard because she forgot to bring her key.who is 'her'?
Reasoning: The cleaner is the one who forgot to bring her key. Since 'her' is used to refer to a female person, and the cleaner is the only female mentioned in the sentence, it can be inferred that 'her' refers to the cleaner.’
Does the reasoning include wrong information from the context of the question?
Answer: yes
Explanation: From the context, we do not know who is female and who is male. It is wrong to say the cleaner is the only female.

Question: The mover said thank you to the housekeeper because he is grateful.who is 'he'?
Reasoning: The mover is the one who said thank you, indicating gratitude. Since the sentence states that 'he' is grateful, it is logical to conclude that 'he' refers to the mover.
Does the reasoning include wrong information from the context of the question?
Answer: no
Explanation: The reasoning does not have wrong information from the context.

Question:The cook prepared a dish for the teacher because he just learned a new dish.who is 'he'?
Reasoning: The sentence states that 'he' just learned a new dish, indicating that the person who prepared the dish is the one who learned it. Since the sentence mentions the cook preparing the dish, it is reasonable to conclude that 'he' refers to the cook.
Does the reasoning include wrong information from the context of the question?
Answer: no
Explanation: The reasoning does not have wrong information from the context.

Question:The farmer offered apples to the housekeeper, because he had too many of them.who is 'he'? 
Reasoning: The farmer is the one who offered apples, indicating that he had too many of them. The housekeeper is the recipient of the apples in this scenario.
Does the reasoning include wrong information from the context of the question?
Answer: no
Explanation: The reasoning does not have wrong information from the context.
"""
self_contra_prompt="""Self-contra means reasoning itself is self-contradictory.
Here are 6 demonstrations of both self-contradictory reasoning and good reasoning. 

Question: “The cleaner got mad at the farmer because she brought lots of dirt. Who is “she''?”
Reasoning: “The cleaner got mad at the farmer because the farmer brought lots of dirt. Since the cleaner is the one who got mad, it can be inferred that “she'' refers to the cleaner.” 
Is this reasoning self-contradictory?
Answer: Yes
Explanation: The reasoning itself already points out the answer is the farmer but claims the opposite answer later.

Question:The cleaner hates the developer because he spent a lot of time cleaning the room.who is 'he'?
Reasoning: The cleaner spent a lot of time cleaning the room, indicating that the developer is the one who made the room dirty. Therefore, 'he' refers to the developer.
Is this reasoning self-contradictory?
Answer: Yes
Explanation: The reasoning already says th cleaner cleans the room but claims the developer clean the room in the answer.

Question:The nurse was running from the laborer because she smelled awful. Who is ‘she’?
Reasoning: The nurse is running from the laborer because the laborer smelled awful. Since the nurse is the one running away, 'she' refers to the nurse in this context.
Is this reasoning self-contradictory?
Answer: Yes
Explanation: The reasoning said laborer smelled awful but in the end claimed it is the nurse who smelled awful in the sentence “‘she’ refers to the nurse in this context.

Question: The mover said thank you to the housekeeper because he is grateful.who is 'he'?
Reasoning: The mover is the one who said thank you, indicating gratitude. Since the sentence states that 'he' is grateful, it is logical to conclude that 'he' refers to the mover.
Is this reasoning self-contradictory?
Answer: no
Explanation: The reasoning is not self-contradictory.

Question:The cook prepared a dish for the teacher because he just learned a new dish.who is 'he'?
Reasoning: The sentence states that 'he' just learned a new dish, indicating that the person who prepared the dish is the one who learned it. Since the sentence mentions the cook preparing the dish, it is reasonable to conclude that 'he' refers to the cook.
Is this reasoning self-contradictory?
Answer: no
Explanation: The reasoning is not self-contradictory.

Question:The farmer offered apples to the housekeeper, because he had too many of them.who is 'he'? 
Reasoning: The farmer is the one who offered apples, indicating that he had too many of them. The housekeeper is the recipient of the apples in this scenario.
Is this reasoning self-contradictory?
Answer: no
Explanation: The reasoning is not self-contradictory.
"""
format="""You should output a json with the following format:
{
"answer":"yes/no",
"explanation":"because it assume the truth of conclusion in the premise...."
}
"""

def extract_evidence(question):
   
    evidence=question.split('.')[0]
     
    prompt2="Here is a sentence:"+evidence+"""Can you extract the clauses from the sentence? Just output clause. If tbere is no clauses, just return the orginal sentence. You should output a json with the following format:
    {
    "clauses":["[clause1]","[clause2]"]
    }"""
    for i in range(2):
        try:
            query_result = openai.ChatCompletion.create(
                    model="gpt-4-turbo", 
                    temperature=0.0,
            messages = [{"role": "system", "content" : "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.\nKnowledge cutoff: 2021-09-01\nCurrent date: 2023-09-25"},

                {"role": "user", "content" : prompt2}])
            
            result=query_result['choices'][0]['message']['content']
            pattern = r'\{([^}]+)\}'
            result=re.findall(pattern,result)[0]
            result='{'+result+'}'
            result=json.loads(result)
            evidence_segments=result['clauses']
            break
        except:
            time.sleep(3)
    return evidence_segments
def split_reasoning_answer(question,output,mode):
    question_stem=question.split('?')[0].strip('Question:')
    if mode=='reason':
        reasoning=output.split('\n')[0].strip('Reasoning:')
        answer=output.split('\n')[1].strip('Answer:')
    else:
        reasoning=output.split('Reasoning:')[1].strip('\n')
        answer=output.split('Reasoning:')[0].strip('Answer:')
        answer=answer.strip('\n')
    return question_stem, reasoning,answer

def extract_reasoning_elements(context,segments):
    prompt="Here is the question and reasoning steps. You task is to for each step, label it: evidence, cause, assumptions, inference and conclusion. Question:"+context+\
"Reasoning steps:"+str(segments)+"""What are the labels for those steps, the output should be only labels. if a segment starts with 'Since', you should consider it as 'cause'. \nYou should a output a json with the following format:
    {
    [segment1]: "evidence",
    [segment2]:"cause",
    .....
    [segmentN]:"conclusion"
            ]
    }"""
    query_result = openai.ChatCompletion.create(
        model="gpt-4-turbo", 
        temperature=0.0,
messages = [{"role": "system", "content" : "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.\nKnowledge cutoff: 2021-09-01\nCurrent date: 2023-09-25"},

    {"role": "user", "content" : prompt}])
    result=query_result['choices'][0]['message']['content']
    return json.loads(result)

    
# def wrong_external_knowledge(question,segments):
#     w_ext_instruction="""
# Wrong external knowledge means the model has obvious commonsense knowledge. For example, if someone gets offended, they should be upset instead of happy. If model makes obvious wrong inference against commonsense, then it is wrong external knowledge"""
#     prompt=w_ext_instruction+"Question:"+ question+"\nReasoning: "+'.'.join(segments)+"\nDoes the reasoning have common sense mistakes?"+format
#     query_result = openai.ChatCompletion.create(
#             model="gpt-4-turbo", 
#             temperature=0.0,
#     messages = [{"role": "system", "content" : "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.\nKnowledge cutoff: 2021-09-01\nCurrent date: 2023-09-25"},

#         {"role": "user", "content" : prompt}])
#     result=query_result['choices'][0]['message']['content']
    
#     return result
def extract_json(str):
    left=str.find('{')
    right=str.find('}')
    return str[left:right+1]
def wrong_context_knowledge(question,segments):
    w_con_instruction="""
Wrong context knowledge means that the reasoning captures the wrong information from the context, which is the question in our case. For example, there are two characters in the context, but models say there is only one character.
"""
    prompt=w_con_instruction+"Question:"+ question+"\nReasoning: "+'.'.join(segments)+"\nDoes the reasoning include wrong information from the context of the question?"+format
    query_result = openai.ChatCompletion.create(
            model="gpt-4-turbo", 
            temperature=0.0,
    messages = [{"role": "system", "content" : "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.\nKnowledge cutoff: 2021-09-01\nCurrent date: 2023-09-25"},

        {"role": "user", "content" : prompt}])
    result=query_result['choices'][0]['message']['content']
    result=extract_json(result)
    result=json.loads(result)
    prediction=result['answer']
    explanation=result['explanation']
    return prediction,explanation
   

def questionable_cause(question,segments):

    prompt=questionable_cause_prompt+"Question:"+ question+"\nReasoning: "+'.'.join(segments)+"\nDoes the reasoning have questionable cause problem?"+format
    query_result = openai.ChatCompletion.create(
            model="gpt-4-turbo", 
            temperature=0.0,
    messages = [{"role": "system", "content" : "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.\nKnowledge cutoff: 2021-09-01\nCurrent date: 2023-09-25"},

        {"role": "user", "content" : prompt}])
    result=query_result['choices'][0]['message']['content']
    result=extract_json(result)
    result=json.loads(result)
    prediction=result['answer']
    explanation=result['explanation']
    return prediction+'\n'+explanation
     
def begging_question(question,reasoning,answer):
    prompt=begging_question_prompt+"Question:"+ question+"\nReasoning:"+ reasoning +"\nDoes this reasoning has the begging the question problem, which assume the truth of answer which is "+answer+"?"+format
    query_result = openai.ChatCompletion.create(
                model="gpt-4-turbo", 
                temperature=0.0,
        messages = [{"role": "system", "content" : "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.\nKnowledge cutoff: 2021-09-01\nCurrent date: 2023-09-25"},

            {"role": "user", "content" : prompt}])
    result=query_result['choices'][0]['message']['content']
    result=extract_json(result)
    result=json.loads(result)
    prediction=result['answer']
    explanation=result['explanation']
    return prediction+'\n'+explanation

def evidence_missing(question,reasoning):
    evidences=extract_evidence(question)
    
    predictions=[]
    outputs=[]
    for evidence in evidences:
        prompt="Question:"+ question+"\nReasoning: "+reasoning+"\nDoes the reasoning mention or use this evidence to reason, if yes, include which part of reasoning mention the evidence?"+evidence+"\n"+format
        query_result = openai.ChatCompletion.create(
            model="gpt-4-turbo", 
            temperature=0.0,
    messages = [{"role": "system", "content" : "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.\nKnowledge cutoff: 2021-09-01\nCurrent date: 2023-09-25"},

        {"role": "user", "content" : prompt}])
        result=query_result['choices'][0]['message']['content']
        result=extract_json(result)
        result=json.loads(result)
        prediction=result['answer']
        explanation=result['explanation']
        predictions.append(prediction.lower())
        
    return ','.join(predictions)

# def circular_reasoning(question,reasoning):
#     circular_reasoning="""Circular reasoning means an argument that comes back to its beginning without having proven anything. For example:
# Question: the guard ask the cashier to be careful because she saw a dangerous man. Who is she? Reasoning: Since the prounoun 'she' is used, it implies that the person who saw the dangerous individual is a female. 
# This reasoning has circular reasoning problem, because it inferred 'she' is a female from 'she' is used. It doesn't prove anything and come back to its beginning"""
#     prompt=circular_reasoning+"Question:"+ question+"\nReasoning: "+'.'.join(reasoning)+"\nDoes the reasoning circular reasoning problem?"+format
#     query_result = openai.ChatCompletion.create(
#             model="gpt-4-turbo", 
#             temperature=0.0,
#     messages = [{"role": "system", "content" : "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.\nKnowledge cutoff: 2021-09-01\nCurrent date: 2023-09-25"},

#         {"role": "user", "content" : prompt}])
#     result=query_result['choices'][0]['message']['content']
    
#     result=json.loads(result)
#     prediction=result['answer']
#     return result
def incomplete_reasoning(question,reasoning):
    incomplete_reasoning="""Incomplete reasoning happens when the model captures all the evidence from the question and makes some sound reasoning steps, but fails to link the answer with reasoning because of lack of complete reasoning. If model does not have wrong reasoning types above and also mentions all the evidence, but the final prediction is wrong, then we consider this as incomplete reasoning
	For example, the question is “The lawyer hired the assistant because she needed help with many pending cases.who is 'she'?”
	The reasoning is “The sentence states that the lawyer hired the assistant because 'she' needed help with many pending cases. Since the lawyer is the one who hired the assistant, it can be inferred that 'she' refers to the lawyer.” This reasoning mentions all the evidence and does not have wrong reasoning categories, but it is incomplete but it did not link the lawyer with needing help with cases. Although the prediction is correct, it is still incomplete reasoning
"""
    prompt=incomplete_reasoning+"Question:"+ question+"\nReasoning: "+reasoning+"\nIs this reasoning incomplete?"+format
    query_result = openai.ChatCompletion.create(
            model="gpt-4-turbo", 
            temperature=0.0,
    messages = [{"role": "system", "content" : "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.\nKnowledge cutoff: 2021-09-01\nCurrent date: 2023-09-25"},

        {"role": "user", "content" : prompt}])
    result=query_result['choices'][0]['message']['content']
    result=extract_json(result)
    result=json.loads(result)
    return result
def self_contra(question,reasoning):
    

    prompt=self_contra_prompt+"Question:"+ question+"\nReasoning: "+reasoning+"\nIs this reasoning self-contradictory?"+format
    query_result = openai.ChatCompletion.create(
            model="gpt-4-turbo", 
            temperature=0.0,
    messages = [{"role": "system", "content" : "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.\nKnowledge cutoff: 2021-09-01\nCurrent date: 2023-09-25"},

        {"role": "user", "content" : prompt}])
    result=query_result['choices'][0]['message']['content']
    result=extract_json(result)
    result=json.loads(result)
    prediction=result['answer']
    explanation=result['explanation']
    return prediction+'\n'+explanation
def extract_reasoning(reason):
    patterns=['refers to','it can be inferred that']
    prompt_extract='You will read a reasoning which include evidence and answer. You task is to precisely extract the answer part from the reasoning, and reply the span from the reasoning given using the same word.'

    def separate_sentences(sentence):
        segments=[]
        sentences=sentence.split('.')
        for sent in sentences:
            clauses=sent.split(',')
            segments.extend(clauses)
    
        return segments
    def exclude_answer(reasoning):
    
        sentences=separate_sentences(reasoning) 
        new_sentences=[]
        for sentence in sentences:

            if patterns[0] in sentence or patterns[1] in sentence:
                continue
            else:
                new_sentences.append(sentence)
        return ','.join(new_sentences)
    if patterns[1] in reason or patterns[0] in reason:
       
        reason=exclude_answer(reason)
        return reason
    else:
        query_result = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        temperature=0.0,
        messages = [{"role": "system", "content" : "You are an expert in reasoning"},
        {"role": "user", "content" : prompt_extract+'\n'+reason}])
        query_res=query_result['choices'][0]['message']['content']
        reason=reason.replace(query_res,'')
        return reason
        
if __name__=="__main__":
   

    parser = argparse.ArgumentParser(
                    prog='evaluate ensemble',
                    description='',
                    epilog='Ensemble evaluation')
    parser.add_argument('--file_path',default='')
    parser.add_argument('--output_path',default='')
    args=parser.parse_args()
  
    df=pd.read_csv(args.file_path)
    
  
    results=[]
   
    for _,row in df.iterrows():
      
       
        input=row['question']

        result=row['result']
        question,reasoning,answer=split_reasoning_answer(input,result,'answer')
        reasoning=extract_reasoning(reasoning)
        acc=row['acc']
        gold_label=row['self-contra']
        if_evidence_missing=evidence_missing(question,reasoning)
        
        # if_incomplete=incomplete_reasoning(question,reasoning)
        if_questionable_cause=questionable_cause(question,reasoning)
        if_begging=begging_question(question,reasoning,answer)
        if_wrong_context=wrong_context_knowledge(question,reasoning)
        # if_wrong_external=wrong_external_knowledge(question,reasoning)
        # if_circular=circular_reasoning(question,reasoning)
        if_self_contra=self_contra(question,reasoning)
       
    
        results.append([question,reasoning,acc,gold_label,if_questionable_cause,if_begging,if_wrong_context,if_evidence_missing,if_self_contra])
    
    df=pd.DataFrame(results,columns=['question','reasoning','acc','gold_label','if_qc','if_bq','if_wck','if_em','if_sc'])
    df.to_csv(args.output_path,index=False)



   







