import pandas as pd
import openai
import argparse
import os
openai.api_key=os.environ.get('OPENAI_API_KEY')
patterns=['refers to','it can be inferred that','referred to']
# We try to split the reasoning and the answer. Sometimes the answer has word "refers to","inferred", 
# we will first use those patterns to  extract the answer. For the rest, we will use gpt to split the answer and the reasoning
from sklearn.metrics import f1_score
def separate_sentences(sentence):
    segments=[]
    sentences=sentence.split('.')
    for sent in sentences:
        clauses=sent.split(',')
        segments.extend(clauses)
    
    return segments




prompt_extract='You will read a reasoning which include evidence and answer. You task is to precisely extract the answer part from the reasoning, and reply the span from the reasoning given using the same word.'
def exclude_answer(reasoning):
    
    sentences=separate_sentences(reasoning) 
    new_sentences=[]
    for sentence in sentences:

        if patterns[0] in sentence or patterns[1] in sentence:
            continue
        else:
            new_sentences.append(sentence)
    return ','.join(new_sentences)

def process_df(df):
    reasonings=[]
    # questions=[]
    predicted_answers=[]
    new_prompts=[]
    for index,row in df.iterrows():
        
        reason=row['result'].strip()
        reason=reason.replace('\n\n','\n')
        prompt=row['prompt']
        
        print(reason)
        answer,reason=reason.split('\n')[:2]
        reason=reason.split(':')[1].strip()
        answer=answer.split(':')[1].strip()
        if patterns[1] in reason or patterns[0] in reason:
            total+=1
            reason=exclude_answer(reason)
            reasonings.append(reason)
        
            predicted_answers.append(answer)
            new_prompts.append(prompt)
        
        else:
            
            query_result = openai.ChatCompletion.create(
            model="gpt-4-turbo", 
            temperature=0.0,
            messages = [{"role": "system", "content" : "You are an expert in reasoning"},
            {"role": "user", "content" : prompt_extract+'\n'+reason}])
            query_res=query_result['choices'][0]['message']['content']
            reason=reason.replace(query_res,'')
            print(reason)
            new_prompts.append(prompt)
            reasonings.append(reason)
            predicted_answers.append(answer)
    return reasonings, predicted_answers, new_prompts
    
def predict(new_prompts,reasonings):
    new_predictions=[]
    instruction="You are given a question and a corresponding reasoning. Your task is to answer the question using the reasoning."

    for prompt,reason in zip(new_prompts,reasonings):
            index=prompt.index('?')
            prompt=prompt[:index]
            final_prompt=instruction+'The question is: '+prompt+'The reasoning is: '+reason+'Now answer the question based on the reasoning. Only reply the answer. For example, the accountant, the chief.'
            query_result = openai.ChatCompletion.create(
            model="gpt-4-turbo", 
            temperature=0.0,
            messages = [{"role": "system", "content" : "You are an expert in reasoning"},
            {"role": "user", "content" : final_prompt}])
            query_res=query_result['choices'][0]['message']['content']
            new_predictions.append([query_res])
    return new_predictions
    
def compare_answers(ori_answer,new_answer,question):
    
    ori_tokens=ori_answer.split()
    if len(ori_tokens)>=2:
        answer=' '.join(ori_tokens[1:])
    else:
        answer=ori_tokens[0]
    # print(ori_tokens)
    if answer in new_answer and answer in question:
        return True
    else:
        return False

def entail(df):
    entails=[]
    for index, row in df.iterrows():
        ori_answer=row['result'].strip().split('\n')[0].split(':')[1]
        entail=0
        new_answer=row['new_answer'][0]
        question=row['prompt'].split('?')[0]
        if not compare_answers(ori_answer,new_answer,question):

            # print(ori_answer,new_answer,row['self-contra'],row['finer'])
            entail=1
        entails.append(entail)
    print(f1_score(df['self-contra'],df['ir-o']))
    df['ir-o']=entails

if __name__=="__main__":
    parser = argparse.ArgumentParser(
                    prog='evaluate ir to o',
                    description='',
                    epilog='ir->o')
    parser.add_argument('--file_path',default='')
    parser.add_argument('--output_path',default='')
    args=parser.parse_args()
    df=pd.read_csv(args.file_path)
    total=0
    reasonings, predicted_answers, new_prompts = process_df(df)
    new_predictions = predict(new_prompts, reasonings )
    df['new_answer']=new_predictions
    
  
    df.to_csv(args.output_path)