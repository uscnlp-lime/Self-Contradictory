import pandas as pd
from sklearn.metrics import f1_score,recall_score
import argparse
def get_result_ensemble(df):
    
    pred_labels=[]
    for _,row in df.iterrows():
        if_qc=row['if_qc'].split()[0]
        if_bq=row['if_bq'].split()[0]
        if_wck=row['if_wck'].split()[0]
        if_em=row['if_em'].split()[0]
        if_sc=row['if_sc'].split()[0]
        acc=row['acc']
        if acc==0:
            if 'yes' in if_qc or 'yes' in if_bq or 'yes' in if_wck or 'yes' in if_sc:
                pred_labels.append(0)
            else:
                pred_labels.append(1)
        else:
            if 'yes' in if_qc or 'yes' in if_bq or 'yes' in if_wck or 'yes' in if_sc:
                pred_labels.append(1)
            else:
                pred_labels.append(0)
    return pred_labels
   
def compare_answers(ori_answer,new_answer,question):
    
    ori_tokens=ori_answer.split()
    if len(ori_tokens)>=2:
        answer=' '.join(ori_tokens[1:])
    else:
        answer=ori_tokens[0]
    
    if answer in new_answer and answer in question:
        return True
    else:
        return False
    
def get_result_binary(df):
    return df['pred'].to_list()
def get_result_ir_o(df):
    result=[]
    new_answers=df['new_prediction']
    ori_answers=df['result']
    questions=df['prompt']
    for question,ori,new in zip(questions,ori_answers,new_answers):
        question=question.split('?')[0].strip('Question:')
        ori=ori.strip().split('\n')[0].strip('Answer:')
        if compare_answers(ori,new,question):
            result.append(0)
        else:
            result.append(1)
    return result

if __name__=="__main__":
   

    parser = argparse.ArgumentParser(
                    prog='get metric',
                    description='',
                    epilog='Get metric')
    parser.add_argument('--ensemble_file',default='')
    parser.add_argument('--binary_file',default='')
    parser.add_argument('--ir_o_file',default='')
    args=parser.parse_args()
  
    df_ensemble=pd.read_csv(args.ensemble_file)
    df_binary=pd.read_csv(args.binary_file)
    df_ir_o=pd.read_csv(args.ir_o_file)

    ensemble_result=get_result_ensemble(df_ensemble)
    binary_result=get_result_binary(df_binary)
    ir_o_result=get_result_ir_o(df_ir_o)
    gold_labels=df_ir_o['self-contra'].to_list()
    # deprecated cause IR-O score is low
    # print("IR->O f1 score:",f1_score(gold_labels,ir_o_result))

    print("Binary f1 score:",f1_score(gold_labels,binary_result))
    print("Ensemble f1 score:",f1_score(gold_labels,ensemble_result))

    majority_vote_result=[]
    for binary,ensemble in zip(binary_result,ensemble_result):
        
        if binary==1:
            majority_vote_result.append(1)
        elif ensemble==1:
            majority_vote_result.append(1)
        else:
            majority_vote_result.append(0)
    print("Binary+ensemble f1 score:",f1_score(gold_labels,majority_vote_result))