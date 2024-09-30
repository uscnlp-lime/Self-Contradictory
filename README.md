# Self-Contradictory Reasoning Evaluation and Detection

[Paper link]:(https://arxiv.org/abs/2311.09603)

## Dataset

The original dataset is in folder data/original_data. For winogrande, we only use train_m.jsonl in the paper. 

The annotated dataset is in data/annotated_data. For winogrande and winobias, there are annotations for section 1 and section 2 (self-contra evaluation and finer-grained evaluation). For winogender, there is only section 1.

## Generate reasoning

To run the code, first set your Openai API key and Anthropic API key in the environment. 

<code>python generate_reasoning_multiple.py --model MODEL_NAME --dataset DATA --shot few/zero --type WINOBIAS_TYPE --prompt answer/reason --file FILE_NAME --output_dir OUTPUT_DIR</code>

the config 'type' is only for Winobias dataset

## Evaluate the reasoning
The evaluation code is in auto_detection folder

To run binary detection

<code>python evaluate_binary.py --file_path REASONING_FILE --output_path OUTPUT_FILE</code>

To run finer-grained aided detection

<code>python evaluate_fgq.py --file_path REASONING_FILE --output_path OUTPUT_FILE</code>
