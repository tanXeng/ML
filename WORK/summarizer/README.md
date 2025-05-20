###########################################################################################
######################## OVERVIEW ON SUMMARY TASK #########################################
###########################################################################################
1. `/home/dsta/saic/summary_trainer_mistral_lora.py` -> First fine tuning of mistral 7B model for general writing format and picking out relevant subjects to include in summary.
    Mistral 7B model is trained on:
        - 2024 (article content-human summary) pairs 
        - 4 bit config
        - Supervised Fine Tuning Trainer (SFTTrainer)
    The outputs from this file are as follows:
        A Coy: '/home/dsta/saic/models/A_Coy/A_mistral_with_lora_fine_tuned'
        B Coy: '/home/dsta/saic/mistral_with_lora_fine_tuned'
        C Coy: '/home/dsta/saic/models/C_Coy/C_mistral_with_lora_fine_tuned'
        D Coy: '/home/dsta/saic/models/D_Coy/D_mistral_with_lora_fine_tuned'
        
2. `/home/dsta/saic/CoT_summarizer.py`, USE_DPO = False -> Chain of Thought (CoT) summary file WITHOUT Direct Preference Optimization (DPO)
    - CoT is done with the qLoRA fine tuned models for the respective Coys
        - chunk the article 
        - extract_elements(): extract "entities", "events", "locations", "results" from each chunk
        - generate_summary(): generate a summary based on elements extracted 
    - 2023 data is split into 2 halves: `2023_odd.xlsx` and `2023_even.xlsx`
        - 2023 data is used to generate machine summaries as we don't want to reuse the training data from 2024 (seen in step 1)
        - We split the 2023 into even rows and odd rows so the dates of the articles are still trained on the whole year
        - We use 2023_odd.xlsx for training now because we will use 2023_even.xlsx for testing later
    - The output we will get is an excel file called `odd_year_few_shot.xlsx` with the following columns:
        1. article_content
        2. human_written_summary
        3. machine_written_summary
        
3. `/home/dsta/saic/DPO.py` -> Direct Preference Optimization (DPO) file
    - We train on the data from `odd_year_few_shot.xlsx` using DPO training as follows:
        - "prompt": article_content
        - "chosen": human_written_summary
        - "rejected": machine_written_summary
    - library used is DPOTrainer from trl
    - The outputs of this file are as follows:
        A Coy: '/home/dsta/saic/models/dpo_summarizer_large/A/checkpoint-288'
        B Coy: '/home/dsta/saic/models/dpo_summarizer_large/B/checkpoint-414'
        D Coy: '/home/dsta/saic/models/dpo_summarizer_large/D/dpo_summarizer_200/checkpoint-69'

4. `/home/dsta/saic/CoT_summarizer.py`, USE_DPO = True -> Chain of Thought (CoT) summary file WITH DPO
    - Same steps from (2), but we generate summaries from 200 articles from `2023_even.xlsx`
    


###########################################################################################
######################## UNDERSTANDING PROCESS_EXCEL_FILE #################################
###########################################################################################
In `process_excel_file()`, there are a few pointers to note:
    - Before running `CoT_summarizer.py`, you will have to check the following indexes of where your Column indexes are located (article_content, date, human_summary, etc.)
    - Generally, `process_excel_file()` will output the same excel file, but with the added columns:
        - machine_written_summary
        - Named Entity Recognition (NER) results from human written summary
        - NER results from machine written summary
        - BERTScore f1 between machine summary and human summary
    - You can then go to `/home/admin/Documents/saic_notebook.ipynb` (NOT IN DGX!!!) and run the cell under "Run this to output the comparison between NER results of Human summary and Machine summary"
        - This will output the comparison between human and machine NER results
            ✔ means that the jaccard similarity between the human and machine entry is more than 0.67
            ✘ means that it fails and there is no significant matching entity
    - Files can be found at: `/home/admin/Documents/saic_summary/for nsf to test/evaluation/`
        - {coy}_output_comparison.xlsx is the full file
        
        

###########################################################################################
######################## RUNNING THE FRONT END CODE #######################################
###########################################################################################
*** Take note that this frontend code is not supposed to be operational. It is merely for testing out how the overall frontend MIGHT look like. If you want to improve it, you can find a way to load the checkpoints only when u click "generate summary" button on the streamlit page. Currently every time some element changes on the streamlit page, the whole model loads a new checkpoint again, which might crash the system. ***

The frontend code is located in DGX 1 (192.168.1.105): /home/dsta/saic/frontend.py
To run this code: 
    - `CUDA_VISIBLE_DEVICES=0 streamlit run frontend.py` -> you MUST use CUDA_VISIBLE_DEVICES to run this
    - go to 192.168.1.105:8501
    - upload any B coy document for testing (I use /home/admin/Documents/saic_summary/B/B_2023_even_rows.xlsx)
    - ONLY CLICK GENERATE SUMMARY FOR ROW 0
    - For the stream lit app, only "AI Summary" and "Extracted Elements" are working tabs. The rest are not.
    
    

###########################################################################################
######################## FUTURE EVALUATION METRICS ########################################
###########################################################################################
- Can try testing out:
    Mistral-small
    open-mixtral model
    flan-t5
    BART (Bidirectional and Auto-Regressive Transformers)x
    Prompt optimization and perspective-aware summarization: https://arxiv.org/html/2503.11118v1
    
- 3 evaluation metrics to beat:
    1: BERTScore F1
    2: Comparison between human and machine NER Results (ideally every extracted NER result from human summary should be covered by NER results from machine generated)
    3: ROUGE-L F1 score should be higher
    You can use files from this folder for evaluation: `/home/admin/Documents/saic_summary/for nsf to test/evaluation/`
