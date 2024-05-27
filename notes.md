This is a document to gather relevant papers:

- \[[arxiv](https://arxiv.org/html/2401.04925v3)\] The Impact of Reasoning Step Length on Large Language Models

- \[[arxiv](https://arxiv.org/abs/2307.08678)\] Do Models Explain Themselves? Counterfactual Simulatability of Natural Language Explanations

Question:

IV for Invalid answer?
DIF_SUB does not work
too slow if data gets large
Need a automatic/systematic way to select features
need different window size (as ablation study) to justify our work

TO DO:

Plots and experiments for different N


1: Fix features (by selection) (select on ROC or final results?); think about how to formalize the entire process.

2: extract features should be given pre-defined and should be able to run on test data directly.

3: llama (parser)

4: Viz more (different models).

5: code; setting code ready for pre-trained model (no need to extract features and fit models everytime; also saved model can be used on new data directly)

6: thresold (good results; how to justify?)


# Figures to have


## Methods




TO DO:

ES DATA AND AC DATA (need three seeds?), NEED TO MAKE THEM READY LIKE final.csv, and make information like Name and Model ready.

### Figure

Coefficients of logistic regression three different similarity (which motivates us define our parameters)

##  Main Results

### Table: 

Row Name Methods：
Columns: Dataset/

Time and Cost analysis

### Figure

Comparsion of Different budget and accuracy at different steps for selected dataset

AuROC for our scoring methods

## Abalation Study 

### Table:

Different Scoring/Decision Models

### Figure:

Different N/thresholds Models

### Appendix


To do today:

1: statistical test, find datasets make data nicer.
2: intergate BERT/LR/Customized LR into the code; write notebooks to compare each of them.
3: pull the notebook and show the results with data from llama3
4: clean the code
5: get readt the rebuttal




