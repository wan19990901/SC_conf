# Chain of Thought (CoT) Evaluation and Golden CoT Creation Procedure for each Question

## Procedure:

Open the the test.html file, load the provided csv file directly on it and do the two tasks below:

## Task 1: Evaluating best_cot and sc_cot

For each entry in the dataset, evaluate both the best_cot and sc_cot based on the following criteria:

# Human Evaluation Criteria

## 1. Overall Quality [1-5]
Assesses the correctness and justification of the response.

1. Incomprehensible or completely incorrect
2. Mostly incorrect with major reasoning flaws
3. Partially correct with notable gaps or minor errors
4. Mostly correct with minor imperfections
5. Clear, correct, and fully justified

## 2. Coherency [1-5]
Evaluates logical flow, structure, and conciseness of the reasoning chain.

1. Completely incoherent or excessively verbose
2. Mostly incoherent with occasional clarity
3. Partially coherent but with structural issues or unnecessary verbosity
4. Mostly coherent with minor clarity or conciseness issues
5. Perfectly coherent, well-structured, and concise

**Note:** A higher Coherency score should be given to responses that effectively communicate ideas with brevity while maintaining clarity.

## Task 2: Creating Golden CoT

For each entry in the dataset:

1. Review the second_best_cot (golden CoT).
2. Assess whether the flow of logic in the rationale closely resembles what a human solver would do. Consider the following:
3. If the second_best_cot does not closely resemble human-like reasoning (if yes then keep it as it is):
   - Rewrite a new "golden CoT" based on the your own judgement.

## Save the result

Please click the Download the results and send the saved csv file back to me; Please also try to not quit before you finished them or you might have to start again. 