# Evaluation

At some point I got tired of judging the chatbot by squinting at screenshots, so I wrote an evaluation script.    
*eval.py* connects to the POST /chat endpoint, performs some queries and scores the results with RAGAS, comparing the answers with *golden_set.json* (35 questions).   
The metrics are the ones provided by RAGAS, plus:
- numeric: search a specific substring in the answer
- sentinel: checks for NO_LOCAL_ANSWER if the answer should not be found (basically, the model should NOT hallucinate stuff)
- avg_calls: just the average tool calls
- no_call: how many query answered with no tool usage (hopefully 0)

## Result table

I may keep this table, with the improvements as they come.

| label | faithfulness | answer_relevancy | semantic_similarity | factual_correctness (f1) | ctx_precision | ctx_recall | numeric | sentinel | avg_calls | no_call | avg_secs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma-4-31b-it-new-prompt-28-08 | 0.935 | 0.862 | 0.903 | 0.689 | 0.412 | 0.847 | 0.943 | 0.943 | 1.91 | 0 | 7.4 |
| gemma-4-31b-it-baseline-28-08 | 0.916 | 0.818 | 0.914 | 0.671 | 0.391 | 0.793 | 0.886 | 0.914 | 1.77 | 0 | 6.6 |

## Considerations
- 28/08: new prompt seems to improve the performance of the model. Low context precision, maybe K_document is too high. Also, maybe hybridsearch can fix the tests that are failing.

