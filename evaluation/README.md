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

| label | factual_correctness (f1) | faithfulness | semantic_similarity | ctx_precision | ctx_recall | numeric | sentinel | avg_calls | no_call | avg_secs |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma-4-31b-it-new-prompt-28-08 | **0.706** | **0.936** | 0.906 | 0.348 | **0.860** | **0.943** | **0.943** | 2.0 | 0 | 8.1 |
| gemma-4-31b-it-baseline-28-08 | 0.669 | 0.867 | **0.909** | **0.375** | 0.755 | 0.886 | 0.914 | 1.6 | 0 | **5.1** |

## Updates
- 28/08: New prompt seems to improve the performance of the model. Low context precision, maybe DOCUMENT_K is too high. The remaining numeric failures are ogre and troll hit points, and longsword damage: hybrid search might help. The avg_sec does not seem to be relevant, often depends on the OpenRouter provider.

