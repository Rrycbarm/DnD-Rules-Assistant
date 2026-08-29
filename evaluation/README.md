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
| gemma-4-31b-it-hybrid=0.5-document-k=10-29-08 | **0.726** | **0.957** | **0.916** | 0.302 | **0.916** | **0.971** | **1.000** | 1.71 | 0 | 5.9 |
| gemma-4-31b-it-hybrid=0.5-29-08 | 0.695 | 0.900 | 0.903 | 0.308 | 0.872 | 0.914 | 0.971 | 1.89 | 0 | **5.1** |
| gemma-4-31b-it-document-k=10-29-08 | 0.701 | 0.924 | 0.913 | 0.346 | 0.854 | 0.943 | 0.943 | 1.86 | 0 | 5.4 |
| gemma-4-31b-it-new-prompt-28-08 | 0.706 | 0.936 | 0.906 | 0.348 | 0.860 | 0.943 | 0.943 | 2.00 | 0 | 8.1 |
| gemma-4-31b-it-baseline-28-08 | 0.669 | 0.867 | 0.909 | **0.375** | 0.755 | 0.886 | 0.914 | 1.60 | 0 | **5.1** |

## Updates
- 29/08: increasing DOCUMENT_K does not improve performance. The hybrid search (with BM25_WEIGHT at 0.5) does not improve the performance. But both together improved the retrieval (ogre is now found). The only failing question is the one of the longsword: the retrieval finds some *monsters* which use a longsword, and uses their stats instead of the stats of the *weapon* longsword. Maybe a filter on the category of the file could help, excluding files on another topic. In that case, maybe it could be possible to reduce again DOCUMENT_K. Context precision is low, so there are useless references in the context.
- 28/08: New prompt seems to improve the performance of the model. Low context precision, maybe DOCUMENT_K is too high. The remaining numeric failures are ogre and troll hit points, and longsword damage: hybrid search might help. The avg_sec does not seem to be relevant, often depends on the OpenRouter provider.

