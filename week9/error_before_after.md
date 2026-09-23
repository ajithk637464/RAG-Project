# Same failing call: find_substitute("creme fraiche lite")

Question both times: "What can I use instead of creme fraiche lite?"

The model is the host (llama3.2 via rag.complete_chat). The recipe-search server does not call a model. Only the server env changed: SUBSTITUTE_ERROR_MODE=legacy versus the default. The agent module is the same file.

## Before — docstring "Find a substitute. Returns Error 3 on failure."

Step 1 tool=find_substitute server=recipe-search arguments={"q": "creme fraiche lite substitute"}
Observe: find_substitute needs an ingredient name.

Step 2 tool=find_substitute server=recipe-search arguments={"ingredient": "creme fraiche lite"}
Observe: Error 3

Step 3 final: You can use sour cream or plain Greek yogurt as a substitute for creme fraiche lite.

The model treated "Error 3" as a dead end and invented a substitute. Nothing in that reply came from the substitute table.

## After — docstring tells the model to call find_substitute again with the name after "try"

Docstring the server advertised: "Look up one ingredient in the local substitute table. Input: {\"ingredient\": \"the ingredient name\"}. If nothing matches, the text names a simpler ingredient after the word 'try'. Call find_substitute again with that exact name. Do not invent a substitute, and do not treat the miss as a dead server."

Step 1 tool=find_substitute server=recipe-search arguments={"ingredient": "creme fraiche lite"}
Observe: no substitute matched 'creme fraiche lite': try 'creme fraiche'

Step 2 tool=find_substitute server=recipe-search arguments={"ingredient": "creme fraiche"}
Observe: {"ingredient": "creme fraiche", "substitute": "sour cream, or equal parts heavy cream and plain yogurt", "source": "recipe-search substitute table"}

Step 3 repeats that same successful call.

Step 4 final: sour cream, or equal parts heavy cream and plain yogurt

The same miss is now a spelling hint. The model called find_substitute again with the suggested name, and the final answer is the table row.
