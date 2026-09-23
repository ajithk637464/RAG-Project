# Gateway: one front door, scoped token, denial reaches the model

Config `week9/mcp_config.gateway.json` lists only `servers/gateway.py`. The agent does not open recipe-search or ingredient-db itself. The gateway's tools/list is the union of both, so the discovered names are the same six tools, all labelled server `recipe-gateway`.

Token on that process: `ingredient-db-allergen-only`.

Question: "What allergens does butter have, and what is its nutrition per 100g?"

Step 1 tool=lookup_allergen server=recipe-gateway arguments={"ingredient": "butter"}
Observe: {"ingredient": "butter", "allergens": ["milk"]}

Step 2 tool=lookup_nutrition server=recipe-gateway arguments={"ingredient": "butter"}
Observe: nutrition lookup is not allowed for this token; allergen lookup still works. Call lookup_allergen, and do not invent per-100g numbers.

Step 3 final: Butter is a milk allergen. Nutrition information is not available for this token.

The model reported the denial. It did not invent per-100g numbers.

Audit lines (`week9/gateway_audit.log`), one per tools/call, with caller, tool, and ingredient:

```
caller=recipe-agent tool=lookup_allergen ingredient=butter outcome=ok
caller=recipe-agent tool=lookup_nutrition ingredient=butter outcome=denied
```
