Who wrote it: this ingredient-db process is a local stand-in for the content team's server, and once configured it runs inside the agent trust boundary as third-party code.
What it can reach: the bundled ingredient table only; it does not read the cookbook, the network, or the recipe-search tools.
What it logs: every lookup appends the tool name, the ingredient string, and the token scope to week9/ingredient_db.log.
Stolen token: ingredient-db-full can read allergen flags and per-100g nutrition; ingredient-db-allergen-only cannot read nutrition; either token shows which ingredients users asked about.
Ship or don't: do not ship the full token; ship only the allergen-scoped token, and not until that query log is access-controlled.
