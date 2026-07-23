# Local API configuration template
#
# Copy this file to `configs/api.md`; it is ignored by Git. Replace the two
# global placeholders locally. Do not commit provider keys or response logs.

[global]
base_url = "https://REPLACE_WITH_OPENAI_COMPATIBLE_ENDPOINT/v1"
api_key = "REPLACE_WITH_PROVIDER_KEY"

[tested_agent]
model = "deepseek-v4-flash"
temperature = 0.0
max_tokens = 1024

[user_simulator]
model = "gpt-4o-mini"
temperature = 0.7
max_tokens = 512

[evaluator]
# Example override: all other roles inherit `[global]`, while this role uses
# the official DeepSeek endpoint and a separate key. Uncomment the next two
# lines only when this role needs a different provider.
model = "deepseek-v4-pro"
# base_url = "https://api.deepseek.com/v1"
# api_key = "YOUR_DEEPSEEK_KEY"
temperature = 0.0
max_tokens = 1024

[embedding]
model = "text-embedding-3-small"
temperature = 0.0
max_tokens = 1

[memory]
# Used only by adapters that perform memory extraction or consolidation (e.g. Mem0).
model = "gpt-4o-mini"
temperature = 0.0
max_tokens = 1024
