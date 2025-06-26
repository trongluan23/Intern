import openai
openai.api_key = "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"
print([m.id for m in openai.Model.list()])