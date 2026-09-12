from llm_provider import LLMProvider


llm = LLMProvider()

response = llm.generate(

    system_prompt=(
        "You are a concise Amazon customer "
        "support assistant."
    ),

    user_prompt=(
        "A customer says their package "
        "was supposed to arrive yesterday. "
        "Write a short helpful response."
    )
)

print("\nProvider response:")
print(response)