from openai import OpenAI

def get_llm_blended(
    prompt: str,
    text_a: str,
    text_b: str,
    client,
    model: str,
) -> str:
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": "You are a semantic blending assistant."
            },
            {
                "role": "user",
                "content": f"{prompt}\n\n"
                f"Text A: {text_a}\n"
                f"Text B: {text_b}\n\n"
                "Return only the blended sentence. Do not use any special symbols."
            }
        ],
        temperature=0.2,
    )
    output = response.output_text.strip()

    if output.startswith('"') and output.endswith('"'):
        output = output[1:-1]
        
    output = output.replace('"', '')

    return output.strip()

def blend_instructions(
    text_a: str,
    text_b: str,
    blend_type: str,
    client,
    model: str,
) -> str:
    if blend_type == "concat":
        return f"{text_a} {text_b}"

    elif blend_type == "mix":
        prompt = (
            "Synthesize the two descriptions into a unified spatial narrative."
            "Blend overlapping or related elements when possible, and reorganize the scene to read as a naturally designed environment."
            "Avoid comma-separated listing."
        )
        return get_llm_blended(prompt, text_a, text_b, client, model)

    elif blend_type == "a_base":
        prompt = (
            "Text A is the base instruction. "
            "Text B is an additional modifier instruction. "
            "Rewrite Text A by incorporating elements from Text B."
        )
        return get_llm_blended(prompt, text_a, text_b, client, model)

    elif blend_type == "b_base":
        prompt = (
            "Text B is the base instruction. "
            "Text A is an additional modifier instruction. "
            "Rewrite Text B by incorporating elements from Text A."
        )
        return get_llm_blended(prompt, text_a, text_b, client, model)

    else:
        raise ValueError(f"Unknown blend type: {blend_type}")