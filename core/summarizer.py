from models.llm import llm

def generate_summary(transcript):
    """Generates a summary of the transcript."""
    prompt = f"""
    You are an expert at summarizing YouTube videos. Please provide a concise summary of the following transcript. 
    Focus on the main points and key takeaways. Use bullet points for clarity.

    Transcript:
    {transcript}
    """
    response = llm.invoke(prompt)
    return response.content