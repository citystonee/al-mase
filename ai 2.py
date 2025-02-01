import openai
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Your OpenAI API key (replace with your actual key)
openai.api_key = "your_openai_api_key_here"

def chatbot_response(user_input, model="gpt-3.5-turbo", temperature=0.7, max_tokens=150):
    """
    Generate a response to the user's input using OpenAI's GPT model.

    :param user_input: The input text from the user.
    :param model: The model to use (default is 'gpt-3.5-turbo').
    :param temperature: The creativity level of the response.
    :param max_tokens: Maximum tokens for the response.
    :return: The chatbot's response as a string.
    """
    try:
        logger.info("Sending request to OpenAI API...")
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": user_input}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        logger.info("Response received successfully.")
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Error during API call: {e}")
        return "Sorry, I couldn't process your request. Please try again later."

def main():
    """
    Main loop to interact with the chatbot.
    """
    print("Welcome to ChatGPT! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("ChatGPT: Goodbye!")
            break
        response = chatbot_response(user_input)
        print(f"ChatGPT: {response}")

if __name__ == "__main__":
    main()
