import argparse
from openai import OpenAI
import os
from dotenv import load_dotenv
from database import get_recent_messages
from rich.console import Console
from rich.markdown import Markdown

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def summarize_messages(messages, chat_history=None):
    if not messages and not chat_history:
        return "No messages found."

    combined_messages = "\n".join(messages)
    history = "\n".join(chat_history) if chat_history else ""

    prompt = (
        f"\n\n<messages>\n{combined_messages}\n</messages>\n\n"
        f"<chat_history>\n{history}\n</chat_history>\n\n"
        "Please answer the following question based on the above information."
    )

    try:
        response = client.chat.completions.create(model="o1-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=10000,
        n=1)

        summary = response.choices[0].message.content.strip()
        return summary

    except Exception as e:
        return f"An error occurred while summarizing the messages: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Interactive Discord Message Summarizer.")
    parser.add_argument("--server_id", type=str, required=True, help="The ID of the Discord server to analyze")
    args = parser.parse_args()

    console = Console()
    chat_history = []

    console.print(f"Welcome to the Interactive Discord Message Summarizer for server {args.server_id}.\n")

    try:
        # Prompt for keywords once at the start
        keywords_input = input("Enter keywords (comma-separated) or type 'exit' to quit: ").strip()
        if keywords_input.lower() == 'exit':
            console.print("Exiting the summarizer. Goodbye!")
            return

        # Parse and clean keywords
        keywords = [keyword.strip() for keyword in keywords_input.split(',') if keyword.strip()]
        if not keywords:
            console.print("No valid keywords entered. Exiting.")
            return

        # Fetch recent messages based on the provided keywords
        recent_messages = get_recent_messages(args.server_id, 24 * 30, keywords)
        message_count = len(recent_messages)

        console.print(f"\nLoaded {message_count} recent messages matching keywords {keywords} from server {args.server_id}.\n")

        while True:
            user_question = input("Enter your question or type 'exit' to quit: ").strip()
            if user_question.lower() == 'exit':
                console.print("Exiting the summarizer. Goodbye!")
                break

            chat_history.append(f"User: {user_question}")

            summary = summarize_messages(recent_messages, chat_history)
            chat_history.append(f"AI: {summary}")

            console.print(f"\nSummary:\n")
            md = Markdown(summary)
            console.print(md)
            console.print("-" * 50 + "\n")

    except KeyboardInterrupt:
        console.print("\nInterrupted by user. Exiting.")
    except Exception as e:
        console.print(f"An error occurred: {str(e)}\n")

if __name__ == "__main__":
    main()
