import json
from typing import List, Dict

from dotenv import load_dotenv
import os
import logging
from telegram import Update
from telegram.ext import filters, MessageHandler, ApplicationBuilder, CommandHandler, ContextTypes
import openai

from src.my_agent.chat_context_manager import ChatContextManager

MODEL = "llama-3.3-70b-versatile"
BASE_CONTEXT = "You are a helpful bot."
MAX_HISTORY_MESSAGES = 20
GROUP_SIZE_LEVEL_0 = 5
GROUP_SIZE_HIGHER_LEVELS = 5
MAX_SUMMARY_LEVEL = 3

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def create_start_function(model):
    async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"I'm a bot, please talk to me!\nI'm using model {model}!")
    return start

def create_echo_function(groq_client, model, base_context, ctx: ChatContextManager):
    async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
        ctx.add_message("user", update.message.text)
        context_messages = ctx.get_context(system_prompt=base_context)
        chat_completion = groq_client.chat.completions.create(
            messages=context_messages,
            model=model,
        )
        response = chat_completion.choices[0].message.content
        ctx.add_message("assistant", response)
        await context.bot.send_message(chat_id=update.effective_chat.id, text=response)
    return echo

def generate_summary(groq_client, model):
    def summarize(messages: List[Dict[str, str]]) -> str:
        response = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Create a concise summary of the following conversation between the user and the assistant. Summarize in a clear and structured way using bullet points if helpful. Focus on key questions, concepts, and answers. Omit small talk."},
                {"role": "user", "content": json.dumps(messages, ensure_ascii=False)}
            ]
        )
        return response.choices[0].message.content.strip()
    return summarize

def main():
    load_dotenv()

    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    assert telegram_token is not None, "Telegram token is missing. Check the TELEGRAM_BOT_TOKEN environment variable."

    groq_token = os.getenv("GROQ_API_KEY")
    assert groq_token is not None, "Groq token is missing. Check the GROQ_API_KEY environment variable."
    groq_client = openai.OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_token
    )

    model = os.getenv("MODEL", MODEL)
    base_context = os.getenv("BASE_CONTEXT", BASE_CONTEXT)
    max_history_messages = int(os.getenv("MAX_HISTORY_MESSAGES", MAX_HISTORY_MESSAGES))
    group_size_level_0 = int(os.getenv("GROUP_SIZE_LEVEL_0", GROUP_SIZE_LEVEL_0))
    group_size_higher_levels = int(os.getenv("GROUP_SIZE_HIGHER_LEVELS", GROUP_SIZE_HIGHER_LEVELS))
    max_summary_level = int(os.getenv("MAX_SUMMARY_LEVEL", MAX_SUMMARY_LEVEL))

    summarize_fn = generate_summary(groq_client, model)
    ctx = ChatContextManager(
        session_id="my_session",
        max_history_messages=max_history_messages,
        summarize_fn=summarize_fn,
        group_size_level_0=group_size_level_0,
        group_size_higher_levels=group_size_higher_levels,
        max_summary_level=max_summary_level
    )
    
    application = ApplicationBuilder().token(telegram_token).build()

    start_handler = CommandHandler('start', create_start_function(model))
    echo_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), create_echo_function(groq_client, model, base_context, ctx))

    application.add_handler(start_handler)
    application.add_handler(echo_handler)

    application.run_polling()

if __name__ == '__main__':
    main()
