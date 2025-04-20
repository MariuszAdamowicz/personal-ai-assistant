from dotenv import load_dotenv
import os
import logging
from telegram import Update
from telegram.ext import filters, MessageHandler, ApplicationBuilder, CommandHandler, ContextTypes
import openai

MODEL = "llama-3.3-70b-versatile"

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def create_start_function(model):
    async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"I'm a bot, please talk to me!\nI'm using model {model}!")
    return start

def create_echo_function(groq_client, model):
    async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": update.message.text,
                }
            ],
            model=model,
        )
        await context.bot.send_message(chat_id=update.effective_chat.id, text=chat_completion.choices[0].message.content)
    return echo

def main():
    load_dotenv()

    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    print(f"Telegram: {telegram_token}")    # For Debug only; TODO: Remove from prod release
    assert telegram_token is not None, "Telegram token is missing. Check the TELEGRAM_BOT_TOKEN environment variable."

    groq_token = os.getenv("GROQ_API_KEY")
    assert groq_token is not None, "Groq token is missing. Check the GROQ_API_KEY environment variable."
    groq_client = openai.OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_token
    )
    
    application = ApplicationBuilder().token(telegram_token).build()

    start_handler = CommandHandler('start', create_start_function(MODEL))
    echo_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), create_echo_function(groq_client, MODEL))

    application.add_handler(start_handler)
    application.add_handler(echo_handler)

    application.run_polling()

if __name__ == '__main__':
    main()
