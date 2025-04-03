from dotenv import load_dotenv
import os
import logging
from telegram import Update
from telegram.ext import filters, MessageHandler, ApplicationBuilder, CommandHandler, ContextTypes

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="I'm a bot, please talk to me!")

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text=update.message.text)

def main():
    load_dotenv()
    
    # openai_key = os.getenv("OPENAI_API_KEY")
    # print(f"OpenAI: {openai_key}")    # For Debug only; Remove from prod release
    # assert openai_key is not None, "Missing OpenAI Key"

    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    print(f"Telegram: {telegram_token}")    # For Debug only; Remove from prod release
    assert telegram_token is not None, "Missing Telegram Token"
    
    application = ApplicationBuilder().token(telegram_token).build()

    start_handler = CommandHandler('start', start)
    echo_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), echo)

    application.add_handler(start_handler)
    application.add_handler(echo_handler)

    application.run_polling()

if __name__ == '__main__':
    main()

