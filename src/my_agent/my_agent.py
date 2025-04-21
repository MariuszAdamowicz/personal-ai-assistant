import json
from typing import List, Dict

from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import os
import logging
from telegram import Update
from telegram.ext import filters, MessageHandler, ApplicationBuilder, CommandHandler, ContextTypes
import openai

from src.my_agent.chat_context_manager import ChatContextManager
import requests

MODEL = "llama-3.3-70b-versatile"
PHOTO_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
BASE_CONTEXT = "You are a helpful bot."
MAX_HISTORY_MESSAGES = 20
GROUP_SIZE_LEVEL_0 = 5
GROUP_SIZE_HIGHER_LEVELS = 5
MAX_SUMMARY_LEVEL = 3

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def create_start_function(model, photo_model):
    async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"I'm a bot, please talk to me!\nI'm using model {model}!\nFor photo I'm using model {photo_model}!")
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

def create_photo_function(groq_client, photo_model, ctx: ChatContextManager):
    async def photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
        idx = len(update.message.photo)-1
        file_id = update.message.photo[idx].file_id
        file = await context.bot.getFile(file_id)
        location = file.file_path
        caption = update.message.caption
        logging.info(f"Received photo with file path: {location}")
        logging.info(f"Caption: {caption}")
        messages = [
            {
                "role": "system",
                "content": "You are an assistant specialized in analyzing and describing images in detail."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's in this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": location
                        }
                    }
                ]
            }
        ]

        if caption:
            messages.append({
                "role": "user",
                "content": caption
            })

        logging.info("Checking if the file is a valid image...")
        if is_valid_image(location):
            logging.info(f"Image passed validation: {location}")
            try:
                completion = groq_client.chat.completions.create(
                    messages=messages,
                    model=photo_model,
                )
                response = completion.choices[0].message.content
                ctx.add_message("user", response)
                await context.bot.send_message(chat_id=update.effective_chat.id, text=response)
            except Exception as e:
                logging.error(f"Error during image completion: {e}")
                await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, something went wrong while analyzing the image.")
        else:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="The specified file is not an image.")
    return photo

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

def is_valid_image(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        Image.open(BytesIO(response.content))
        return True
    except Exception as e:
        logging.error(f"Image validation failed for URL {url}: {e}")
        return False

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
    photo_model = os.getenv("PHOTO_MODEL", PHOTO_MODEL)
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

    start_handler = CommandHandler('start', create_start_function(model, photo_model))
    echo_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), create_echo_function(groq_client, model, base_context, ctx))
    photo_handler = MessageHandler(filters.PHOTO, create_photo_function(groq_client, photo_model, ctx))

    application.add_handler(start_handler)
    application.add_handler(echo_handler)
    application.add_handler(photo_handler)

    application.run_polling()

if __name__ == '__main__':
    main()
