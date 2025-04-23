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
import tempfile
import subprocess
from tavily import TavilyClient

from src.my_agent.tools.tool import Tool
from src.my_agent.tools.web_search_tool import WebSearchTool

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # "llama-3.3-70b-versatile"
PHOTO_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
TRANSCRIPTION_MODEL = "whisper-large-v3-turbo"
BASE_CONTEXT = "You are a helpful bot."
MAX_HISTORY_MESSAGES = 20
GROUP_SIZE_LEVEL_0 = 5
GROUP_SIZE_HIGHER_LEVELS = 5
MAX_SUMMARY_LEVEL = 3

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def handle_chat_response(text, update, context, ctx, model, base_context, groq_client, tools: Dict[str, Tool]):
    try:
        logging.debug(f"handle_chat_response: input text: {text}")
        ctx.add_message("user", text)
        tools_context = " ".join([tool.base_context() for tool in tools.values()])
        context_messages = ctx.get_context(system_prompt=f"{base_context} {tools_context} Use the available tools when necessary. Do not generate tool calls manually — use tool_calls field.")
        logging.debug(f"handle_chat_response: prompt/context: {context_messages}")
        logging.debug(f"handle_chat_response: using model: {model}")
        # await context.bot.send_message(chat_id=update.effective_chat.id, text=f">>>>>>>message: {context_messages}\nmodel: {model}\ntools: {[tool.description() for tool in tools.values()]}\ntool_choice: auto")
        chat_completion = groq_client.chat.completions.create(
            messages=context_messages,
            model=model,
            tools=[tool.description() for tool in tools.values()],
            tool_choice="auto"
        )
        logging.debug(f"handle_chat_response: completion object: {chat_completion}")
        response = chat_completion.choices[0].message.content
        logging.debug(f"handle_chat_response: generated response: {response}")

        tool_calls = chat_completion.choices[0].message.tool_calls
        logging.debug(f"handle_chat_response: generated tool_calls: {tool_calls}")
        msg = chat_completion.choices[0].message
        # await context.bot.send_message(chat_id=update.effective_chat.id, text=str(msg))
        if tool_calls:
            context_messages.append({
                "role": "assistant",
                "content": response
            })
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = tools[function_name]
                function_args = json.loads(tool_call.function.arguments)
                logging.debug(f"handle_chat_response: tool_call function name: {function_name}; args: {function_args}")
                # await context.bot.send_message(chat_id=update.effective_chat.id, text=f">>>{function_name}: {function_args}<<<")
                # Call the tool and get the response
                function_response = function_to_call.call(
                    parameters=function_args
                )
                logging.debug(f"handle_chat_response: {function_name} function response: {function_response}")
                # Add the tool response to the conversation
                context_messages.append(
                    {
                        "role": "tool", # Indicates this message is from tool use
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(function_response),
                    }
                )
            # Make a second API call with the updated conversation
            second_response = groq_client.chat.completions.create(
                model=model,
                messages=context_messages
            )
            logging.debug(f"handle_chat_response: second completion object: {second_response}")
            # Return the final response
            second_content = second_response.choices[0].message.content
            ctx.add_message("assistant", second_content)
            await context.bot.send_message(chat_id=update.effective_chat.id, text=second_content)
        else:
            ctx.add_message("assistant", response)
            await context.bot.send_message(chat_id=update.effective_chat.id, text=response)
    except Exception as e:
        logging.error(f"Error in handle_chat_response: {e}", exc_info=True)
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, something went wrong while generating a response.")

def create_start_function(model, photo_model):
    async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"I'm a bot, please talk to me!\nI'm using model {model}!\nFor photo I'm using model {photo_model}!")
    return start

def create_echo_function(groq_client, model, base_context, ctx: ChatContextManager, tools):
    async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.message or not update.message.text:
            logging.warning("No message or text found in update.")
            return
        await handle_chat_response(update.message.text, update, context, ctx, model, base_context, groq_client, tools)
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

def create_transcription_function(groq_client, transcription_model, ctx: ChatContextManager, tools):
    async def transcription(update: Update, context: ContextTypes.DEFAULT_TYPE):
        file_id = update.message.voice.file_id
        file = await context.bot.getFile(file_id)
        location = file.file_path
        logging.info(f"Received audio with file path: {location}")

        try:
            response = requests.get(location, timeout=10)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as ogg_file:
                ogg_file.write(response.content)
                ogg_path = ogg_file.name

            mp3_path = ogg_path.replace(".ogg", ".mp3")
            subprocess.run([
                "ffmpeg",
                "-y",
                "-i", ogg_path,
                "-acodec", "libmp3lame",
                mp3_path
            ], check=True)

            with open(mp3_path, "rb") as mp3_file:
                transcript = groq_client.audio.transcriptions.create(
                    file=mp3_file,
                    model=transcription_model,
                )
                response = transcript.text
                logging.debug(f"Transcript result: {response!r}")
                ctx.add_message("user", response)
                await context.bot.send_message(chat_id=update.effective_chat.id, text=response)
                if not response.strip():
                    logging.warning("Transcript was empty, skipping chat response.")
                    return

            logging.info("Calling handle_chat_response with transcript result.")
            await handle_chat_response(response, update, context, ctx, MODEL, BASE_CONTEXT, groq_client, tools)

        except Exception as e:
            logging.error(f"Error during transcription: {e}")
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, something went wrong while transcribing the audio.")

        finally:
            try:
                os.remove(ogg_path)
                os.remove(mp3_path)
            except Exception as e:
                logging.warning(f"Could not delete temporary files: {e}")

    return transcription

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

def create_tools(tavily_client: TavilyClient) -> dict[str, Tool]:
    return {
        "web_search": WebSearchTool(tavily_client),
    }

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
    tavily_key = os.getenv("TAVILY_API_KEY")
    assert tavily_key is not None, "Tavily key is missing. Check the TAVILY_API_KEY environment variable."
    tavily_client = TavilyClient(api_key=tavily_key)

    model = os.getenv("MODEL", MODEL)
    photo_model = os.getenv("PHOTO_MODEL", PHOTO_MODEL)
    transcription_model = os.getenv("TRANSCRIPTION_MODEL", TRANSCRIPTION_MODEL)
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

    tools: Dict[str, Tool] = create_tools(tavily_client)
    
    application = ApplicationBuilder().token(telegram_token).build()

    start_handler = CommandHandler('start', create_start_function(model, photo_model))
    echo_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), create_echo_function(groq_client, model, base_context, ctx, tools))
    photo_handler = MessageHandler(filters.PHOTO, create_photo_function(groq_client, photo_model, ctx))
    transcription_handler = MessageHandler(filters.VOICE, create_transcription_function(groq_client, transcription_model, ctx, tools))

    application.add_handler(start_handler)
    application.add_handler(echo_handler)
    application.add_handler(photo_handler)
    application.add_handler(transcription_handler)

    application.run_polling()

if __name__ == '__main__':
    main()
