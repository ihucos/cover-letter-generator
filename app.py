#!/usr/bin/env uv run --script
# /// script
# dependencies = [
#     "markdownify",
#     "playwright",
#     "PyMuPDF",
#     "llm",
#     "llm-anthropic",
#     "gradio",
#     "redis",
# ]
# ///

import hashlib
import os
import subprocess
import gradio as gr
import fitz  # PyMuPDF
import llm
from markdownify import markdownify as md
from playwright.async_api import async_playwright
import redis.asyncio as redis

# Initialize Redis connection pool
redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)

# assert 0, llm.get_models()

# Configuration
MODEL_ID = "anthropic/claude-haiku-4-5-20251001"
DEFAULT_PROMPT = "Write a short and concise application cover letter."


async def get_model():
    return llm.get_async_model(MODEL_ID)


async def prompt_cached(user_prompt):
    model = await get_model()
    prompt_hash = hashlib.sha256((user_prompt + MODEL_ID).encode("utf-8")).hexdigest()

    cached_answer = await redis_client.get(prompt_hash)
    if cached_answer:
        return cached_answer

    response = await model.prompt(user_prompt)
    answer = await response.text()
    await redis_client.set(prompt_hash, answer, ex=3600)
    return answer


async def fetch_offer(url):
    if not url:
        return ""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.goto(url, wait_until="networkidle", timeout=60000)
            content = await page.content()
            markdown = md(content, strip=["a", "img", "svg"])
            return markdown
        except Exception as e:
            return f"Error fetching URL: {str(e)}"
        finally:
            await browser.close()


async def fetch_pdf(pdf_file):
    if pdf_file is None:
        return ""

    doc = fitz.open(pdf_file.name)
    html_list = [page.get_text("html") for page in doc]
    doc.close()

    document_md = md("".join(html_list), strip=["a", "img", "svg"])
    cleaned_md = await prompt_cached(
        f"Reformat this CV content into clean markdown:\n\n{document_md}"
    )
    return cleaned_md


async def generate_cover_letter(job_url, cv_file, custom_prompt):
    if not job_url or not cv_file:
        yield "### ⚠️ Error\nPlease provide both a Job URL and a CV PDF."
        return

    yield "### ⏳ Step 1/3: Scraping job offer..."
    offer_md = await fetch_offer(job_url)

    yield "### ⏳ Step 2/3: Processing PDF..."
    cv_md = await fetch_pdf(cv_file)

    yield "### ⏳ Step 3/3: Drafting cover letter..."
    model = await get_model()

    # Use the custom prompt provided by the user
    response = await model.prompt(
        custom_prompt,
        fragments=[f"JOB OFFER:\n{offer_md}", f"CV CONTENT:\n{cv_md}"],
    )

    yield await response.text()


# --- Gradio UI ---

with gr.Blocks(title="Cover Letter") as demo:
    gr.Markdown("\n\n# Cover Letter Generator\n")

    with gr.Row():
        with gr.Column():
            job_url_input = gr.Textbox(
                label="Job Posting URL",
                placeholder="https://company.com/jobs/123...",
            )
            cv_upload = gr.File(label="Upload CV (PDF)", file_types=[".pdf"])
            prompt_input = gr.Textbox(label="Prompt", value=DEFAULT_PROMPT, lines=3)
            submit_btn = gr.Button("Generate Letter", variant="primary")

        with gr.Column():
            output_text = gr.Markdown(label="Result")

    # Added prompt_input to the inputs list
    submit_btn.click(
        fn=generate_cover_letter,
        inputs=[job_url_input, cv_upload, prompt_input],
        outputs=output_text,
    )

if __name__ == "__main__":
    # Install playwright browsers on startup
    subprocess.run(["playwright", "install", "chromium"])
    if os.environ.get("PRODUCTION"):
        demo.queue().launch(server_name="0.0.0.0", server_port=80)
    else:
        demo.launch()
