import os
# Use openai as example
import openai
import tiktoken

from paper_collector.latex_parser import clean_latex_format, clean_latex_code

openai.api_key = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = "gpt-3.5-turbo"
MAX_CHUNK_TOKENS = 3000



class IdeaExtractor:

    def num_tokens(self, text: str, encoding_name: str = "gpt2") -> int:
        """
        Count tokens in text using tiktoken.
        """
        enc = tiktoken.get_encoding(encoding_name)
        return len(enc.encode(text))

    def split_text_into_chunks(self, text: str, max_tokens: int = MAX_CHUNK_TOKENS, encoding_name: str = "gpt2"):
        """
        Split text into chunks not exceeding max_tokens (approx), splitting on paragraph boundaries.
        """
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current = ""
        for p in paragraphs:
            if self.num_tokens(current + "\n\n" + p, encoding_name) <= max_tokens:
                current = current + "\n\n" + p if current else p
            else:
                if current:
                    chunks.append(current)
                # If single paragraph alone is too big, split by sentences
                if self.num_tokens(p, encoding_name) <= max_tokens:
                    current = p
                else:
                    # fallback: split by sentences naively
                    sentences = p.split(". ")
                    current = ""
                    for s in sentences:
                        part = (current + ". " + s).strip() if current else (s + (". " if not s.endswith(".") else ""))
                        if self.num_tokens(part, encoding_name) <= max_tokens:
                            current = part
                        else:
                            if current:
                                chunks.append(current)
                            current = s + (". " if not s.endswith(".") else "")
                    # end sentence loop
        if current:
            chunks.append(current)
        return chunks


    def extract_ideas_with_openai(
        self, 
        text: str,
        model: str = DEFAULT_MODEL,
        max_chunk_tokens: int = MAX_CHUNK_TOKENS,
        temperature: float = 0.0
    ) -> dict:
        """
        Extract ideas from text via OpenAI ChatCompletion:
        - Splits text into chunks if needed
        - For each chunk, asks the model for main ideas, summary, and keywords
        - Aggregates results across chunks
        Returns a dict:
        {
            "chunk_summaries": [ ... ],
            "chunk_keywords": [ ... ],
            "chunk_ideas": [ ... ],
            "combined_summary": str,
            "combined_keywords": [...],
            "combined_ideas": [...],
        }
        """

        if openai.api_key is None:
            raise RuntimeError("OPENAI_API_KEY not set in environment")

        # Remove the unnecessary latex command in the text part.
        text = clean_latex_format(text)


        # Split into chunks
        chunks = self.split_text_into_chunks(text, max_tokens=max_chunk_tokens)
        chunk_summaries = []
        chunk_keywords = []
        chunk_ideas = []

        for idx, chunk in enumerate(chunks, 1):
            prompt = (
                "You are a helpful assistant. Given the following text, please extract:\n"
                "1. A concise summary (2–3 sentences).\n"
                "2. A list of the main ideas or themes (as bullet points).\n"
                "3. A list of top keywords or key phrases.\n\n"
                "Text:\n"
                "'''\n"
                f"{chunk}\n"
                "'''\n"
                "Respond in JSON with keys: summary, ideas, keywords."
            )
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "system", "content": "You extract main ideas from text."},
                        {"role": "user", "content": prompt}],
                temperature=temperature,
            )
            content = response.choices[0].message.content.strip()
            # Try to parse JSON from content
            import json
            try:
                parsed = json.loads(content)
                summary = parsed.get("summary", "")
                ideas = parsed.get("ideas", [])
                keywords = parsed.get("keywords", [])
            except Exception:
                # Fallback: simple parsing if not strict JSON
                # This is naive: look for lines starting with '-' or numbered lists.
                summary, ideas, keywords = None, None, None
                lines = content.splitlines()
                # Extract summary as first paragraph
                summary = lines[0] if lines else ""
                # Extract bullets
                ideas = [line.lstrip(" -\u2022") for line in lines if line.strip().startswith(("-", "•"))]
                # Keywords: look for a line containing 'keywords'
                for line in lines:
                    if "keyword" in line.lower():
                        # after colon?
                        parts = line.split(":", 1)
                        if len(parts) > 1:
                            kws = [k.strip() for k in parts[1].split(",") if k.strip()]
                            keywords = kws
                            break
                keywords = keywords or []
            chunk_summaries.append(summary)
            chunk_ideas.append(ideas)
            chunk_keywords.append(keywords)

        # Combine across chunks: you can simply join summaries, then ask model to summarize that
        combined = "\n\n".join(chunk_summaries)
        # Ask model to combine summaries and dedupe ideas/keywords
        combine_prompt = (
            "You are a helpful assistant. Given the following partial summaries and idea lists, produce:\n"
            "1. A unified concise summary.\n"
            "2. A deduplicated list of main ideas or themes.\n"
            "3. A deduplicated list of keywords/key phrases.\n\n"
            "Inputs:\n"
            "Summaries:\n"
            "'''\n" + combined + "\n'''\n"
            "Ideas lists (JSON array per chunk):\n"
            f"{chunk_ideas}\n"
            "Keywords lists (JSON array per chunk):\n"
            f"{chunk_keywords}\n"
            "Respond in JSON with keys: summary, ideas, keywords."
        )
        response2 = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "system", "content": "You merge and dedupe extracted ideas."},
                    {"role": "user", "content": combine_prompt}],
            temperature=temperature,
        )
        content2 = response2.choices[0].message.content.strip()
        try:
            parsed2 = json.loads(content2)
            combined_summary = parsed2.get("summary", "")
            combined_ideas = parsed2.get("ideas", [])
            combined_keywords = parsed2.get("keywords", [])
        except Exception:
            # Fallback: set to chunk-joined outputs
            combined_summary = combined
            # Flatten and dedupe
            seen = set()
            combined_ideas = []
            for lst in chunk_ideas:
                for item in lst or []:
                    if item not in seen:
                        seen.add(item); combined_ideas.append(item)
            seen = set()
            combined_keywords = []
            for lst in chunk_keywords:
                for item in lst or []:
                    if item not in seen:
                        seen.add(item); combined_keywords.append(item)

        return {
            "chunk_summaries": chunk_summaries,
            "chunk_ideas": chunk_ideas,
            "chunk_keywords": chunk_keywords,
            "combined_summary": combined_summary,
            "combined_ideas": combined_ideas,
            "combined_keywords": combined_keywords,
        }
