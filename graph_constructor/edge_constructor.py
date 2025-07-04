from typing import List
import openai
import os
import numpy as np

"""
We don't this program at current stage.
As we focus on data crawling and non-llm paper breakdown first.
"""


class EdgeConstructor:

    """
    Rough idea: embedding based idea-evidence matching
    Concern: this matching is based only on semantic similarities. However, idea-evidence matching should be based on how well the evidence supports the idea, not how similar are their textual descriptions.
    """

    def __init__(
        self,
        openai_api_key: str = None,
        embedding_model: str = "text-embedding-ada-002",
        chat_model: str = "gpt-3.5-turbo",
    ):
        if openai_api_key:
            openai.api_key = openai_api_key
        else:
            if os.getenv("OPENAI_API_KEY") is None:
                raise RuntimeError("OPENAI_API_KEY not set in environment and no key provided")
            openai.api_key = os.getenv("OPENAI_API_KEY")
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        pass

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two 1-D numpy arrays."""
        # add small epsilon to norms to avoid division by zero
        norm_a = np.linalg.norm(a) + 1e-10
        norm_b = np.linalg.norm(b) + 1e-10
        return float(np.dot(a, b) / (norm_a * norm_b))


    def match_node_to_idea_embedding(self, node_description : str, ideas: List[str]) -> int:
        """
        node_description: the textual description of node representing figures, tables, links and so on.
        ideas: list of ideas, from which one idea will be chosen to match with the node description.

        This method matches the given node (representing figure, table, link and so on) description to the ideas of the paper
        """

        if not ideas:
            return None


        try:
            inputs = [node_description] + ideas
            resp = openai.Embedding.create(
                input=inputs,
                model=self.embedding_model
            )
            # The API returns resp['data'], a list of dicts with 'embedding' field,
            # in the same order as inputs.
            embeddings = [np.array(item['embedding'], dtype=np.float32) for item in resp['data']]
            node_emb = embeddings[0]
            idea_embs = embeddings[1:]
            sims = [self._cosine_similarity(node_emb, idea_emb) for idea_emb in idea_embs]

            best_idx = int(np.argmax(sims))

            return best_idx

        except Exception as e:
            return None


    def match_node_to_idea_llm(self, node_description : str, ideas: List[str]) -> int:
            try:
                # Build a prompt asking to choose the best matching idea index (0-based).
                idea_list_str = "\n".join(f"{i}: {idea}" for i, idea in enumerate(ideas))
                prompt = (
                    "You are an assistant that matches a node description to the most relevant idea from a list. "
                    "Given the node description and the enumerated list of ideas (with indices), "
                    "return a JSON array with a single integer: the index (0-based) of the best matching idea.\n\n"
                    f"Node description:\n\"\"\"\n{node_description}\n\"\"\"\n\n"
                    "Ideas:\n"
                    f"{idea_list_str}\n\n"
                    "Respond with JSON, e.g.: [3]\n"
                )
                response = openai.ChatCompletion.create(
                    model=self.chat_model,
                    messages=[
                        {"role": "system", "content": "You match descriptions to idea indices."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                )
                content = response.choices[0].message.content.strip()
                # Try to parse JSON array
                import json
                parsed = json.loads(content)
                if isinstance(parsed, list) and parsed and isinstance(parsed[0], int):
                    idx = parsed[0]
                    if 0 <= idx < len(ideas):
                        return idx
                return None
            except Exception:
                return None

    pass