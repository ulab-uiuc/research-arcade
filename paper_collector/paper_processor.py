import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm


@dataclass
class Figure:
    """Represents a figure in an academic paper."""

    caption: str
    path: str
    name: str
    label: str


class PaperProcessor:
    """Handles the processing of academic papers and their figures."""

    def __init__(self, data_dir: str, figures_dir: str, output_dir: str):
        """Initialize with directory paths for data processing.

        Args:
            data_dir: Directory containing paper JSON files
            figures_dir: Directory containing figure files
            output_dir: Directory for processed output
        """
        self.data_dir = data_dir
        self.figures_dir = figures_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    @staticmethod
    def list_json_files(directory: str) -> List[str]:
        """Get all JSON files in directory."""
        return glob.glob(os.path.join(directory, "*.json"))

    @staticmethod
    def clean_filename(filename: str) -> str:
        """Remove file extensions and standardize filename."""
        return filename.replace(".pdf", "").replace(".png", "").replace(".jpg", "")

    @staticmethod
    def extract_label(latex_label: str) -> Optional[str]:
        """Extract clean label from LaTeX label string."""
        if not latex_label:
            return None
        matches = re.findall(r"\\label\{([^}]+)\}", latex_label)
        return PaperProcessor.clean_filename(matches[0]) if matches else None

    def extract_caption(self, latex_caption: str) -> str:
        """Extract clean caption from LaTeX caption string."""
        if not latex_caption:
            return None
        matches = re.findall(r"\\caption\{([^}]+)\}", latex_caption)
        return PaperProcessor.clean_filename(matches[0].strip()) if matches else None

    def load_paper(self, file_path: str) -> Tuple[Optional[dict], Dict[str, str]]:
        """Load paper JSON and get its figure paths."""
        try:
            arxiv_id = os.path.basename(file_path).split(".json")[0]
            with open(file_path, "r") as f:
                paper_data = json.load(f)
            figure_paths = self.get_figure_paths(arxiv_id)
            return paper_data, figure_paths
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, {}

    def get_figure_paths(self, arxiv_id: str) -> Dict[str, str]:
        """Get mapping of figure names to their full paths."""
        figure_pattern = os.path.join(self.figures_dir, arxiv_id, "*")
        figure_paths = {}

        for path in glob.glob(figure_pattern):
            name = self.clean_filename(os.path.basename(path))
            figure_paths[name] = path

        return figure_paths

    def create_figure_objects(
        self, figures: List[dict], figure_paths: Dict[str, str]
    ) -> List[Figure]:
        """Create Figure objects with path, name, and label."""
        figure_objects = []

        for fig in figures:
            caption = self.extract_caption(fig.get("caption"))
            label = self.extract_label(fig.get("label"))
            if not label or "figure_paths" not in fig:
                continue

            for fig_path in fig["figure_paths"]:
                name = self.clean_filename(fig_path.replace("/", "_"))
                full_path = figure_paths.get(name)

                if full_path:
                    figure_objects.append(
                        Figure(caption=caption, path=full_path, name=name, label=label)
                    )
        print("Latex begin end figure has {}".format(len(figures)))
        print("Figure objects has {}".format(len(figure_objects)))
        return figure_objects

    def find_figure_references(self, text: str) -> List[str]:
        """Find all figure references in text."""
        pattern = r"\\ref\{([^}]+)\}"
        return re.findall(pattern, text)

    def process_paper(self, data: dict, figure_paths: Dict[str, str]) -> Optional[Dict]:
        """Process paper data and extract chunks with figure references."""
        if not all(key in data for key in ["sections", "figure"]):
            return None

        figures = self.create_figure_objects(data["figure"], figure_paths)
        label_to_figures = self._create_label_mapping(figures)

        all_chunks = []
        fig_chunks = []

        for section in data["sections"].values():
            chunks = [
                chunk.strip()
                for chunk in section["content"].split("\n\n")
                if chunk.strip()
            ]

            for chunk in chunks:
                all_chunks.append({"content": chunk})
                if "\\ref{" in chunk:
                    referenced_figures = []
                    for label in self.find_figure_references(chunk):
                        if label in label_to_figures:
                            referenced_figures.extend(
                                [
                                    {
                                        "caption": fig.caption,
                                        "path": fig.path,
                                        "name": fig.name,
                                        "label": fig.label,
                                    }
                                    for fig in label_to_figures[label]
                                ]
                            )

                    if referenced_figures:
                        fig_chunks.append(
                            {"content": chunk, "figures": referenced_figures}
                        )

        print("I find {} chunks with figure references".format(len(fig_chunks)))
        return {
            "fig_connected_chunks": fig_chunks,
            "all_chunks": all_chunks,
        }

    @staticmethod
    def _create_label_mapping(figures: List[Figure]) -> Dict[str, List[Figure]]:
        """Create mapping from labels to figures."""
        label_to_figures = {}
        for fig in figures:
            if fig.label not in label_to_figures:
                label_to_figures[fig.label] = []
            label_to_figures[fig.label].append(fig)
        return label_to_figures

    def save_processed_data(self, data: dict, original_path: str):
        """Save processed paper data to JSON file."""
        if not data:
            return

        filename = os.path.basename(original_path).replace(".json", "-processed.json")
        save_path = os.path.join(self.output_dir, filename)

        with open(save_path, "w") as f:
            json.dump(data, f, indent=4)

    def process_all_papers(self):
        """Process all papers in the data directory."""
        for paper_path in tqdm(self.list_json_files(self.data_dir)):
            paper_data, figure_paths = self.load_paper(paper_path)
            if paper_data:
                processed_data = self.process_paper(paper_data, figure_paths)
                self.save_processed_data(processed_data, paper_path)
