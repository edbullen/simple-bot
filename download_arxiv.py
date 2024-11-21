# download the Arxiv academic papers

"""
 Open-access repository of electronic preprints and postprints (known as e-prints) approved for posting after
 moderation, but not peer review. It consists of scientific papers in the fields of mathematics, physics, astronomy,
 electrical engineering, computer science, quantitative biology, statistics, mathematical finance and economics, which
 can be accessed online. In many fields of mathematics and physics, almost all scientific papers are self-archived on
 the arXiv repository before publication in a peer-reviewed journal.
"""

import arxiv
import os

# Directory to save PDFs
DIR = "./data/arxiv/"
#os.makedirs(save_dir, exist_ok=True)

# Search query (e.g., Computer Science)
SEARCH_QUERY = "cat:cs.*"  # 'cs' is the category for Computer Science


def get_papers(save_dir, search_query):
    # Fetch results (limit to a manageable number)
    search = arxiv.Search(
        query=search_query,
        max_results=20,  # Adjust this for more or fewer results
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    for result in search.results():
        pdf_url = result.pdf_url
        file_name = os.path.join(save_dir, f"{result.get_short_id()}.pdf")
        # Download PDF
        result.download_pdf(dirpath=save_dir)
        print(f"Downloaded {file_name}")


if __name__ == '__main__':
    get_papers(DIR, SEARCH_QUERY)
