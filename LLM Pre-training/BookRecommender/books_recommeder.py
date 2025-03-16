import pandas as pd
import numpy as np

from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain.embeddings import FastEmbedEmbeddings
from langchain_text_splitters import CharacterTextSplitter

import gradio as gr


books = pd.read_csv('books_with_emotions.csv')
# Make the image larger
books['large_thumbnail'] = books['thumbnail'] + "&fife=w800"
books['large_thumbnail'] = np.where(books['large_thumbnail'].isna(), 
                                    'cover-not_founc.jpg',
                                    books['large_thumbnail'])

raw_document = TextLoader('tagged_descriptions.txt').load()
text_splitter = CharacterTextSplitter(separator='\n', chunk_size=0, chunk_overlap=o)
documents = text_splitter.split_documents(raw_document)

# Create a vector database and store the embeddings of the documents
embedding_model = FastEmbedEmbeddings()
db_books = Chroma(embedding_function=embedding_model, persist_directory="./chroma_db")
batch_size = 50
for i in range(0, len(documents), batch_size):
    batch = documents[i : i + batch_size]
    db_books.add_documents(batch)


def retrieve_semantic_recommendations(
        query : str,
        category : str = None,
        tone : str = None,
        initial_top_k : int = 50,
        final_top_k : int = 10
) -> pd.DataFrame:
    # Get the top k records based on the similarity search
    records = db_books.similarity_search(query, k=initial_top_k)

    # Get the isbn for all the records
    books_list = [int(rec.page_content.strip('"').split()[i]) for rec in records]

    # Get the books based on the isbn value
    book_recs = books[books['isbn13'].isin(books_list)]

    # Filter the records based on the category
    if category != 'All':
        book_recs = book_recs[book_recs['simple_categories'] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    # Sort the records based on the tone
    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs


def recommend_books(query, category, tone):
    results = []
    recommendations = retrieve_semantic_recommendations(query, category, tone)

    for _, row in recommendations.iterrows():
        description = row['description']

        authors_split = row['authors'].split(';')

        if len(authors_split) == 2:
            authors = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors = f"{", ".join(authors_split[:-1])} and {authors_split[-1]}"
        else:
            authors = row['authors']

        caption = f"{row['title']} by {authors}: {description}"
        results.append((row['large_thumbnail'], caption))

    return results

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book:",
                                placeholder = "e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a category:", value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone:", value = "All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended books", columns = 8, rows = 2)

    submit_button.click(fn = recommend_books,
                        inputs = [user_query, category_dropdown, tone_dropdown],
                        outputs = output)


if __name__ == "__main__":
    dashboard.launch()