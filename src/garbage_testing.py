# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from dotenv import load_dotenv

# load_dotenv()  

# # Set your API key


# # Initialize the model (you can also use 'gemini-embedding-2-preview')
# embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", api_key=os.getenv("GOOGLE_API_KEY"))

# # Test the embedding
# query_result = embeddings.embed_query("This is a test document.")
# print(f"Embedding length: {len(query_result)}")