<p align="center">
    <b>AwaDB - the AI Native database for embedding vectors</b>. <br />
    Store and search embedding vectors for LLM Applications!
</p>


The core API is only 4 steps: 

```python
import awadb
# 1. Initialize awadb client!
awadb_client = awadb.Client()

# 2. Create table!
awadb_client.Create("testdb")

# 3. Add docs to the table. Can also update and delete the doc!
awadb_client.Add([{'id':'123'}, {'name':'jim'}, {'gender':'male'}, {'age':39}, 'hello', 'world', [1, 3.5, 3]])
awadb_client.Add([{'id':'235'}, {'name':'vincent'}, {'gender':'male'}, {'age':28}, 'what', 'doing', [1, 3.4, 2]])
awadb_client.Add([{'id':'398'}, {'name':'david'}, {'gender':'female'}, {'age':45}, 'yu', 'hi', [1, 2.4, 4]])
awadb_client.Add([{'id':'345'}, {'name':'tom'}, {'gender':'female'}, {'age':25}, 'hhuhu', 'hello', [1.3, 2.9, 8.9]])

# 4. Search by specified vector query and the most TopK similar results
results = awadb_client.Search([3.0, 3.1, 4.2], 3)

# Output the results
print(results)

```

You can also directly use awadb to do the text semantic retrieval
Here the text is embedded by SentenceTransformer which is supported by Hugging Face(https://huggingface.co)   
```another example
import awadb
# 1. Initialize awadb client!
awadb_client = awadb.Client()

# 2. Create table
awadb_client.Create("test_llm1") 

# 3. Add sentences, the sentence is embedded with SentenceTransformer by default, you can also embed the sentences all by yourself with OpenAI or other LLMs
awadb_client.Add([{'embedding_text':'The man is happy'}, {'source' : 'pic1'}])
awadb_client.Add([{'embedding_text':'The man is very happy'}, {'source' : 'pic2'}])
awadb_client.Add([{'embedding_text':'The cat is happy'}, {'source' : 'pic3'}])
awadb_client.Add(['The man is eating', 'pic4'])

# 4. Search the most Top3 sentences by the specified query
query = "The man is happy"
results = awadb_client.Search(query, 3)

# Output the results
print(results)

```

## What are the Embeddings?

Any unstructured data(image/text/audio/video) can be transferred to vectors which are generally understanded by computers through AI(LLMs or other deep neural networks). For example, "The man is happy"-this sentence can be transferred to a 384-dimension vector(a list of numbers `[0.23, 1.98, ....]`) by SentenceTransformer language model. This process is called embedding.

More detailed information about embeddings can be read from OpenAI(https://platform.openai.com/docs/guides/embeddings/what-are-embeddings)

Awadb uses [Sentence Transformers](https://huggingface.co/sentence-transformers) to embed the sentence by default, while you can also use OpenAI or other LLMs to do the embeddings according to your needs.


## Combined with LLMs(OpenAI, Llama, Vicuna, Alpace, ChatGLM, Dolly) and Langchain


## Get involved

Welcome any PR contributors or ideas to improve the project. 
- [Roadmap and your ideas](https://github.com/awa-ai/awadb/roadmap)
- [Issues and PR](https://github.com/awa-ai/awadb/issues)

## License

[Apache 2.0](./LICENSE)
