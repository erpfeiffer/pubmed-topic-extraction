# Pubmed Topic Extraction

What are common subtopics, and their distribution over time, within the hundreds to thousands of publication records pulled down from a Pubmed search?

This project implements gensim's Latent Dirichlet Allocation (LDA) module to identify topics in a data set based on a user-specified Pubmed search query. Common topics and their associated terms are visualized using networkx. Topic frequency and distribution over time are also examined.

This project was developed in Python 3.5, using Bio, gensim, and networkx. It will be necessary to customize the script, providing a contact email (e.g. Entrez.email = "name@host.com") prior to querying Entrez. 
