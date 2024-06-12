import os

# Cinderella story defined in sample.txt
with open('demo/sample.txt', 'r') as file:
    text = file.read()

print(text[:100])

from raptor import RetrievalAugmentation
from raptor import RetrievalAugmentationConfig
from raptor import SBertEmbeddingModel
from raptor import LocalPhi3Model

# p_qa_model = LocalPhi3Model()
# p_embedding_model = SBertEmbeddingModel()
retrievalAugmentationConfig = RetrievalAugmentationConfig(
    embedding_model=SBertEmbeddingModel(),
    qa_model=LocalPhi3Model(),
    summarization_model=LocalPhi3Model())
RA = RetrievalAugmentation(retrievalAugmentationConfig)

# construct the tree
RA.add_documents(text)


# SBertEmbeddingModel()
#
# LocalPhi3Model()