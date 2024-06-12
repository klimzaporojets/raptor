import os

# Cinderella story defined in sample.txt
with open('demo/sample.txt', 'r') as file:
    text = file.read()

print(text[:100])

from raptor import RetrievalAugmentation
from raptor import RetrievalAugmentationConfig
from raptor import SBertEmbeddingModel
from raptor import LocalPhi3Model

p_qa_model = LocalPhi3Model()
p_embedding_model = SBertEmbeddingModel()
RA = RetrievalAugmentation(RetrievalAugmentationConfig(embedding_model=p_embedding_model
                                                       , qa_model=p_qa_model))

# construct the tree
RA.add_documents(text)


# SBertEmbeddingModel()
#
# LocalPhi3Model()