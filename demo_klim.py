import os

from raptor.SummarizationModels import Phi3SummarizationModel

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

print('+++++initializing RetrievalAugmentation')

retrievalAugmentationConfig = RetrievalAugmentationConfig(
    embedding_model=SBertEmbeddingModel(),
    qa_model=LocalPhi3Model(),
    summarization_model=Phi3SummarizationModel())

RA = RetrievalAugmentation(retrievalAugmentationConfig)

# construct the tree
print('+++++constructing the tree')
RA.add_documents(text)

question = "How did Cinderella reach her happy ending?"
answer = RA.answer_question(question=question)
print(f"+++++Answer to the question {question} from the tree just created: ", answer)

SAVE_PATH = "demo/cinderella_klim"
RA.save(SAVE_PATH)

print("+++++Just saved the tree to: ", SAVE_PATH)

RA = RetrievalAugmentation(config=retrievalAugmentationConfig,
                           tree=SAVE_PATH)
answer = RA.answer_question(question=question)

print(f"+++++Answer to the question {question} from the tree just loaded tree: ", answer)
