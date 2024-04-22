import lmql

# lmql.model("local:meta-llama/Llama-2-13b-hf", tokenizer="togethercomputer/LLaMA-2-7B-32K", trust_remote_code=True)


@lmql.query
def chain_of_thought(input_sentence):
    '''lmql
sample(temperature=0.4, max_len=4096)

"Please analyze the sentence: {input_sentence} Does it explicitly mention any of the digits from 0 to 9 as the answer? If a digit is mentioned, state which one. Otherwise, indicate 'None'. The digit mentioned is: [ANSWER]" where ANSWER in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "None"]
from
    lmql.model("local:meta-llama/Llama-2-13b-hf", cuda=True)
    return ANSWER
    '''

print(chain_of_thought("The image shows a single digit, which is the number 1. The digit is displayed in a black and white color scheme, making it stand out against the dark background. The image is a close-up of the digit, emphasizing its prominence in the scene."))

print(chain_of_thought("The image shows a single digit, which is the number 3. The digit is displayed in a black and white color scheme, making it stand out against the dark background. The image is a close-up of the digit, emphasizing its prominence in the scene."))


print(chain_of_thought("The image shows a single digit, which is the number 8. The digit is displayed in a black and white color scheme, making it stand out against the dark background. The image is a close-up of the digit, emphasizing its prominence in the scene."))


print(chain_of_thought("The image shows a single digit.The digit is displayed in a black and white color scheme, making it stand out against the dark background. The image is a close-up of the digit, emphasizing its prominence in the scene."))
