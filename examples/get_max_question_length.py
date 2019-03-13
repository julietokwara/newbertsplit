import json
def get_max_question_length():
    """Read a SQuAD json file into a list of SquadExample."""
    files = ["train-v2.0.json", "dev-v2.0.json","test-v2.0.json"]

    for input_file in files:
        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)["data"]

        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False
        max_q = float("-inf")
        mquestion = ""
        examples = []
        for entry in input_data:
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"]
                doc_tokens = []
                char_to_word_offset = []
                prev_is_whitespace = True
                for c in paragraph_text:
                    if is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(doc_tokens) - 1)

                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]


                    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
                    query_tokens = tokenizer.tokenize(example.question_text)
                    question_length = len(query_tokens)
                    if(question_length > max_q):
                        max_q = query_tokens
                        mquestion = question_text

    print(mquestion)
    return max_q
#is_training, version_2_with_negative

train = "/Users/julietokwara/documents/juniorWinter/cs224n/project/pytorch-pretrained-BERT/data/train-v2.0.json"
dev = "/Users/julietokwara/documents/juniorWinter/cs224n/project/pytorch-pretrained-BERT/data/dev-v2.0.json"
test = "/Users/julietokwara/documents/juniorWinter/cs224n/project/pytorch-pretrained-BERT/data/test-v2.0.json"

print("max train question length is " + str (get_max_question_length(train) ) )
print("max dev question length is " + str (get_max_question_length(dev)) )
print("max train question length is " + str(get_max_question_length(test)) )

