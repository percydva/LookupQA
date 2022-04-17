from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, default_data_collator, Trainer, TrainingArguments
from transformers.pipelines import pipeline
from tqdm import tqdm
import torch
import json
import math
import wikipedia as wiki

raw_squad = load_dataset('squad_v2')
raw_aqa = load_dataset('adversarial_qa', 'adversarialQA')

raw_train = concatenate_datasets([raw_squad['train'],raw_aqa['train'].remove_columns('metadata')])
raw_validation = concatenate_datasets([raw_squad['validation'],raw_aqa['validation'].remove_columns('metadata')])

### Document Reader model ###
class DocumentReader:
    def __init__(self, checkpoint = None, pretrained_path=None, pretrained_model='bert-base-uncased', train_data=None, validation_data=None, model_out='./reader/model', pred_out='./reader/pred'):
        if pretrained_path:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
            self.model = AutoModelForQuestionAnswering.from_pretrained(pretrained_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
            self.model = AutoModelForQuestionAnswering.from_pretrained(pretrained_model)
        
        self.model.cuda()
        self.pipeline = pipeline('question-answering', model=self.model, tokenizer=self.tokenizer, device=0)
        self.max_length = 384
        self.train_data = train_data
        self.validation_data = validation_data
        self.model_out = model_out
        self.pred_out = pred_out
        self.checkpoint = checkpoint
        self.threshold = 0.1
    
    def preprocess_function(self, examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=self.max_length,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            if len(answer['answer_start']):
                start_char = answer["answer_start"][0]
                end_char = answer["answer_start"][0] + len(answer["text"][0])
                sequence_ids = inputs.sequence_ids(i)

                # Find the start and end of the context
                idx = 0
                while sequence_ids[idx] != 1:
                    idx += 1
                context_start = idx
                while sequence_ids[idx] == 1:
                    idx += 1
                context_end = idx - 1

                # If the answer is not fully inside the context, label it (0, 0)
                if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                    start_positions.append(0)
                    end_positions.append(0)
                else:
                    # Otherwise it's the start and end token positions
                    idx = context_start
                    while idx <= context_end and offset[idx][0] <= start_char:
                        idx += 1
                    start_positions.append(idx - 1)

                    idx = context_end
                    while idx >= context_start and offset[idx][1] >= end_char:
                        idx -= 1
                    end_positions.append(idx + 1)
            else:
                start_positions.append(0)
                end_positions.append(0)
        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs
    
    def finetune(self, checkpoint):
        self.tokenized_train = self.train_data.map(self.preprocess_function, batched=True, remove_columns=self.train_data.column_names)
        self.tokenized_validation = self.validation_data.map(self.preprocess_function, batched=True, remove_columns=self.validation_data.column_names)
        
        self.data_collator = default_data_collator
        self.training_args = TrainingArguments(
            output_dir=self.model_out,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=20,
            per_device_eval_batch_size=20,
            num_train_epochs=3,
            weight_decay=0.01,
        )
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.tokenized_train,
            eval_dataset=self.tokenized_validation,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
        )
        torch.cuda.empty_cache()
        if checkpoint:
            self.trainer.train(checkpoint)
        else:
            self.trainer.train()
        self.save()
        self.pipeline = pipeline('question-answering', model=self.model, tokenizer=self.tokenizer, device=0)
        
    def save(self):
        self.model.save_pretrained(self.model_out)
        self.tokenizer.save_pretrained(self.model_out)
        
    def get_answer(self, question, text):   
        total_input_length = len(question.split())+len(text.split())
        text_input_length = len(text.split())
        if total_input_length > self.max_length:
            context_length = self.max_length-len(question.split())
            words = text.split()
            contexts = [' '.join(words[i: min(text_input_length, i+context_length)]) for i in range(0,math.ceil(text_input_length/context_length),context_length)]
        else:
            contexts = [text]
        results = []
        for context in contexts:
            inputs = {
                'question': question,
                'context': context
            }
            results.append(self.pipeline(inputs))
        res = max(results, key=lambda result: result['score'])
        if res['score'] >= self.threshold:
            return res['answer']
        return ''
    
    def get_answers(self, data=None, ans_out=None):
        answers = {}
        if data==None:
            data = self.validation_data
        for example in tqdm(data):
            answers[example['id']] = self.get_answer(example['question'], example['context'])
        if ans_out:
            with open(ans_out,'w+') as of:
                json.dump(answers, of)
        return answers

### Document Retriever model ###
class DocumentRetriever:
    def __init__(self, n_pages = 1, summary=False):
        self.n_pages = n_pages
        self.summary = summary
        
    def get_text(self, question):
        res = wiki.search(question)
        text = " "
        for i in range(min(self.n_pages,len(res))):
            try:
                if self.summary:
                    wiki_page = wiki.summary(res[i])
                else:
                    wiki_page = wiki.page(res[i]).content
            except:
                continue
            text+=wiki_page+' '
        return text
        
### Full end-to-end model - LookupQA ###
class LookupQA:
    def __init__(self, retriever=None, reader=None):
        if retriever:
            self.retriever = retriever
        else:
            self.retriever = DocumentRetriever()
        if reader:
            self.reader = reader
        else:
            self.reader = DocumentReader()
    
    def get_answer(self, question):
        context = self.retriever.get_text(question)
        return self.reader.get_answer(question, context)
            
    def get_answers(self, data=None, ans_out=None):
        answers = {}
        if data==None:
            data = self.validation_data
        for example in tqdm(data):
            answers[example['id']] = self.get_answer(example['question'])
        if ans_out:
            with open(ans_out,'w+') as of:
                json.dump(answers, of)
        return answers