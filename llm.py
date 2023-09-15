import os
import time
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained("./llama-7b")
model = LlamaForCausalLM.from_pretrained("./llama-7b")
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"


filelist=os.listdir('./movieprompt')
for promptfile in filelist:
    with open('./movieprompt/'+promptfile,'r') as fprompt:
        prompt=fprompt.read()
        batch = tokenizer(prompt,return_tensors="pt",padding=True)
        batch = {k: v for k, v in batch.items()}
        with torch.inference_mode():
            print("Start")
            t0 = time.time()
            generated = model.generate(batch["input_ids"], attention_mask=batch["attention_mask"], max_length=2000, no_repeat_ngram_size=5, temperature=0.7, top_k=40, top_p=0.1, repetition_penalty=1/0.85, do_sample=True, bad_words_ids=[[396],[29892],[2277],[29937],[3579],[461],[4136],[673]])
            t1 = time.time()
            print(f"Output generated in {(t1-t0):.2f} seconds")
            for g in generated:
                with open('./movieprompt2result/'+promptfile,'w') as fout:
                    fout.write(tokenizer.decode(g))

