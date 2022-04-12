from flask import Flask,request,jsonify
from collections import OrderedDict
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModel
import re,os, pdftotext



app = Flask(__name__)

model_name = {1:"deepset/bert-base-cased-squad2",
              2:"AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru",
              }


tokenizer = AutoTokenizer.from_pretrained(model_name[1]) 
model = AutoModelForQuestionAnswering.from_pretrained(model_name[1])

############### Saving the tokenizer and model to disk ######################
# tokenizer.save_pretrained('models/')
# model.save_pretrained('models/')


# model_paths ==> [ "bert-base-cased-squad2/", "models/token/" ,"models/QAmodel/"
# tokenizer = AutoTokenizer.from_pretrained("models/", local_files_only=True) 
# model = AutoModel.from_pretrained("models/", local_files_only=True)



def get_chunked_ans(question,text):
  inputs = tokenizer.encode_plus(question, text, return_tensors='pt')
  # identify question tokens (token_type_ids = 0)
  qmask = inputs['token_type_ids'].lt(1)
  qt = torch.masked_select(inputs['input_ids'], qmask)
  print(f"The question consists of {qt.size()[0]} tokens.")

  chunk_size = model.config.max_position_embeddings - qt.size()[0] - 1 # the "-1" accounts for
  # having to add a [SEP] token to the end of each chunk
  print(f"Each chunk will contain {chunk_size - 2} tokens of the context provided.")

  # create a dict of dicts; each sub-dict mimics the structure of pre-chunked model input
  chunked_input = OrderedDict()
  for k,v in inputs.items():
      q = torch.masked_select(v, qmask)
      c = torch.masked_select(v, ~qmask)
      chunks = torch.split(c, chunk_size)

      for i, chunk in enumerate(chunks):
          if i not in chunked_input:
              chunked_input[i] = {}

          thing = torch.cat((q, chunk))
          if i != len(chunks)-1:
              if k == 'input_ids':
                  thing = torch.cat((thing, torch.tensor([102])))
              else:
                  thing = torch.cat((thing, torch.tensor([1])))
          # print('thing : ',thing)
          chunked_input[i][k] = torch.unsqueeze(thing, dim=0)
          # print('final : ',chunked_input[i][k])
  return chunked_input

def convert_ids_to_string(tokenizer, input_ids):
    return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids))


def cat_chunk(chunked_input):
  answer = ''
  # now we iterate over our chunks, looking for the best answer from each chunk
  conf_text = 0
  for _, chunk in chunked_input.items():
      answer_start_scores, answer_end_scores = model(**chunk)

      answer_start,conf_start = torch.argmax(answer_start_scores),round(torch.max(answer_start_scores).item(),2)
      answer_end,conf_end = torch.argmax(answer_end_scores) + 1 , round(torch.max(answer_end_scores).item(),2)
      conf_chunk = max(conf_start,conf_end)
      if(conf_chunk>conf_text):
        conf_text = conf_chunk
      ans = convert_ids_to_string(tokenizer, chunk['input_ids'][0][answer_start:answer_end])
      # if the ans == [CLS] then the model did not find a real answer in this chunk
      if ans != '[CLS]':
          answer += ans + " / "
  return answer, conf_text


############################################# FINAL FUNCTIONs ##############################################

def get_ans(question,context):
  chunked_input = get_chunked_ans(question,context)
  ans,conf = cat_chunk(chunked_input)
  return ans,conf

def PDFtoTEXT(file,path_to_pdf):

  with open(path_to_pdf,'rb') as f:
      pdf = pdftotext.PDF(f)

  pdfText = {}

  # Iterate over all the pages
  for pg_idx,page in enumerate(pdf):
    pdfText['pg'+str(pg_idx)] = page
  return pdfText

def get_ans_dict(pdfText,query):
  ans_dict = {}
  for key,value in pdfText.items():
    pg_no = int(key[2:]) + 1
    print('checking page number ',str(pg_no),'....')
    value = re.sub(r'[^\w\s]','',value)
    ans,conf = get_ans(query,value)
    if(ans != ''):
      key = (str(pg_no), conf)
      ans_dict[key] = ans
  return ans_dict

def argmax_ans(ans_dict):
  fin_ans = None
  conf_temp = 0
  for k,v in ans_dict.items():
    # print(k,v)
    if(k[1]>conf_temp):
      conf_temp = k[1]
      fin_ans = v
  return fin_ans

def clean_folder(folder_path):
  files = glob.glob(folder_path+'/*');
  try:
      for f in files:
          os.remove(f)
  except:
      pass

@app.route('/askME',methods=['POST'])
def antaryami():
    ###### take inputs ############3
    file = request.files['file']
    query = request.form['query']

    ############ solution ################

    #### creatings folders to store data ####
    try:
      os.mkdir('pdf')
    except Exception as e:
      print('Exception occurred : ',e)
    #########################################


    ####### saving input pdf to folder ######
    file_name = file.filename
    path_to_pdf = 'pdf/' + str(file_name)
    file.save(path_to_pdf)
    #########################################

    if(query==''or query==None):
      return jsonify({'message':"No Queries are mentioned !!","success":False})

    pdfText = PDFtoTEXT(file,path_to_pdf)
    ans_dict = get_ans_dict(pdfText,query)
    print('\n','ans_dict : ',ans_dict,'\n')
    fin_ans = argmax_ans(ans_dict)

    ####### convert ans_dict's keys data type from tuple to string because otherwise it can't be jsonified ###########
    ans_dict_mod = {}
    for key,value in ans_dict.items():
      print('key : ',key)
      print('value : ',value)
      ans_dict_mod[str(key)] = value
    ##################################################################################################################
    # return jsonify({"Selected Answer":fin_ans,"success": True})
    clean_folder('pdf')
    return jsonify({"Selected Answer":fin_ans,"All Answers":ans_dict_mod,"success": True})
        


if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=6000)