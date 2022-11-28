import torch
import argparse
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('filename')
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--model', type=str, default='Salesforce/codegen-350M-multi')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--max_length', type=int, default=512)
parser.add_argument('--output', type=str, default='out.html')

HTML_TEMPLATE = """
<!doctype html>

<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">

<title>{site_title}</title>
<meta name="author" content="luungoc2005">

<style>
body {
  font-family: "Lucida Console", "Courier New", monospace;
}

span {
  white-space: pre;
}

.tag {
  position: relative;
}

.tag-extra {
  z-index: 999;
  background-color: gray;
  position: absolute;
  top: 100%;
  left: 0;
  display: none;
  white-space: nowrap;
}

.tag-extra.show {
  display: initial;
}

table {
  margin-top: 8px;
}
</style>
</head>
<body>
{body_content}
<script>
function handleTagHover(event) {
  const element = event.target;
  const targetElement = document.getElementById(`${element.id}-extra`)
  if (!targetElement) return
  targetElement.classList.add('show')
}

function handleTagLeave(event) {
  const element = event.target;
  const targetElement = document.getElementById(`${element.id}-extra`)
  if (!targetElement) return
  targetElement.classList.remove('show')
}

window.addEventListener("load", (event) => {
  document.querySelectorAll('.tag').forEach(element => {
    element.addEventListener('mousemove', handleTagHover)
    element.addEventListener('mouseleave', handleTagLeave) 
  })
});
</script>
</body>
</html>
"""
TAG_TEMPLATE = """
<span id="{tag_id}" class="tag" style="background-color: {text_color}">{tag_text}{table_suggestions}</span>    
"""
TABLE_TEMPLATE = """
<div id="{tag_id}-extra" class="tag-extra">
    <div style="display: flex; flex-direction: row;">
        <div style="text-align: left;">[ {tag_text} ]</div>
        <div style="text-align: right; margin-left: 8px;">{tag_probability}</div>
    </div>
    <table style="width: 100%;">{table_content}</table>
</div>
"""
ROW_TEMPLATE = """
<tr>
    <td>[ {suggestion} ]</td>
    <td style="text-align: right;">{probability}</td>
</tr>
"""

def _get_str(tensor):
    return str(tensor.cpu().tolist())

def _get_color(attr):
    # clip values to prevent CSS errors (Values should be from [-1,1])
    threshold = 0.01
    attr = max(0, min(1, attr))
    if attr > threshold:
        hue = 120
        sat = 75
        lig = 100 - int(50 * attr)
    else:
        hue = 0
        sat = 75
        lig = 100 + 10 * int(math.log10(attr))
    return "hsl({}, {}%, {}%)".format(hue, sat, lig)

def process_tag_text(tag_text):
    return tag_text \
            .replace("\n", "<br>") \
            .replace("\t", "    ")

def get_result_html(file_name, input_text, tokenizer, inputs, output_probs, best_next_tokens_list):
    current_idx = 0
    span_start = 0
    offset_mapping = inputs['offset_mapping'][0]
    input_ids = inputs['input_ids'][0]

    body_content = [] 
    for span_idx, [span_start, span_end] in enumerate(offset_mapping):
        tag_content = ""
        tag_text = process_tag_text(input_text[span_start:span_end])

        if span_idx == 0:
            tag_content = f"<span>{tag_text}</span>"
        else:
            if span_start != offset_mapping[span_idx - 1][1]:
                breaking_text = process_tag_text(input_text[offset_mapping[span_idx - 1][1]:span_start])
                tag_content = f"<span>{breaking_text}</span>"
                body_content.append(tag_content)

            suggest_rows = []
            pred_ids = best_next_tokens_list[span_idx - 1] 
            for token_idx, token in enumerate(tokenizer.convert_ids_to_tokens(pred_ids)):
                suggest_rows.append(ROW_TEMPLATE \
                        .strip() \
                        .replace("{suggestion}", token) \
                        .replace("{probability}", _get_str(output_probs[span_idx - 1][pred_ids[token_idx]])))

            table_suggestions = TABLE_TEMPLATE \
                    .strip() \
                    .replace("{tag_id}", str(span_idx)) \
                    .replace("{tag_text}", tag_text) \
                    .replace("{tag_probability}", _get_str(output_probs[span_idx - 1][input_ids[span_idx]])) \
                    .replace("{table_content}", ''.join(suggest_rows))

            tag_content = TAG_TEMPLATE \
                    .strip() \
                    .replace("{tag_id}", str(span_idx)) \
                    .replace("{text_color}", _get_color(output_probs[span_idx - 1][input_ids[span_idx]])) \
                    .replace("{tag_text}", tag_text) \
                    .replace("{table_suggestions}", table_suggestions)

        body_content.append(tag_content)

    return HTML_TEMPLATE \
            .strip() \
            .replace("{site_title}", file_name) \
            .replace("{body_content}", ''.join(body_content))
                
def get_batches_from_input_ids(input_ids, batch_size, max_length):
    all_input_batches = []
    all_attn_batches = []
    all_pred_batches = []
    all_length_batches = []

    input_batch = []
    attn_batch = []
    pred_batch = []
    length_batch = []

    batch_count = 0

    for ix in range(1, len(input_ids[0])):
        begin = max(0, ix - max_length)
        seq_length = min(ix, max_length)
        
        attn = torch.zeros(1, max_length).long()
        batch = torch.zeros(1, max_length).long()
        
        batch[:, :seq_length] = input_ids[:, begin:ix]
        attn[:, :seq_length] = 1
        
        input_batch.append(batch)
        attn_batch.append(attn)
        pred_batch.append(input_ids[0, ix])
        length_batch.append(seq_length - 1)
        
        batch_count += 1
        if batch_count > batch_size or ix == len(input_ids[0]) - 1:
            all_input_batches.append(torch.cat(input_batch, dim=0))
            all_attn_batches.append(torch.cat(attn_batch, dim=0))
            all_pred_batches.append(pred_batch)
            all_length_batches.append(length_batch)
            input_batch = []
            attn_batch = []
            pred_batch = []
            length_batch = []
            batch_count = 0

    return all_input_batches, all_attn_batches, all_pred_batches, all_length_batches

if __name__ == '__main__':
    args = parser.parse_args()

    device = args.device
    batch_size = args.batch_size
    max_length = args.max_length

    file_name = ""
    with open(args.filename, 'r') as input_file:
        file_content = input_file.read()
        file_name = input_file.name
    
    if file_content == '':
        exit()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)

    print("Finished loading model")

    inputs = tokenizer(file_content, return_tensors="pt", return_offsets_mapping=True).to(device)

    all_input_batches, all_attn_batches, all_pred_batches, all_length_batches = get_batches_from_input_ids(inputs['input_ids'], max_length, batch_size)
        
    best_next_tokens_list = []
    probs_list = []

    for batch_ix, batch in tqdm(enumerate(all_input_batches), total=len(all_input_batches)):
        # print(f'batch: {batch_ix}')
        batch_outputs = model(batch.to(device), attention_mask=all_attn_batches[batch_ix].to(device))
#     batch_outputs = model(batch.to(device))

        for token_ix, token in enumerate(all_pred_batches[batch_ix]):
#         print(token_ix, all_pred_batches[batch_ix])
            next_token_logits = batch_outputs.logits[token_ix, all_length_batches[batch_ix][token_ix], :]
            next_token_scores = torch.nn.functional.softmax(
                next_token_logits, dim=-1
            )
            best_next_tokens = torch.topk(next_token_scores, 5, dim=-1)
            best_next_tokens_list.append(best_next_tokens.indices)
            probs_list.append(next_token_scores)

            # print(f'{tokenizer.convert_ids_to_tokens([token])} - {next_token_scores[token]} - best match: {tokenizer.convert_ids_to_tokens(best_next_tokens.indices)}')

    with open(args.output, 'w') as output_file:
        output_file.write(get_result_html(file_name, file_content, tokenizer, inputs, probs_list, best_next_tokens_list))
