import torch
import argparse
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from packaging import version

parser = argparse.ArgumentParser()
parser.add_argument('filename')
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--model', type=str, default='Salesforce/codegen-350M-multi')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--max_length', type=int, default=2048)
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
    if torch.is_tensor(tensor):
        return str(tensor.cpu().tolist())
    else:
        return str(tensor)

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
    span_start = 0
    offset_mapping = inputs['offset_mapping'][0]

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
            pred_ids, pred_probs = best_next_tokens_list[span_idx - 1]

            for token_idx, token in enumerate(tokenizer.convert_ids_to_tokens(pred_ids)):
                suggest_rows.append(ROW_TEMPLATE \
                        .strip() \
                        .replace("{suggestion}", token) \
                        .replace("{probability}", _get_str(pred_probs[token_idx])))

            table_suggestions = TABLE_TEMPLATE \
                    .strip() \
                    .replace("{tag_id}", str(span_idx)) \
                    .replace("{tag_text}", tag_text) \
                    .replace("{tag_probability}", _get_str(output_probs[span_idx - 1])) \
                    .replace("{table_content}", ''.join(suggest_rows))

            tag_content = TAG_TEMPLATE \
                    .strip() \
                    .replace("{tag_id}", str(span_idx)) \
                    .replace("{text_color}", _get_color(output_probs[span_idx - 1])) \
                    .replace("{tag_text}", tag_text) \
                    .replace("{table_suggestions}", table_suggestions)

        body_content.append(tag_content)

    return HTML_TEMPLATE \
            .strip() \
            .replace("{site_title}", file_name) \
            .replace("{body_content}", ''.join(body_content))
                
def get_batches_from_input_ids(input_ids, batch_size, max_length):

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

        if batch_count >= batch_size or ix == len(input_ids[0]) - 1:
            input_batch = torch.cat(input_batch, dim=0)
            attn_batch = torch.cat(attn_batch, dim=0)

            yield input_batch, attn_batch, pred_batch, length_batch

            input_batch = []
            attn_batch = []
            pred_batch = []
            length_batch = []
            batch_count = 0

def torch_jit_model_eval(model, example_batch):
    try:
        jit_model = model.eval()
        with torch.no_grad():
            if version.parse(version.parse(torch.__version__).base_version) >= version.parse("1.14.0"):
                if isinstance(example_batch, dict):
                    jit_model = torch.jit.trace(jit_model, example_kwarg_inputs=example_batch, strict=False)
                else:
                    jit_model = torch.jit.trace(
                        jit_model,
                        example_kwarg_inputs={key: example_batch[key] for key in example_batch},
                        strict=False,
                    )
            else:
                jit_inputs = []
                for key in example_batch:
                    example_tensor = torch.ones_like(example_batch[key])
                    jit_inputs.append(example_tensor)
                jit_inputs = tuple(jit_inputs)
                jit_model = torch.jit.trace(jit_model, jit_inputs, strict=False)
        jit_model = torch.jit.freeze(jit_model)
        jit_model(**example_batch)
        jit_model(**example_batch)
        model = jit_model
    except (RuntimeError, TypeError, ValueError, NameError, IndexError) as e:
        print(f"failed to use PyTorch jit mode due to: {e}.")

    return model

def train_iter(model, model_inputs, pred_batch, length_batch):
    best_next_tokens_list = []
    probs_list = []

    batch_outputs = model(**model_inputs)
    # batch_outputs = model(batch.to(device))

    for token_ix, token in enumerate(pred_batch):
#         print(token_ix, all_pred_batches[batch_ix])
        next_token_logits = batch_outputs.logits[token_ix, length_batch[token_ix], :]
        next_token_scores = torch.nn.functional.softmax(
            next_token_logits, dim=-1
        )
        best_next_tokens = torch.topk(next_token_scores, 5, dim=-1)

        best_next_tokens_list.append((best_next_tokens.indices.detach().numpy(), best_next_tokens.values.detach().numpy()))
        probs_list.append(next_token_scores[token].detach().numpy())

    return best_next_tokens_list, probs_list

def main(args):
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
    input_ids = inputs['input_ids']

    best_next_tokens_list = []
    probs_list = []
    jit_model = None

    for batch_ix, batch in tqdm(
            enumerate(get_batches_from_input_ids(input_ids, batch_size, max_length)), 
            total=math.ceil(len(input_ids[0]) / batch_size)
        ):
        input_batch, attn_batch, pred_batch, length_batch = batch
        model_inputs = {
            'input_ids': input_batch.to(device),
            'attention_mask': attn_batch.to(device),
        }

        if jit_model is None:
            jit_model = torch_jit_model_eval(model, model_inputs)
            del model
        # print(f'batch: {batch_ix}, {batch.size()}')
        
        with torch.no_grad():
            best_next_tokens, probs = train_iter(jit_model, model_inputs, pred_batch, length_batch)
        
        best_next_tokens_list.extend(best_next_tokens)
        probs_list.extend(probs)

    with open(args.output, 'w') as output_file:
        output_file.write(get_result_html(file_name, file_content, tokenizer, inputs, probs_list, best_next_tokens_list))

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    main(args)
