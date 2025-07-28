<div style="display: flex; align-items: center;">
  <h1>Chart-R1: Chain-of-Thought Supervision and Reinforcement for Advanced Chart Reasoner</h1>
</div>

<div align="center">
<a href='https://arxiv.org/abs/2507.15509'><img src='https://img.shields.io/badge/Arxiv-2507.15509-b31b1b.svg?logo=arXiv'></a>&ensp;<a href='https://huggingface.co/collections/DocTron/chart-r1-68834834a239e09e9abcb5f4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face%20-models-blue'></a>&ensp;<a href=https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE><img src='https://img.shields.io/badge/License-Apache_2.0-green.svg'></a>

Lei Chen, Xuanle Zhao, Zhixiong Zeng†, Jing Huang, Yufeng Zhong, Lin Ma*
</div>
<div align="center">
<strong>Meituan Group</strong>
</div>
<div align="center">
† Project Leader; * Corresponding Author
</div>


---
**Chart-R1** is a vision-language model that enables complex chart reasoning through reinforcement learning fine-tuning. As the **first** to apply R1-Style methods to the chart domain, it employs programmatic data synthesis to generate high-quality step-by-step reasoning data for charts. Chart-R1's two-stage training includes Chart-COT (chain-of-thought supervision) and Chart-RFT (numerically sensitive reinforcement fine-tuning). Experiments show Chart-R1 achieves significant advantages on open-source benchmarks and the ChartRQA dataset, comparable to large-scale models like GPT-4o and Claude-3.5, proving R1-Style effectiveness for chart reasoning.
<div align="center">
<img src="./assets/chart_r1_radar.png"  width="100%">
</div>

## 📢 News and Updates
* ```2025.07.25``` We upload our model weights [Chart-R1](https://huggingface.co/DocTron/Chart-R1) and [Chart-COT](https://huggingface.co/DocTron/Chart-COT) to HuggingFace.
* ```2025.07.21``` 🔥🔥🔥 We release the technical report of **Chart-R1** at arXiv [link](https://arxiv.org/abs/2507.15509).


## 🤗 Models
|  Model   | Download Link  |
|  ----  | ----  |
|  Chart-COT |  [DocTron/Chart-COT](https://huggingface.co/DocTron/Chart-COT)  |
|  Chart-R1  |  [DocTron/Chart-R1](https://huggingface.co/DocTron/Chart-R1)   |

The ```Chart-COT``` is Qwen2.5-VL-7B-Instruct fine-tuned with supervised learning on the ChartRQA-SFT dataset. The ```Chart-R1``` is Chart-COT further optimized through reinforcement fine-tuning (RFT).


## 📊 Performance
<table>
<thead>
  <tr>
    <th rowspan="3"></th>
    <th rowspan="3">Model Name</th>
    <th colspan="5">Chart Reasoning Benchmarks</th>
  </tr>
  <tr>
    <th rowspan="2">ChartQA</th>
    <th rowspan="2">CharXiv-RQ</th>
    <th rowspan="2">ChartQAPro</th>
    <th colspan="2">ChartRQA</th>
  </tr>
  <tr>
    <th>single</th>
    <th>multi</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="5">Proprietary</td>
    <td>GPT-4o</td>
    <td>85.7</td>
    <td>47.1</td>
    <td>37.67</td>
    <td>44.37</td>
    <td>46.55</td>
  </tr>
  <tr>
    <td>Gemini-1.5-Flash</td>
    <td>79.0</td>
    <td>33.9</td>
    <td>42.96</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>Gemini-1.5-Pro</td>
    <td>87.2</td>
    <td>43.3</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>Gemini-2.5-Flash</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>59.12</td>
    <td>59.17</td>
  </tr>
  <tr>
    <td>Claude-3.5-Sonnet</td>
    <td>90.8</td>
    <td>60.2</td>
    <td>43.58</td>
    <td>52.79</td>
    <td>56.05</td>
  </tr>
  <tr>
    <td rowspan="5">General-domain Open-source</td>
    <td>Phi-3.5-Vision</td>
    <td>81.8</td>
    <td>32.7</td>
    <td>24.73</td>
    <td>31.08</td>
    <td>24.32</td>
  </tr>
  <tr>
    <td>DeepSeek-VL2</td>
    <td>86.0</td>
    <td>-</td>
    <td>16.28</td>
    <td>23.15</td>
    <td>20.29</td>
  </tr>
  <tr>
    <td>InternVL3-8B</td>
    <td>86.6</td>
    <td>37.6</td>
    <td>-</td>
    <td>37.51</td>
    <td>31.73</td>
  </tr>
  <tr>
    <td>InternVL3-38B</td>
    <td>89.2</td>
    <td>46.4</td>
    <td>-</td>
    <td>46.09</td>
    <td>38.36</td>
  </tr>
  <tr>
    <td>Qwen2.5-VL-7B</td>
    <td>87.3</td>
    <td>42.5</td>
    <td>36.61</td>
    <td>44.59</td>
    <td>40.57</td>
  </tr>
  <tr>
    <td rowspan="5">Chart-domain</td>
    <td>ChartLlama</td>
    <td>69.66</td>
    <td>14.2</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>TinyChart</td>
    <td>83.60</td>
    <td>8.3</td>
    <td>13.25</td>
    <td>6.75</td>
    <td>6.11</td>
  </tr>
  <tr>
    <td>ChartGemma</td>
    <td>80.16</td>
    <td>12.5</td>
    <td>6.84</td>
    <td>7.18</td>
    <td>9.23</td>
  </tr>
  <tr>
    <td>ChartReasoner</td>
    <td>86.93</td>
    <td>-</td>
    <td>39.97</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td><b>Chart-R1-7B (Ours)</b></td>
    <td><b>91.04</b></td>
    <td><b>46.2</b></td>
    <td><b>44.04</b></td>
    <td><b>52.09</b></td>
    <td><b>49.93</b></td>
  </tr>
</tbody>
</table>


## 🔍 Usage Example
Below is a simple example of how to use Chart-R1 for multimodal reasoning tasks:
```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Load model
model_path = 'DocTron/Chart-R1'

# Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto", attn_implementation="flash_attention_2"
)

# Use the following system_prompt and pixel range by default
system_prompt = "Solve the question. The user asks a question, and you solves it. You first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> Since 1+1=2, so the answer is 2. </think><answer> 2 </answer>, which means assistant's output should start with <think> and end with </answer>."

processor = AutoProcessor.from_pretrained(model_path, min_pixels=1280*28*28, max_pixels=16384*28*28)

# Set generation parameters by default
generate_kwargs = dict(
    max_new_tokens=2048,
    top_p=0.001,
    top_k=1,
    temperature=0.01,
    repetition_penalty=1.0
)

# Prepare input with image and text
messages = [
    {
        "role": "system",
        "content": system_prompt
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "assets/example_case.jpg",
            },
            {"type": "text", "text": "What is the difference in percentage of U.S. people who thinks scientists should take active part in policy debates and those thinks they should focus on establishing sound scientific facts?"},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, **generate_kwargs)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text[0])

# <think>Step 1: Identify the percentage of U.S. people who think scientists should 'Take an active role in policy debates'. This is 60%. Step 2: Identify the percentage of U.S. people who think scientists should 'Focus on establishing sound scientific facts'. This is 39%. Step 3: Calculate the difference between these two percentages: 60% - 39% = 21%.</think><answer>21</answer>
```


## 📌 Acknowledgement
We sincerely appreciate [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and [MM-EUREKA](https://github.com/ModalMinds/MM-EUREKA) for providing reference training framework.


## 📖 Citation
If you find this project useful, please feel free to leave a star and cite our paper:
```
@misc{chen2025chartr1,
      title={Chart-R1: Chain-of-Thought Supervision and Reinforcement for Advanced Chart Reasoner}, 
      author={Lei Chen and Xuanle Zhao and Zhixiong Zeng and Jing Huang and Yufeng Zhong and Lin Ma},
      year={2025},
      eprint={2507.15509},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2507.15509}, 
}
```
