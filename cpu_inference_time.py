import torch
from transformers import OPTForCausalLM, AutoTokenizer
import time

# 设置设备为CPU
device = torch.device("cpu")

# 加载模型和分词器
model_name = "facebook/opt-2.7b"  # 可以替换为其他模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = OPTForCausalLM.from_pretrained(model_name).to(device)

# 设置为评估模式
model.eval()

# 示例输入句子
input_sentence = "Hello, this is a test sentence."

# 分词
inputs = tokenizer(input_sentence, return_tensors="pt").to(device)

# 用于存储时间的字典
layer_times_prefill = {}
layer_times_decode = {}

class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()
        elapsed_time = (self.end_time - self.start_time) * 1000  # 转换为毫秒
        return elapsed_time

timers_prefill = {}
timers_decode = {}

def pre_hook(module, input, phase):
    layer_name = module.__class__.__name__
    if phase == "prefill":
        if layer_name not in timers_prefill:
            timers_prefill[layer_name] = []
        timer = Timer()
        timers_prefill[layer_name].append(timer)
        timer.start()
    elif phase == "decode":
        if layer_name not in timers_decode:
            timers_decode[layer_name] = []
        timer = Timer()
        timers_decode[layer_name].append(timer)
        timer.start()

def post_hook(module, input, output, phase):
    layer_name = module.__class__.__name__
    if phase == "prefill":
        timer = timers_prefill[layer_name][-1]
        elapsed_time = timer.stop()
        if layer_name in layer_times_prefill:
            layer_times_prefill[layer_name].append(elapsed_time)
        else:
            layer_times_prefill[layer_name] = [elapsed_time]
    elif phase == "decode":
        timer = timers_decode[layer_name][-1]
        elapsed_time = timer.stop()
        if layer_name in layer_times_decode:
            layer_times_decode[layer_name].append(elapsed_time)
        else:
            layer_times_decode[layer_name] = [elapsed_time]

# 为模型中的所有层注册钩子
for name, module in model.named_modules():
    module.register_forward_pre_hook(lambda module, input: pre_hook(module, input, "prefill"))
    module.register_forward_hook(lambda module, input, output: post_hook(module, input, output, "prefill"))

# 执行prefill阶段
prefill_start_time = time.time()
with torch.no_grad():
    input_ids = inputs['input_ids']
    outputs = model.generate(input_ids, max_new_tokens=1, return_dict_in_generate=True, output_scores=True)
prefill_end_time = time.time()

# 重新注册钩子用于decode阶段
for name, module in model.named_modules():
    module.register_forward_pre_hook(lambda module, input: pre_hook(module, input, "decode"))
    module.register_forward_hook(lambda module, input, output: post_hook(module, input, output, "decode"))

# 执行decode阶段
decode_start_time = time.time()
with torch.no_grad():
    next_input_ids = outputs['sequences']  # 使用上一步生成的第一个token作为输入
    outputs = model.generate(next_input_ids, max_new_tokens=49, return_dict_in_generate=True, output_scores=True)
decode_end_time = time.time()

print(f"Prefill time: {prefill_end_time - prefill_start_time:.6f} s")
print(f"Decode time: {decode_end_time - decode_start_time:.6f} s")

# 打印每个层的推理时间（Prefill阶段）
print("\nPrefill Phase Layer Times:")
for layer, times in layer_times_prefill.items():
    avg_time = sum(times) / len(times)
    print(f"Layer: {layer}, Average Time taken per inference: {avg_time:.6f} ms")

# 打印每个层的推理时间（Decode阶段）
print("\nDecode Phase Layer Times:")
for layer, times in layer_times_decode.items():
    avg_time = sum(times) / len(times)
    print(f"Layer: {layer}, Average Time taken per inference: {avg_time:.6f} ms")
