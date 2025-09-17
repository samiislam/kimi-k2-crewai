from llama_cpp import Llama

# Point to the first GGUF shard
model_path = r".\models\Kimi-K2-Instruct-Q2_K-00001-of-00008.gguf"

# Initialize with tiny context to avoid heavy memory usage
llm = Llama(model_path=model_path, n_ctx=1, n_gpu_layers=-1)

# Print all available metadata
print("=== GGUF Metadata ===")
for k, v in llm.metadata.items():
    print(f"{k}: {v}")

# Try to get parameter count if present in metadata
if "general.parameter_count" in llm.metadata:
    params = int(llm.metadata["general.parameter_count"])
    print(f"\n✅ Model has {params:,} parameters")
else:
    print("\n⚠️ Parameter count not found directly in metadata.")
