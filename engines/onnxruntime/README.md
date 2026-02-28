# 找到你的 onnxruntime-genai 源码/安装目录（有 src/ort_genai.h 的那个）
OGA_DIR=<path of onnxruntime-genai>

mkdir build && cd build
cmake .. -DOGA_DIR=$OGA_DIR -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 运行 benchmark
./model_bench -m ~/gemma-3-4b-onnx -e cpu --bench --pp 8 128 512 --tg 128 --reps 5

# 正常推理
./model_bench -m ~/gemma-3-4b-onnx -e cpu -p "Hello, who are you?"