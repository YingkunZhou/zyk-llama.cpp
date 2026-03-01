# MNN

MNN bench is very simple, just use official C/C++ built-binary tools

- https://mnn-docs.readthedocs.io/en/latest/transformers/llm.html
- https://huggingface.co/taobao-mnn/gemma-3-4b-it-q4_0-mnn

```bash
export CC=clang-21
export CXX=clang++-21
git clone https://github.com/alibaba/MNN.git --depth 1
mkdir build && cd build
cmake .. -DMNN_LOW_MEMORY=true -DMNN_CPU_WEIGHT_DEQUANT_GEMM=true -DMNN_BUILD_LLM=true -DMNN_SUPPORT_TRANSFORMER_FUSE=true

#X86 laptop (e.g. hx365 zen5)：可添加 -DMNN_AVX512=true 以利用 AVX512 指令集加速。
#Android (e.g. 8gen2/orangepi5b)：可添加 -DMNN_OPENCL=true 以利用 GPU 加速。
#MacOS (e.g. M1 Mini)：可添加 -DMNN_METAL=ON 以利用 GPU 加速。

make -j$(nproc)
./llm_bench -m ~/gemma-3-4b-it-qat-q4_0-mnn/config.json -a cpu -t 4 -p 1,2,3,4,5,6,7,8 -n 32 -rep 3
./llm_bench -m ~/gemma-3-4b-it-qat-q4_0-mnn/config.json -a opencl -t 4 -p 1,2,3,4,5,6,7,8 -n 32 -rep 3
./llm_bench -m ~/gemma-3-4b-it-qat-q4_0-mnn/config.json -a metal -t 4 -p 1,2,3,4,5,6,7,8 -n 32 -rep 3
```
