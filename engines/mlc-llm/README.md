# reference

- https://llm.mlc.ai/docs/install/mlc_llm.html#install-mlc-packages
- https://claude.ai/share/e34824ad-d3d7-41c5-9f43-058f37fa688b

# how to run

```python
import huggingface_hub as hf_hub

model_id = "mlc-ai/gemma-3-4b-it-q4f32_1-MLC"
model_path = "gemma-3-4b-it-q4f32_1-MLC"

hf_hub.snapshot_download(model_id, local_dir=model_path)
```

```bash
conda activate your-environment
python -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cpu mlc-ai-nightly-cpu
# only download cpu version, tvm will automatically choose the best vulkan/opencl backend
python -c "import mlc_llm; print(mlc_llm)"
# Prints out: <module 'mlc_llm' from '/path-to-env/lib/python3.13/site-packages/mlc_llm/__init__.py'>
python mlc_bench.py --hf-tokenizer google/gemma-3-4b-it --pp 1 2 3 4 5 6 7 8 --tg 32
python mlc_bench.py --pp 1 2 3 4 5 6 7 8 --tg 32
# python mlc_bench.py --hf-tokenizer google/gemma-3-4b-it --pp 1 8 32 64 128 --tg 32
```

<details>
<summary>if you want to build by yourself</summary>

```bash
# clone from GitHub
git clone --recursive https://github.com/apache/tvm.git && cd tvm
# create the build directory
rm -rf build && mkdir build && cd build
# specify build requirements in `config.cmake`
cp ../cmake/config.cmake .
export CC=clang-21
export CXX=clang++-21
sed -i 's|llvm-config --ignore-libllvm|llvm-config-21 --ignore-libllvm|' config.cmake

```

```bash
pip install numpy
pip install psutil
pip install shortuuid
pip install fastapi
```

```bash
cd <you path prefix>/tvm/3rdparty/tvm-ffi
pip install -e . --no-deps
```

```bash
# clone from GitHub
git clone --recursive https://github.com/mlc-ai/mlc-llm.git && cd mlc-llm/
# create build directory
mkdir -p build && cd build
# generate build configuration
python ../cmake/gen_cmake_config.py
# build mlc_llm libraries
cmake .. && make -j $(nproc) && cd ..
```

</details>
