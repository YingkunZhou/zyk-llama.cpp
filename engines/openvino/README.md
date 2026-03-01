https://claude.ai/share/42dda723-a465-48ab-b949-7ee292bcb201

# prepare the model (convert if necessary)
```bash
conda create -yn ov python=3.12
conda activate ov
git clone https://github.com/openvinotoolkit/openvino.genai.git --depth 1
cd openvino.genai/tools/llm_bench
pip install -r requirements.txt
# 其实各种共享链接库都能够在对应的安装目录中找到，这样就不需要下面的编译了
convert_tokenizer --with-detokenizer ~/gemma-3-4b-it-qat-q4_0-ov --output ~/gemma-3-4b-it-qat-q4_0-ov
```

# build openvino & openvino.genai

```bash
git clone --recursive https://github.com/openvinotoolkit/openvino.git
cd openvino
git submodule update --init --recursive

mkdir build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/openvino \
  -DENABLE_PYTHON=ON \
  -DENABLE_TESTS=OFF \
  -DENABLE_SAMPLES=OFF \
  -GNinja

ninja -j$(nproc)
ninja install
```

```bash
unset CC CXX
#  733 | #error Eigen requires at least c++14 support.
git clone --recursive https://github.com/openvinotoolkit/openvino.genai.git
cd openvino.genai

mkdir build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/openvino_genai \
  -DOpenVINO_DIR=/opt/openvino/runtime/cmake \
  -GNinja

ninja -j$(nproc)
ninja install
```

# build the benchmark
```bash
mkdir build && cd build
cmake .. -DOV_DIR=$OV_DIR -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

# run benchmark

```bash
./ov_bench -m ~/gemma-3-4b-it-qat-q4_0-ov -d CPU --pp 2 3 4 5 6 7 8 --tg 32 --warmup 1 --iters 3 --mode separate --threads 4
./ov_bench -m ~/gemma-3-4b-it-qat-q4_0-ov -d GPU --pp 2 3 4 5 6 7 8 --tg 32 --warmup 1 --iters 3 --mode separate
```