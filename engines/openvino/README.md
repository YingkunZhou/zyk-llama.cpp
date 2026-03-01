
```bash
mkdir build && cd build
cmake .. -DOV_DIR=$OV_DIR -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./ov_bench -m ~/gemma-3-4b-it-qat-q4_0-ov -d CPU --pp 1 2 3 4 5 6 7 8 --tg 32 --warmup 1 --iters 3 --threads 4 --mode separate
```