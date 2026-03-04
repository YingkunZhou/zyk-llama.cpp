https://claude.ai/share/ea20b682-11db-46f5-9c6e-8abac2ac5f1d

```bash
import huggingface_hub as hf_hub
model_id = "pytorch/gemma-3-4b-it-HQQ-INT8-INT4"
model_path = "gemma-3-4b-it-HQQ-INT8-INT4"
hf_hub.snapshot_download(model_id, local_dir=model_path)
```

```bash
git clone https://github.com/pytorch/executorch.git
cd executorch
git submodule update --init --recursive
# python install_executorch.py
```

# build on Linux

```bash
conda install cmake
cd executorch
cmake --workflow llm-release
```

```diff
diff --git a/include/flatcc/portable/grisu3_print.h b/include/flatcc/portable/grisu3_print.h
index d748408..4f71985 100644
--- a/include/flatcc/portable/grisu3_print.h
+++ b/include/flatcc/portable/grisu3_print.h
@@ -183,7 +183,7 @@ static int grisu3_i_to_str(int val, char *str)

 static int grisu3_print_nan(uint64_t v, char *dst)
 {
-    static char hexdigits[16] = "0123456789ABCDEF";
+    static char hexdigits[17] = "0123456789ABCDEF";
     int i = 0;

     dst[0] = 'N';
diff --git a/include/flatcc/portable/pprintint.h b/include/flatcc/portable/pprintint.h
index d05f376..d6d954c 100644
--- a/include/flatcc/portable/pprintint.h
+++ b/include/flatcc/portable/pprintint.h
@@ -385,7 +385,7 @@ static int print_int8(int8_t n, char *p)

     if ((sign = n < 0)) {
         *p++ = '-';
-        n = -n;
+        n = (int8_t)-n;
     }
     return print_uint8((uint8_t)n, p) + sign;
 }
@@ -396,7 +396,7 @@ static int print_int16(int16_t n, char *p)

     if ((sign = n < 0)) {
         *p++ = '-';
-        n = -n;
+        n = (int16_t)-n;
     }
     return print_uint16((uint16_t)n, p) + sign;
 }
```

```bash
cd examples/models/gemma3
cp <root of my llama.cpp>/engines/executorch/e2e_runner.cpp .
cmake --workflow --preset gemma3-cpu
cd -
```

```bash
./cmake-out/examples/models/gemma3/gemma3_e2e_runner \
  --model_path ~/gemma-3-4b-it-HQQ-INT8-INT4/model.pte \
  --tokenizer_path ~/gemma-3-4b-it-HQQ-INT8-INT4/tokenizer.json \
  --benchmark --pp "2,3,4,5,6,7,8" --tg "32" --n_runs 3 --cpu_threads 4
```

# cross-compile for Android

```bash
# on x86 host
cd executorch

cmake --preset android-arm64-v8a \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-30 \
  -DEXECUTORCH_BUILD_XNNPACK=ON \
  -DEXECUTORCH_BUILD_EXTENSION_LLM=ON \
  -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=${PWD}/cmake-out-android-arm64-v8a

CMAKE_BUILD_PARALLEL_LEVEL=16 cmake --build cmake-out-android-arm64-v8a --target install

# replaced with modified e2e_runner.cpp and CMakePresets.json
cd examples/models/gemma3
cmake --workflow --preset gemma3-android-arm64
cd -
tar czf cmake-out-android-arm64-v8a.tar.gz cmake-out-android-arm64-v8a
```

```bash
# on android device termux
scp <x86-host>:~/executorch/cmake-out-android-arm64-v8a.tar.gz .
tar xf cmake-out-android-arm64-v8a.tar.gz
taskset -c 4-7 ./cmake-out-android-arm64-v8a/examples/models/gemma3/gemma3_e2e_runner \
  --model_path ~/work/gemma-3-4b-it-HQQ-INT8-INT4/model.pte \
  --tokenizer_path ~/work/gemma-3-4b-it-HQQ-INT8-INT4/tokenizer.json \
  --benchmark --pp "2,3,4,5,6,7,8" --tg "32" --n_runs 3 --cpu_threads 4
```

<details>
<summary>用下面两个编译不同ISA的和前面编译的没什么区别</summary>

```bash
cmake --preset android-arm64-v8a \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-30 \
  -DCMAKE_CXX_FLAGS="-march=armv8.2-a+dotprod" \
  -DCMAKE_C_FLAGS="-march=armv8.2-a+dotprod" \
  -DEXECUTORCH_BUILD_XNNPACK=ON \
  -DEXECUTORCH_BUILD_EXTENSION_LLM=ON \
  -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=${PWD}/cmake-out-android-arm64-v8a

cmake --preset android-arm64-v8a \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-30 \
  -DCMAKE_CXX_FLAGS="-march=armv8.6-a+i8mm" \
  -DCMAKE_C_FLAGS="-march=armv8.6-a+i8mm" \
  -DEXECUTORCH_BUILD_XNNPACK=ON \
  -DEXECUTORCH_BUILD_EXTENSION_LLM=ON \
  -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_ANDROID=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=${PWD}/cmake-out-android-arm64-v8a
```

</details>
