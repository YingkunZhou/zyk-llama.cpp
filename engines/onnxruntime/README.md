https://claude.ai/share/ffe2d805-63c8-4822-9f09-db3ac1da5498

```bash
export CC=clang-21
export CXX=clang++-21
```

## build onnxruntime

```bash
git clone https://github.com/microsoft/onnxruntime.git --depth=1
cd onnxruntime
#git submodule sync
git submodule update --init --recursive
./build.sh --config Release --use_xnnpack --build_shared_lib --parallel --compile_no_warning_as_error --skip_tests --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF #onnxruntime_USE_AVX=ON onnxruntime_USE_AVX2=ON onnxruntime_USE_AVX512=ON
cd build/Linux/Release
DESTDIR=../onnxruntime make install -j`nproc`
cd ../onnxruntime
mv usr/local/include/onnxruntime/ include
mv usr/local/lib .
rm -rf usr
ORT_DIR=$PWD

CMake Warning at CMakeLists.txt:532 (message):
  onnxruntime_USE_SVE was set but it is not supported on this platform.  It
  will be disabled.
CMake Warning at CMakeLists.txt:543 (message):
  KleidiAI is not supported on this platform.
Call Stack (most recent call first):
  CMakeLists.txt:551 (is_kleidiai_supported)
CMake Warning at CMakeLists.txt:554 (message):
  onnxruntime_USE_KLEIDIAI was set but it is not supported.  It will be
  disabled.
```

### X86-64

```diff
diff --git a/onnxruntime/core/common/spin_pause.cc b/onnxruntime/core/common/spin_pause.cc
index 329f3f1..fc6506d 100644
--- a/onnxruntime/core/common/spin_pause.cc
+++ b/onnxruntime/core/common/spin_pause.cc
@@ -19,6 +19,17 @@
 #endif
 #endif

+#if defined(__clang__)
+  #define TPAUSE(ctrl, deadline) \
+      __builtin_ia32_tpause((ctrl), \
+          static_cast<uint32_t>((deadline) >> 32), \
+          static_cast<uint32_t>(deadline))
+#else
+  // GCC
+  #define TPAUSE(ctrl, deadline) \
+      __builtin_ia32_tpause((ctrl), (deadline))
+#endif
+
 namespace onnxruntime {
 namespace concurrency {

@@ -35,7 +46,8 @@ void SpinPause() {
 #if defined(_WIN32)
     _tpause(0x0, __rdtsc() + tpause_spin_delay_cycles);
 #elif defined(__linux__)
-    __builtin_ia32_tpause(0x0, __rdtsc() + tpause_spin_delay_cycles);
+    // __builtin_ia32_tpause(0x0, __rdtsc() + tpause_spin_delay_cycles);
+    TPAUSE(0x0, __rdtsc() + tpause_spin_delay_cycles);
 #else
     _mm_pause();
 #endif
```

### AARCH64 Linux

```bash
./build.sh --config Release \
  --use_xnnpack \
  --build_shared_lib \
  --parallel \
  --compile_no_warning_as_error \
  --skip_tests \
  --cmake_extra_defines \
    onnxruntime_BUILD_UNIT_TESTS=OFF \
    CMAKE_SYSTEM_NAME=Linux \
    CMAKE_SYSTEM_PROCESSOR=aarch64 \
    onnxruntime_USE_SVE=OFF \
    onnxruntime_USE_KLEIDIAI=OFF
```

### AARCH64 Android

```bash
export BASEDIR=$PWD
export ANDROID_NDK=$BASEDIR/android-ndk-r29
export ANDROID_SDK=$BASEDIR/android-sdk
cd onnxruntime
./build.sh --use_nnapi --use_xnnpack --config Release --android --android_sdk_path $ANDROID_SDK --android_ndk_path $ANDROID_NDK --android_abi arm64-v8a --android_api 30 --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --skip_tests
DESTDIR=../onnxruntime make install -j`nproc`
cd ../onnxruntime
mv usr/local/include/onnxruntime/ include
mv usr/local/lib .
rm -rf usr
cd ..
tar czf onnxruntime.tar.gz onnxruntime
```

## build onnxruntime-genai

```bash
# git clone https://github.com/microsoft/onnxruntime-genai --depth 1
wget https://github.com/microsoft/onnxruntime-genai/archive/refs/tags/v0.12.0.tar.gz
tar xf v0.12.0.tar.gz
cd onnxruntime-genai-0.12.0
# wget https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-linux-aarch64-1.24.2.tgz
# tar xf onnxruntime-linux-aarch64-1.24.2.tgz
# ORT_DIR=$PWD/onnxruntime-linux-aarch64-1.24.2
# wget https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-linux-x64-1.24.2.tgz
# tar xf onnxruntime-linux-x64-1.24.2.tgz
# ORT_DIR=$PWD/onnxruntime-linux-x64-1.24.2
python build.py --config Release --ort_home $ORT_DIR --skip_wheel --skip_examples
cd build/Linux/Release
DESTDIR=../onnxruntime make install -j`nproc`
cd ../onnxruntime
mv usr/local/include .
mv usr/local/lib .
rm -rf usr
OGA_DIR=$PWD
cd lib
ln -sf $ORT_DIR/lib/libonnxruntime.so.1.25.0 libonnxruntime.so
```

### AARCH64 Android (Termux)

```bash
tar xf onnxruntime.tar.gz
cd onnxruntime
ln -sf include headers
ln -sf lib jni
ORT_DIR=$PWD
cd ..
tar xf v0.12.0.tar.gz
cd onnxruntime-genai-0.12.0
python build.py --config Release --ort_home $ORT_DIR --skip_wheel --skip_examples --cmake_extra_defines CMAKE_ANDROID_ARCH_ABI=arm64-v8a
cd build/Linux/Release
DESTDIR=../onnxruntime make install -j`nproc`
cd ../onnxruntime
mv usr/local/include .
mv usr/local/lib .
rm -rf usr
OGA_DIR=$PWD
cd lib
ln -sf $ORT_DIR/lib/libonnxruntime.so libonnxruntime.so
```

## use prebuilt (only-x86)

```bash
wget https://github.com/microsoft/onnxruntime-genai/releases/download/v0.12.0/onnxruntime-genai-0.12.0-linux-x64.tar.gz
tar xf onnxruntime-genai-0.12.0-linux-x64.tar.gz
cd onnxruntime-genai-0.12.0-linux-x64
cp onnxruntime-linux-x64-1.24.2/lib/libonnxruntime.so.1.24.2 lib/libonnxruntime.so
```

# 编译 benchmark tool

```bash
mkdir build && cd build
cmake .. -DOGA_DIR=$OGA_DIR -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

```json
$ head -n 20 ~/gemma-3-4b-onnx/genai_config.json
{
    "model": {
        "bos_token_id": 2,
        "context_length": 131072,
        "decoder": {
            "session_options": {
                "log_id": "onnxruntime-genai",
                "provider_options": [],
                "intra_op_num_threads": 4,
                "inter_op_num_threads": 1
            },
            "filename": "gemma-3-text.onnx",
```

# 运行 benchmark
./model_bench -m ~/gemma-3-4b-onnx -e cpu --bench --pp 8 128 512 --tg 128 --reps 5

# 正常推理
./model_bench -m ~/gemma-3-4b-onnx -e cpu -p "Hello, who are you?"

