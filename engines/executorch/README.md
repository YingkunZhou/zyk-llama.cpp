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
cmake --workflow --preset gemma3-cpu
cd -
```

```bash
./cmake-out/examples/models/gemma3/gemma3_e2e_runner \
  --model_path ~/gemma-3-4b-it-HQQ-INT8-INT4/model.pte \
  --tokenizer_path ~/gemma-3-4b-it-HQQ-INT8-INT4/tokenizer.json \
  --benchmark --pp "2,3,4,5,6,7,8" --tg "32" --n_runs 3 --cpu_threads 4
```
