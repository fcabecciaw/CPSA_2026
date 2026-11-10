# Vitis AI 3.5 VART on Kria KV260 (Ubuntu 22.04)

Reproducible native build of the Vitis AI 3.5 runtime (unilog, xir,
target_factory, vart) plus `show_dpu` and a query/status-capable `xdputil`
on the KV260 board.

Verified on 2026-11-06 with:

| Item | Value |
|---|---|
| Board | Kria KV260, aarch64, 4 cores, 3.8 GiB RAM |
| OS | Ubuntu 22.04.5 LTS |
| Kernel | 5.15.0-1080-xilinx-zynqmp |
| XRT | `xrt 2.13.479-0ubuntu2` (headers `/usr/include/xrt`, libs `/lib/libxrt_*`) |
| DPU app | `kv260-benchmark-b4096`, active slot 0 |
| Expected fingerprint | `0x101000016010407` (`DPUCZDX8G_ISA1_B4096`) |
| Install prefix | `/usr/local` |
| Source tree | `/home/kria/vart_build/Vitis-AI` |
| Build trees/logs | `/home/kria/vart_build/build`, `/home/kria/vart_build/*.log` |

The pre-existing Ubuntu PPA packages `vitis-ai-runtime 2.0` /
`vitis-ai-library 2.0` were already removed before this build and were not
reinstalled. They are not compatible with ISA1 and segfault at load time.

## 1. Source ref

Upstream repository: <https://github.com/Xilinx/Vitis-AI>

```sh
mkdir -p ~/vart_build && cd ~/vart_build
git clone --depth 1 --branch v3.5 --filter=blob:none --sparse \
  https://github.com/Xilinx/Vitis-AI.git Vitis-AI
cd Vitis-AI
git sparse-checkout set src/vai_runtime src/vai_library/usefultools
git rev-parse HEAD
# 7a0d5a695027b0dc1494c7299918959608c3fab2
git describe --tags --exact-match
# v3.5
```

Only `src/vai_runtime` and `src/vai_library/usefultools` are needed. The
checked-out tree is ~9 MiB.

## 2. Build dependencies

Installed with apt, no `apt upgrade`, no kernel/firmware changes:

```sh
sudo apt-get install -y --no-install-recommends \
  cmake protobuf-compiler libgoogle-glog-dev libjson-c-dev \
  libeigen3-dev libssl-dev pkg-config python3-dev pybind11-dev libpython3-dev
```

Versions observed: cmake 3.22.1, g++ 11.4.0, protoc 3.12.4, pybind11 2.9.1,
Python 3.10.12. `libboost-all-dev` was already present.

No swap was added. Builds use `-j2` to stay within 3.9 GiB RAM.

## 3. Build and install

All modules are built natively for aarch64 in `Release` mode with install
prefix `/usr/local`. `-s` strips the installed shared objects.

### 3.1 unilog

```sh
cd ~/vart_build/Vitis-AI/src/vai_runtime/unilog
cmake -S . -B ~/vart_build/build/unilog \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_TEST=OFF \
  -DBUILD_PYTHON=OFF \
  -DCMAKE_CXX_FLAGS_RELEASE="-O3 -s"
cmake --build ~/vart_build/build/unilog -j2
sudo cmake --install ~/vart_build/build/unilog
```

`BUILD_PYTHON=OFF` only skips the optional `py_unilog` module; the required
`xir` and `vart` Python bindings are built below.

### 3.2 xir (with Python binding)

```sh
cd ~/vart_build/Vitis-AI/src/vai_runtime/xir
cmake -S . -B ~/vart_build/build/xir \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_TEST=OFF \
  -DBUILD_CONTRIB=OFF \
  -DBUILD_DOC=OFF \
  -DBUILD_PYTHON=ON \
  -DPYTHON_SITE_PACKAGES=lib/python3.10/dist-packages \
  -DCMAKE_CXX_FLAGS_RELEASE="-O3 -s"
cmake --build ~/vart_build/build/xir -j2
sudo cmake --install ~/vart_build/build/xir
```

`PYTHON_SITE_PACKAGES` makes the binding install to
`/usr/local/lib/python3.10/dist-packages`, which is already on Python's
default `sys.path`.

### 3.3 target_factory

```sh
cd ~/vart_build/Vitis-AI/src/vai_runtime/target_factory
cmake -S . -B ~/vart_build/build/target_factory \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_TEST=OFF \
  -DCMAKE_CXX_FLAGS_RELEASE="-O3 -s"
cmake --build ~/vart_build/build/target_factory -j2
sudo cmake --install ~/vart_build/build/target_factory
```

### 3.4 XRT 2.13 compatibility patch (required)

XRT 2.13 removed the `xclIPSetReadRange()` API. Vitis AI 3.5 calls it
unconditionally in `vart/xrt-device-handle`, so the vart build fails with:

```text
xrt_device_handle_imp.cpp:324:11: error: 'xclIPSetReadRange' was not declared in this scope
```

`vart/dpu-controller` already guards this call with its own
`HAS_xclIPSetReadRange` CMake check and needs no patch. Patch only the
unconditional call:

```sh
cd ~/vart_build/Vitis-AI/src/vai_runtime/vart/xrt-device-handle
cp src/xrt_device_handle_imp.cpp src/xrt_device_handle_imp.cpp.orig
sed -i \
  's|      r = xclIPSetReadRange(x.handle, x.cu_index, 0x10, 0x1F0);|      // XRT 2.13 removed xclIPSetReadRange(); the API is no longer available.\n      r = 0;|' \
  src/xrt_device_handle_imp.cpp
```

This only affects the optional read-range restriction; register reads via
`xclRegRead()` continue to work.

### 3.5 vart (with Python binding)

```sh
cd ~/vart_build/Vitis-AI/src/vai_runtime/vart
cmake -S . -B ~/vart_build/build/vart \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DCMAKE_PREFIX_PATH=/usr/local \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_TEST=OFF \
  -DBUILD_PYTHON=ON \
  -DPYTHON_SITE_PACKAGES=lib/python3.10/dist-packages \
  -DENABLE_CPU_RUNNER=OFF \
  -DENABLE_SIM_RUNNER=OFF \
  -DCMAKE_CXX_FLAGS_RELEASE="-O3 -s"
cmake --build ~/vart_build/build/vart -j2
sudo cmake --install ~/vart_build/build/vart
sudo ldconfig
```

`show_dpu` is built and installed unconditionally by the
`dpu-controller` component (`/usr/local/bin/show_dpu`).

## 4. Installed layout

| Path | Contents |
|---|---|
| `/usr/local/lib/libunilog.so*` | unilog |
| `/usr/local/lib/libxir.so*` | xir |
| `/usr/local/lib/libtarget-factory.so*` | target_factory |
| `/usr/local/lib/libvart-*.so*` | vart components (runner, dpu-runner, dpu-controller, xrt-device-handle, util, ...) |
| `/usr/local/include/{UniLog,xir,vart,vitis}` | C++ headers |
| `/usr/local/share/cmake/{unilog,xir,target-factory,vart}` | CMake package configs |
| `/usr/local/lib/python3.10/dist-packages/{xir,vart}.cpython-310-aarch64-linux-gnu.so` | Python bindings |
| `/usr/local/bin/show_dpu` | DPU fingerprint tool |

`/usr/local/lib` is already in the default loader path on this image, so no
`/etc/ld.so.conf.d` entry or `LD_LIBRARY_PATH` is needed.

Install size: `/usr/local` is 9.4 MiB total. Source, build trees and logs in
`/home/kria/vart_build` use ~50 MiB. Root filesystem went from 7.5 G used to
7.7 G used (including apt packages).

## 5. xdputil

In Vitis AI 3.5, `xdputil` is **not** part of `src/vai_runtime`; it lives in
`src/vai_library/usefultools`. The Python front end is a thin wrapper around
the `tools_extra_ops` pybind11 module. Building the full upstream module
would require the whole `vitis-ai-library` (OpenCV, Eigen, graph runner,
performance test), which is out of scope for the runtime. Instead, the
upstream query/status sources were compiled into a small `tools_extra_ops`
module:

```sh
cd ~/vart_build
cat > tools_extra_ops_min.cpp <<'EOF'
// Minimal pybind11 module: exposes the upstream Vitis-AI 3.5 xdputil_query /
// xdputil_status entry points only (query/status build for KV260).
#include <pybind11/pybind11.h>

#include "tools_extra_ops.hpp"

namespace py = pybind11;

PYBIND11_MODULE(tools_extra_ops, m) {
  m.doc() = "Vitis-AI 3.5 xdputil query/status bindings (query/status build)";
  m.def("xdputil_query", &xdputil_query);
  m.def("xdputil_status", &xdputil_status);
}
EOF

SRC=~/vart_build/Vitis-AI/src

# Keep stdout valid JSON: the upstream AIE-frequency probe prints diagnostics
# with std::cout on this board. Route those messages to stderr instead.
sed -i -e 's/std::cout << "Err /std::cerr << "Err /' \
       -e 's/std::cout <<"Err/std::cerr <<"Err/' \
  $SRC/vai_library/usefultools/src/xdputil_query.cpp

g++ -std=c++17 -O2 -fPIC -shared -DENABLE_XRT \
  $(python3 -m pybind11 --includes) \
  -I$SRC/vai_library/usefultools/src \
  -I$SRC/vai_runtime/vart/xrt-device-handle/src \
  -I/usr/local/include -I/usr/include/xrt \
  $SRC/vai_library/usefultools/src/xdputil_query.cpp \
  $SRC/vai_library/usefultools/src/read_register.cpp \
  $SRC/vai_library/usefultools/src/xilinx_version.cpp \
  tools_extra_ops_min.cpp \
  -L/usr/local/lib -lxir -lvart-xrt-device-handle -ltarget-factory \
  -lunilog -lglog -lprotobuf -ldl -Wl,-rpath,/usr/local/lib \
  -o /usr/local/lib/python3.10/dist-packages/tools_extra_ops.cpython-310-aarch64-linux-gnu.so

# Upstream Python front end; only query/status components are installed.
SP=/usr/local/lib/python3.10/dist-packages
PY=$SRC/vai_library/usefultools/python
sudo install -m 0644 $PY/xdputil.py $SP/xdputil.py
sudo install -d $SP/xdputil_component
sudo install -m 0644 $PY/xdputil_component/__init__.py \
  $PY/xdputil_component/query.py $PY/xdputil_component/status.py \
  $SP/xdputil_component/
sudo install -m 0755 $PY/xdputil /usr/local/bin/xdputil
```

Available subcommands: `xdputil query` and `xdputil status`.
`xdputil run`, `benchmark`, `xmodel` etc. are intentionally absent because
they need the full `vitis-ai-library`, which was not built.

The optional AIE-frequency probe emits `Err open idcode file failure` on this
board (no AIE in this design). The patch above redirects that upstream
diagnostic to stderr, so `xdputil query` stdout is valid JSON. The message
does not affect the JSON result or the exit code.

## 6. Verification (observed)

### 6.1 show_dpu

```console
$ show_dpu
device_core_id=0 device= 0 core = 0 fingerprint = 0x101000016010407 batch = 1 full_cu_name=DPUCZDX8G:DPUCZDX8G_1
```

### 6.2 xdputil query

```console
$ xdputil query
{
    "DPU IP Spec":{
        "DPU Core Count":1,
        "IP version":"v4.0.0",
        "generation timestamp":"2022-05-14 14-30-00",
        "git commit id":"4772d51",
        "git commit time":2022051323,
        "regmap":"1to1 version"
    },
    "VAI Version":{
        "libvart-runner.so":"Xilinx vart-runner Version: 3.5.0-7a0d5a695027b0dc1494c7299918959608c3fab2  2026-11-06-14:38:35 ",
        "libvitis_ai_library-dpu_task.so":"libvitis_ai_library-dpu_task.so: cannot open shared object file: No such file or directory",
        "libxir.so":"Xilinx xir Version: xir-7a0d5a695027b0dc1494c7299918959608c3fab2 2026-11-06-14:13:57",
        "target_factory":"target-factory.3.5.0 7a0d5a695027b0dc1494c7299918959608c3fab2"
    },
    "kernels":[
        {
            "AIE Frequency (Hz)":0,
            "DPU Arch":"DPUCZDX8G_ISA1_B4096_0101000016010407",
            "DPU Frequency (MHz)":300,
            "IP Type":"DPU",
            "Load Parallel":2,
            "Load augmentation":"enable",
            "Load minus mean":"disable",
            "Save Parallel":2,
            "XRT Frequency (MHz)":300,
            "cu_addr":"0xa0010000",
            "cu_handle":"0xaaab0c77c460",
            "cu_idx":0,
            "cu_mask":1,
            "cu_name":"DPUCZDX8G:DPUCZDX8G_1",
            "device_id":0,
            "fingerprint":"0x101000016010407",
            "name":"DPU Core 0"
        }
    ]
}
```

```console
$ xdputil query > /tmp/q.json 2>/tmp/q.err
$ python3 -c 'import json; d=json.load(open("/tmp/q.json")); print(d["kernels"][0]["fingerprint"])'
0x101000016010407
```

The `libvitis_ai_library-dpu_task.so` entry is informational only; that
library is not part of the runtime build.

`xdputil status` also exits 0 and prints the DPU register map.

### 6.3 Python imports (new login shell, no extra env)

```console
$ bash -lc 'python3 -c "import xir, vart; print(\"ok\")"'
ok
$ bash -lc 'python3 -c "import vart, xir; print(\"ok\")"'
ok
```

### 6.4 C++ application build

A minimal program using the installed headers/libs builds and runs with g++
and no extra flags beyond the library paths:

```cpp
#include <iostream>
#include <vart/runner.hpp>
#include <vitis/ai/target_factory.hpp>
#include <xir/graph/graph.hpp>

int main() {
  auto target = vitis::ai::target_factory()->create(0x101000016010407ULL);
  std::cout << "target: " << target.name() << " type: " << target.type()
            << std::endl;
  return 0;
}
```

```sh
g++ -std=c++17 -O2 vart_cpp_test.cpp -I/usr/local/include -L/usr/local/lib \
  -lvart-runner -lxir -ltarget-factory -lglog -lunilog \
  -Wl,-rpath,/usr/local/lib -o vart_cpp_test
./vart_cpp_test
# target: DPUCZDX8G_ISA1_B4096_0101000016010407 type: DPUCZDX8G
```

### 6.5 Filesystem health

```console
$ findmnt -no OPTIONS /
rw,relatime,discard,errors=remount-ro
$ sudo dmesg | grep -c 'EXT4-fs error'
0
```

No I/O or EXT4 errors were observed after the build. (The only `dmesg` line
matching `errors` is the normal boot-time `EXT4-fs ... errors=remount-ro`
mount option.)

## 7. Notes and limitations

- The runtime is built for the native aarch64 board; no cross-compilation or
  PetaLinux SDK is involved.
- Python modules land in `/usr/local/lib/python3.10/dist-packages`, which is
  on the default `sys.path`; no `.pth` file or profile snippet is required.
- `show_dpu` and `xdputil query` require the `kv260-benchmark-b4096` app to
  be loaded. `/etc/vart.conf` points at
  `/lib/firmware/xilinx/kv260-benchmark-b4096/kv260-benchmark-b4096.xclbin`.
- Only `query` and `status` are available in `xdputil`; the remaining
  subcommands belong to the full `vitis-ai-library` and are not needed for
  runtime validation.
- To remove the runtime: delete the `/usr/local` files listed in section 4
  and `~/vart_build`, then run `sudo ldconfig`.
