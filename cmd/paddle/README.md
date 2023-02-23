# paddle

The [official guide][1] provides step-by-step instructions to install Paddle Inference Go API, but obviously it's tedious to do it manually.

`paddle` is a CLI tool that does the hard work to give you "one-click installation".


## Installation

```bash
$ go install github.com/go-aie/paddle/cmd/paddle@latest
```

<details open>
  <summary> Usage </summary>

```bash
$ paddle install -h
Usage: paddle install <url>

Install Paddle Inference Go API.

Arguments:
  <url>    The URL of the compressed Paddle Inference C Library.

Flags:
  -h, --help          Show context-sensitive help.

  -o, --out=STRING    The output directory.
  -f, --force         Force to download Paddle Inference C Library (even if it already exists).
  -v, --verbose       Print the progress messages.
```
</details>


## Install Paddle Inference Go API

```bash
$ cd <your-project-root-path>
$ paddle install -v <url-of-paddle-inference-c-lib>
```

Note:

- `<your-project-root-path>`: The root path of your local project.
- `<url-of-paddle-inference-c-lib>`: Check out [下载安装 Linux 推理库][2] to select the C library you need.
- Since `paddle install` will modify the read-only Go module cache, most of the time, you need the sudo privilege to run the command (i.e. `sudo paddle install ...`).

Take Mac M1 as an example, we should select the C library [labeled as "m1"][3] and run:

```bash
$ cd example-project-on-mac-m1
$ paddle install -v https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/MacOS/m1_clang_noavx_openblas/paddle_inference_c_install_dir.tgz
```


[1]: https://www.paddlepaddle.org.cn/inference/master/guides/install/go_install.html
[2]: https://www.paddlepaddle.org.cn/inference/master/guides/install/download_lib.html
[3]: https://www.paddlepaddle.org.cn/inference/master/guides/install/download_lib.html#id5
