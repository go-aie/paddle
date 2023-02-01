# paddle

[![Go Reference](https://pkg.go.dev/badge/go-aie/paddle/vulndb.svg)][2]

Go Inference Engine for [PaddlePaddle][1].


## Installation

1. Install [Paddle Inference Go API][3]
2. Install `paddle`

    ```bash
    $ go get -u github.com/go-aie/paddle
    ```


## Documentation

Check out the [documentation][2].


## Known Issues

### BLAS error

When using a high `maxConcurrency` (e.g. running benchmarks), sometimes you will get a BLAS error:

```
BLAS : Program is Terminated. Because you tried to allocate too many memory regions.
```

#### Solution

Set OpenBLAS to use a single thread:

```bash
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

See also:
- https://github.com/autogluon/autogluon/issues/1020
- https://groups.google.com/g/cmu-openface/c/CwVFyKJPWP4


## License

[MIT](LICENSE)


[1]: https://github.com/PaddlePaddle/Paddle
[2]: https://pkg.go.dev/github.com/go-aie/paddle
[3]: https://www.paddlepaddle.org.cn/inference/master/guides/install/go_install.html
