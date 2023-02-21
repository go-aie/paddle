package paddle

import (
	"fmt"

	"github.com/go-aie/xslices"
	paddle "github.com/paddlepaddle/paddle/paddle/fluid/inference/goapi"
)

// Engine is an inference engine.
type Engine struct {
	predictorPool *PredictorPool
}

func NewEngine(model, params string, maxConcurrency int) *Engine {
	config := paddle.NewConfig()
	config.SetModel(model, params)
	config.EnableMemoryOptim(true) // Enable the memory optimization

	return &Engine{
		predictorPool: NewPredictorPool(config, maxConcurrency),
	}
}

func (e *Engine) Infer(inputs []Tensor) (outputs []Tensor) {
	predictor, put := e.predictorPool.Get()
	defer put()

	inputNames := predictor.GetInputNames()
	if len(inputs) != len(inputNames) {
		panic(fmt.Errorf("inputs mismatch the length of %v", inputNames))
	}

	// Set the inference input.
	for i, name := range inputNames {
		inputHandle := predictor.GetInputHandle(name)
		inputHandle.Reshape(inputs[i].Shape)
		inputHandle.CopyFromCpu(inputs[i].Data)
	}

	// Run the inference engine.
	predictor.Run()

	// Get the inference output.
	for _, name := range predictor.GetOutputNames() {
		outputHandle := predictor.GetOutputHandle(name)
		outputs = append(outputs, e.getOutputTensor(outputHandle))
	}

	// Clear all temporary tensors to release the memory.
	//
	// See also:
	// - https://github.com/PaddlePaddle/Paddle/issues/43346
	// - https://github.com/PaddlePaddle/PaddleOCR/discussions/6977
	predictor.ClearIntermediateTensor()
	predictor.TryShrinkMemory()

	return
}

func (e *Engine) getOutputTensor(handle *paddle.Tensor) Tensor {
	var data interface{}
	shape := handle.Shape()
	length := numElements(shape)

	switch dataType := handle.Type(); dataType {
	case paddle.Float32:
		data = make([]float32, length)
	case paddle.Int32:
		data = make([]int32, length)
	case paddle.Int64:
		data = make([]int64, length)
	case paddle.Uint8:
		data = make([]uint8, length)
	case paddle.Int8:
		data = make([]int8, length)
	default:
		panic(fmt.Errorf("unknown output data type %T", dataType))
	}

	handle.CopyToCpu(data)

	return Tensor{
		Shape: shape,
		Data:  data,
	}
}

type Tensor struct {
	Shape []int32
	Data  interface{}
}

func NewTensorFromOneDimSlice[E any](slice []E) Tensor {
	if len(slice) == 0 {
		return Tensor{}
	}
	return Tensor{
		Shape: []int32{int32(len(slice))},
		Data:  slice,
	}
}

func NewTensorFromTwoDimSlice[E any](slice [][]E) Tensor {
	if len(slice) == 0 {
		return Tensor{}
	}

	var flattened []E
	for _, batch := range slice {
		flattened = append(flattened, batch...)
	}

	batchSize, dataSize := len(slice), len(slice[0])
	return Tensor{
		Shape: []int32{int32(batchSize), int32(dataSize)},
		Data:  flattened,
	}
}

func NewTensorFromThreeDimSlice[E any](slice [][][]E) Tensor {
	if len(slice) == 0 {
		return Tensor{}
	}

	var flattened []E
	for _, batch := range slice {
		for _, d1 := range batch {
			flattened = append(flattened, d1...)
		}
	}

	batchSize, dataSize1 := len(slice), len(slice[0])
	var dataSize2 int
	if dataSize1 > 0 {
		dataSize2 = len(slice[0][0])
	}

	return Tensor{
		Shape: []int32{int32(batchSize), int32(dataSize1), int32(dataSize2)},
		Data:  flattened,
	}
}

func NewTensorFromFourDimSlice[E any](slice [][][][]E) Tensor {
	if len(slice) == 0 {
		return Tensor{}
	}

	var flattened []E
	for _, batch := range slice {
		for _, d1 := range batch {
			for _, d2 := range d1 {
				flattened = append(flattened, d2...)
			}
		}
	}

	batchSize, dataSize1 := len(slice), len(slice[0])
	var dataSize2, dataSize3 int
	if dataSize1 > 0 {
		dataSize2 = len(slice[0][0])
		if dataSize2 > 0 {
			dataSize3 = len(slice[0][0][0])
		}
	}

	return Tensor{
		Shape: []int32{int32(batchSize), int32(dataSize1), int32(dataSize2), int32(dataSize3)},
		Data:  flattened,
	}
}

func numElements(shape []int32) int32 {
	n := int32(1)
	for _, v := range shape {
		n *= v
	}
	return n
}

type TypedTensor[E xslices.Number] struct {
	Shape []int32
	Data  []E
}

func NewTypedTensor[E xslices.Number](t Tensor) TypedTensor[E] {
	var data []E
	switch v := t.Data.(type) {
	case []float32:
		for _, d := range v {
			data = append(data, E(d))
		}
	case []int32:
		for _, d := range v {
			data = append(data, E(d))
		}
	case []int64:
		for _, d := range v {
			data = append(data, E(d))
		}
	case []uint8:
		for _, d := range v {
			data = append(data, E(d))
		}
	case []int8:
		for _, d := range v {
			data = append(data, E(d))
		}
	}
	return TypedTensor[E]{
		Shape: t.Shape,
		Data:  data,
	}
}
