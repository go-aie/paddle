package paddle_test

import (
	"testing"

	"github.com/go-aie/paddle"
	"github.com/google/go-cmp/cmp"
)

func TestMatrix_Norm(t *testing.T) {
	tests := []struct {
		inTensor paddle.Tensor
		wantData []float32
	}{
		{
			inTensor: paddle.Tensor{
				Shape: []int32{5, 1},
				Data:  []float32{1, 2, 3, 4, 5},
			},
			wantData: []float32{0.13483997, 0.26967994, 0.40451991, 0.53935989, 0.67419986},
		},
	}
	for _, tt := range tests {
		m := paddle.NewMatrixFromTensor[float32](tt.inTensor)
		gotData := m.Norm().RawData()
		if !cmp.Equal(gotData, tt.wantData) {
			diff := cmp.Diff(gotData, tt.wantData)
			t.Errorf("Want - Got: %s", diff)
		}
	}
}

func TestMatrix_SetFunc(t *testing.T) {
	tests := []struct {
		inTensor paddle.Tensor
		inOffset []int // rowStart, rowEnd, colStart, colEnd
		inFunc   func(float32) float32
		wantData []float32
	}{
		{
			inTensor: paddle.Tensor{
				Shape: []int32{5, 5},
				Data: []float32{
					1, 2, 3, 4, 5,
					1, 2, 3, 4, 5,
					1, 2, 3, 4, 5,
					1, 2, 3, 4, 5,
					1, 2, 3, 4, 5,
				},
			},
			inOffset: []int{1, 4, 1, 4},
			inFunc: func(v float32) float32 {
				return v * v
			},
			wantData: []float32{
				1, 2, 3, 4, 5,
				1, 4, 9, 16, 5,
				1, 4, 9, 16, 5,
				1, 4, 9, 16, 5,
				1, 2, 3, 4, 5,
			},
		},
	}
	for _, tt := range tests {
		m := paddle.NewMatrixFromTensor[float32](tt.inTensor)
		rowStart, rowEnd, colStart, colEnd := tt.inOffset[0], tt.inOffset[1], tt.inOffset[2], tt.inOffset[3]
		gotData := m.SetFunc(rowStart, rowEnd, colStart, colEnd, tt.inFunc).RawData()
		if !cmp.Equal(gotData, tt.wantData) {
			diff := cmp.Diff(gotData, tt.wantData)
			t.Errorf("Want - Got: %s", diff)
		}
	}
}

func TestMatrix_Pad(t *testing.T) {
	tests := []struct {
		inTensor   paddle.Tensor
		inPadDims  []int // row, col
		inPadValue float32
		wantData   []float32
	}{
		{
			inTensor: paddle.Tensor{
				Shape: []int32{5, 5},
				Data: []float32{
					1, 2, 3, 4, 5,
					1, 2, 3, 4, 5,
					1, 2, 3, 4, 5,
					1, 2, 3, 4, 5,
					1, 2, 3, 4, 5,
				},
			},
			inPadDims:  []int{3, 3},
			inPadValue: 0,
			wantData: []float32{
				1, 2, 3, 4, 5, 0, 0, 0,
				1, 2, 3, 4, 5, 0, 0, 0,
				1, 2, 3, 4, 5, 0, 0, 0,
				1, 2, 3, 4, 5, 0, 0, 0,
				1, 2, 3, 4, 5, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0,
			},
		},
	}
	for _, tt := range tests {
		m := paddle.NewMatrixFromTensor[float32](tt.inTensor)
		row, col := tt.inPadDims[0], tt.inPadDims[1]
		gotData := m.Pad(row, col, tt.inPadValue).RawData()
		if !cmp.Equal(gotData, tt.wantData) {
			diff := cmp.Diff(gotData, tt.wantData)
			t.Errorf("Want - Got: %s", diff)
		}
	}
}
