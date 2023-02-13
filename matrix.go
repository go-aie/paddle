package paddle

import (
	"gonum.org/v1/gonum/mat"
)

type Matrix[E Number] struct {
	m *mat.Dense
}

func NewMatrix[E Number](t Tensor) *Matrix[E] {
	tt := NewTypedTensor[E](t)
	if len(tt.Shape) != 2 {
		panic("t is not a matrix")
	}

	m := mat.NewDense(int(tt.Shape[0]), int(tt.Shape[1]), NumberToFloat64(tt.Data))
	return &Matrix[E]{m: m}
}

func (m *Matrix[E]) Row(i int) []E {
	r := mat.Row(nil, i, m.m)
	return Float64ToNumber[E](r)
}

func (m *Matrix[E]) Col(j int) []E {
	c := mat.Col(nil, j, m.m)
	return Float64ToNumber[E](c)
}

func (m *Matrix[E]) Rows() [][]E {
	var rows [][]E

	r, _ := m.m.Dims()
	for i := 0; i < r; i++ {
		rows = append(rows, m.Row(i))
	}

	return rows
}

func (m *Matrix[E]) Cols() [][]E {
	var cols [][]E

	_, c := m.m.Dims()
	for j := 0; j < c; j++ {
		cols = append(cols, m.Col(j))
	}

	return cols
}

func (m *Matrix[E]) Norm() *Matrix[E] {
	r, c := m.m.Dims()

	norm := m.m.Norm(2) // the square root of the sum of the squares of the elements
	data := make([]float64, r*c)
	for i := 0; i < len(data); i++ {
		data[i] = norm
	}

	normDense := mat.NewDense(r, c, data)
	m.m.DivElem(m.m, normDense)

	return m
}

func (m *Matrix[E]) RawData() []E {
	return Float64ToNumber[E](m.m.RawMatrix().Data)
}
